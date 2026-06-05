"""Indexing layer: ChromaDB for vectors + full text storage + fast change detection."""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings

from .chunking import chunk_text
from .config import DocQConfig, SUPPORTED_EXTS, IGNORE_NAMES, IGNORE_GLOBS
from .parsers import extract_text


@dataclass
class DocMeta:
    path: str          # relative posix path
    mtime: float
    size: int
    content_hash: str
    chunk_count: int = 0


class Indexer:
    def __init__(self, cfg: DocQConfig):
        self.cfg = cfg
        cfg.ensure_dirs()

        # Chroma persistent client - fast local
        self.client = chromadb.PersistentClient(
            path=str(cfg.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=False)
        )
        # One collection per folder root (name derived to be fs safe)
        coll_name = "docq_" + hashlib.sha1(str(cfg.folder.resolve()).encode()).hexdigest()[:12]
        # Use default embedding (sentence-transformers all-MiniLM-L6-v2 via chromadb)
        self.collection = self.client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.full_texts: Dict[str, str] = {}  # relpath -> full_text
        self.doc_metas: Dict[str, DocMeta] = {}
        self._load_full_texts()
        self._load_doc_metas()

    # ---------- Persistence for full docs + metas ----------
    def _load_full_texts(self) -> None:
        p = self.cfg.full_texts_path
        if p.exists():
            try:
                self.full_texts = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self.full_texts = {}

    def _save_full_texts(self) -> None:
        tmp = self.cfg.full_texts_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.full_texts, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cfg.full_texts_path)

    def _load_doc_metas(self) -> None:
        # Store lightweight meta next to full texts for change detection
        meta_path = self.cfg.full_texts_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                self.doc_metas = {k: DocMeta(**v) for k, v in raw.items()}
            except Exception:
                self.doc_metas = {}

    def _save_doc_metas(self) -> None:
        meta_path = self.cfg.full_texts_path.with_suffix(".meta.json")
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.doc_metas.items()}, indent=2), encoding="utf-8")
        tmp.replace(meta_path)

    # ---------- File utilities ----------
    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.cfg.folder.resolve()).as_posix()
        except Exception:
            return p.name  # fallback

    def _should_ignore(self, p: Path) -> bool:
        # Check any path component (so we prune .venv, .git anywhere in tree)
        for part in p.parts:
            if part in IGNORE_NAMES:
                return True
        name = p.name
        for g in IGNORE_GLOBS:
            if g.startswith("*.") and name.endswith(g[1:]):
                return True
            if g.endswith("*") and name.startswith(g[:-1]):
                return True
        return False

    def _content_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def _is_supported(self, p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not self._should_ignore(p)

    # ---------- Core index ops ----------
    def scan_files(self) -> List[Path]:
        """Return all currently supported files under folder (recursive)."""
        files: List[Path] = []
        root = self.cfg.folder
        for p in root.rglob("*"):
            if self._is_supported(p):
                files.append(p)
        return files

    def get_file_state(self, p: Path) -> Tuple[float, int, str]:
        st = p.stat()
        data = p.read_bytes()
        return st.st_mtime, st.st_size, self._content_hash(data)

    def needs_reindex(self, rel: str, mtime: float, size: int, h: str) -> bool:
        meta = self.doc_metas.get(rel)
        if not meta:
            return True
        return not (abs(meta.mtime - mtime) < 1e-3 and meta.size == size and meta.content_hash == h)

    def add_or_update_file(self, p: Path) -> bool:
        """Parse, chunk, embed, store. Returns True if changed."""
        if not self._is_supported(p):
            return False
        rel = self._rel(p)
        try:
            mtime, size, h = self.get_file_state(p)
        except Exception:
            return False

        if not self.needs_reindex(rel, mtime, size, h):
            return False  # up to date

        # Remove old chunks for this doc
        self._delete_chunks_for(rel)

        text = extract_text(p)
        if not text or not text.strip():
            # still record meta so we don't re-try constantly
            self.doc_metas[rel] = DocMeta(path=rel, mtime=mtime, size=size, content_hash=h, chunk_count=0)
            self.full_texts[rel] = ""
            self._save_full_texts()
            self._save_doc_metas()
            return True

        self.full_texts[rel] = text

        chunks = list(chunk_text(text, chunk_size=self.cfg.chunk_size, overlap=self.cfg.chunk_overlap))
        if not chunks:
            chunks = [(text[:2000], 0, min(2000, len(text)))]

        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []
        for i, (chunk, cstart, cend) in enumerate(chunks):
            cid = f"{rel}::chunk::{i}"
            ids.append(cid)
            docs.append(chunk)
            metas.append({
                "source": rel,
                "chunk_index": i,
                "start": cstart,
                "end": cend,
                "mtime": mtime,
            })

        # Add in batch (chroma will embed)
        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)

        self.doc_metas[rel] = DocMeta(path=rel, mtime=mtime, size=size, content_hash=h, chunk_count=len(ids))
        self._save_full_texts()
        self._save_doc_metas()
        return True

    def _delete_chunks_for(self, rel: str) -> None:
        try:
            # Chroma where filter on metadata
            self.collection.delete(where={"source": rel})
        except Exception:
            # If nothing to delete or collection empty, ignore
            pass

    def remove_file(self, p: Path) -> None:
        rel = self._rel(p)
        self._delete_chunks_for(rel)
        self.full_texts.pop(rel, None)
        self.doc_metas.pop(rel, None)
        self._save_full_texts()
        self._save_doc_metas()

    def reindex_all(self, progress_cb: Optional[callable] = None) -> int:
        """Force full reindex of current files. Returns count of (re)indexed files."""
        files = self.scan_files()
        count = 0
        for p in files:
            if self.add_or_update_file(p):
                count += 1
            if progress_cb:
                progress_cb(p, count, len(files))
        # Also purge docs that no longer exist on disk
        existing_rels = {self._rel(p) for p in files}
        for rel in list(self.doc_metas.keys()):
            if rel not in existing_rels:
                self._delete_chunks_for(rel)
                self.full_texts.pop(rel, None)
                self.doc_metas.pop(rel, None)
        self._save_full_texts()
        self._save_doc_metas()
        return count

    def stats(self) -> Dict[str, int | str]:
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "documents": len(self.doc_metas),
            "chunks": count,
            "folder": str(self.cfg.folder),
        }

    # ---------- Context assembly for "full context" ----------
    def get_full_context(self, query: str, top_k: Optional[int] = None, max_chars: Optional[int] = None, max_docs: Optional[int] = None) -> Tuple[str, List[dict]]:
        """
        Retrieve top chunks, select most relevant *documents*, return
        (big_context_string_with_full_docs, list_of_source_infos)
        This is the key to "full context" answers.
        """
        top_k = top_k or self.cfg.top_k
        max_chars = max_chars or self.cfg.max_context_chars
        max_docs = max_docs or self.cfg.max_docs_in_context

        if self.collection.count() == 0:
            return "(no documents indexed yet)", []

        # Semantic search
        res = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, 50),
            include=["documents", "metadatas", "distances"]
        )

        # Group chunks by source + keep best distance per doc
        doc_scores: Dict[str, float] = {}
        doc_chunks: Dict[str, List[Tuple[int, str]]] = {}
        for docs, metas, dists in zip(res["documents"], res["metadatas"], res["distances"]):
            for doc, meta, dist in zip(docs, metas, dists):
                src = meta["source"]
                idx = meta.get("chunk_index", 0)
                if src not in doc_scores or dist < doc_scores[src]:
                    doc_scores[src] = dist
                doc_chunks.setdefault(src, []).append((idx, doc))

        if not doc_scores:
            return "(no relevant chunks found)", []

        # Rank docs by best chunk similarity (lower dist = better for cosine)
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1])[:max_docs]

        context_parts: List[str] = []
        used_chars = 0
        sources_info: List[dict] = []

        for src, score in ranked:
            full = self.full_texts.get(src, "")
            if not full:
                # fallback to concatenated chunks
                chs = sorted(doc_chunks.get(src, []))
                full = "\n\n".join(c for _, c in chs)

            # For truly full context on the doc, we include as much as budget allows.
            # Prioritize the beginning + the chunks that matched (simple heuristic).
            header = f"\n\n===== DOCUMENT: {src} (relevance={score:.3f}) =====\n"
            remaining = max_chars - used_chars - len(header)
            if remaining <= 200:
                break

            # Always try to give the entire doc if it fits; otherwise give top chunks + head/tail.
            if len(full) <= remaining:
                body = full
            else:
                # Include the actual matching chunks first (they are gold), then prefix of doc
                matched = "\n\n--- matching excerpts ---\n" + "\n\n".join(c for _, c in sorted(doc_chunks.get(src, []))[:4])
                prefix = full[: int(remaining * 0.6)]
                body = prefix + matched + "\n\n[... document continues ...]"

            part = header + body
            if used_chars + len(part) > max_chars:
                part = part[: max_chars - used_chars]
            context_parts.append(part)
            used_chars += len(part)

            sources_info.append({
                "source": src,
                "score": float(score),
                "matched_chunks": len(doc_chunks.get(src, [])),
                "full_length": len(full),
            })

        context = "".join(context_parts).strip()
        if not context:
            context = "(retrieved no usable context)"
        return context, sources_info
