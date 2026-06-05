"""Configuration and constants for docq."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

# Default folder to watch is current working dir unless overridden
DEFAULT_FOLDER = Path.cwd()

# Storage inside the folder (hidden)
DOCQA_DIR_NAME = ".docqa"
CHROMA_SUBDIR = "chroma"
FULL_TEXTS_FILE = "full_texts.json"  # simple persist for full doc texts
CONFIG_FILE = "config.json"

# Supported text-extractable extensions (lowercase)
SUPPORTED_EXTS: Set[str] = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".pdf",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv",
    ".json", ".jsonl",
    ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf",
    ".sql", ".r", ".m", ".scala", ".swift", ".kt",
}

# Files / dirs to always ignore (relative names)
IGNORE_NAMES: Set[str] = {
    ".git", ".svn", ".hg",
    ".docqa", "__pycache__", ".venv", "venv", "node_modules",
    ".idea", ".vscode",
    ".DS_Store", "Thumbs.db",
}

# Ignore by glob patterns (simple endswith for speed)
IGNORE_GLOBS = ("*.tmp", "*.temp", "*.bak", "*.swp", "~$*", "*.lock", "*.exe", "*.dll", "*.zip", "*.tar", "*.gz", "*.rar", "*.7z", "*.bin", "*.dat", "*.db", "*.sqlite", "*.parquet")

# Chunking params (tuned for good retrieval + context stuffing)
CHUNK_SIZE = 1200  # chars ~ 250-350 tokens
CHUNK_OVERLAP = 200

# Retrieval
DEFAULT_TOP_K_CHUNKS = 12
MAX_CONTEXT_CHARS = 24000  # plenty for "full context" on relevant docs; LLM context window is separate concern
MAX_DOCS_IN_CONTEXT = 6  # top relevant full docs to stuff (prevents prompt blowup)

# LLM defaults (user overrides via env or CLI)
DEFAULT_LLM_MODEL = "qwen3.5:2b"  # Good balance. For max speed use "qwen3.5:2b" or "qwen2.5-coder:1.5b-base" (much faster on CPU).
DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama OpenAI compat
DEFAULT_API_KEY = "ollama"  # dummy for most local servers

SYSTEM_PROMPT = """You are a precise, helpful assistant with access to the user's private documents in a local folder.

CRITICAL RULES:
- Ground every factual claim in the provided CONTEXT only.
- If the answer is not in the CONTEXT, say: "I don't have that information in the available documents."
- Quote short relevant excerpts (use "..." for elision) and always cite the source document filename.
- Prefer full sentences from the docs.
- Be concise but complete. Use bullet points or numbered lists when helpful.
- Never invent details, numbers, names, or dates.
- If multiple docs conflict, note the conflict and cite both.
"""

USER_PROMPT_TEMPLATE = """CONTEXT (full relevant documents or key excerpts):

{context}

---

QUESTION: {question}

Answer the question using ONLY the CONTEXT above. Cite sources inline by filename. If insufficient information, state it clearly.
"""

@dataclass
class DocQConfig:
    folder: Path = field(default_factory=lambda: Path.cwd())
    chroma_persist_dir: Path = field(default_factory=lambda: Path.cwd() / DOCQA_DIR_NAME / CHROMA_SUBDIR)
    full_texts_path: Path = field(default_factory=lambda: Path.cwd() / DOCQA_DIR_NAME / FULL_TEXTS_FILE)
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_BASE_URL
    llm_api_key: str = DEFAULT_API_KEY
    top_k: int = DEFAULT_TOP_K_CHUNKS
    max_context_chars: int = MAX_CONTEXT_CHARS
    max_docs_in_context: int = MAX_DOCS_IN_CONTEXT
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    @classmethod
    def from_env(cls, folder: Path | None = None) -> "DocQConfig":
        folder = folder or Path(os.environ.get("DOCQA_FOLDER", Path.cwd())).resolve()
        base = folder / DOCQA_DIR_NAME
        return cls(
            folder=folder,
            chroma_persist_dir=base / CHROMA_SUBDIR,
            full_texts_path=base / FULL_TEXTS_FILE,
            llm_model=os.environ.get("DOCQA_LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_base_url=os.environ.get("DOCQA_LLM_BASE_URL", DEFAULT_BASE_URL),
            llm_api_key=os.environ.get("DOCQA_LLM_API_KEY", DEFAULT_API_KEY),
            top_k=int(os.environ.get("DOCQA_TOP_K", DEFAULT_TOP_K_CHUNKS)),
            max_context_chars=int(os.environ.get("DOCQA_MAX_CONTEXT_CHARS", MAX_CONTEXT_CHARS)),
            max_docs_in_context=int(os.environ.get("DOCQA_MAX_DOCS", MAX_DOCS_IN_CONTEXT)),
        )

    def ensure_dirs(self) -> None:
        self.chroma_persist_dir.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.full_texts_path.parent.mkdir(parents=True, exist_ok=True)
