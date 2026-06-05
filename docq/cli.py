"""Main CLI and interactive experience for docq - fast answers with full document context."""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from .config import DocQConfig
from .indexer import Indexer
from .llm import LLMClient
from .watcher import FolderWatcher

app = typer.Typer(
    name="docq",
    help="Fast local document Q&A. Drop any supported docs in a folder and ask questions. Answers use full relevant document context.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _get_cfg(folder: Optional[Path] = None, ctx: Optional[typer.Context] = None) -> DocQConfig:
    if folder is None and ctx is not None and ctx.obj:
        folder = ctx.obj.get("folder")
    return DocQConfig.from_env(folder)


def _get_indexer(cfg: DocQConfig) -> Indexer:
    return Indexer(cfg)


def _print_header(cfg: DocQConfig, stats: dict) -> None:
    console.rule("[bold cyan]docq[/bold cyan] — fast answers, full context")
    rprint(f"[dim]Folder:[/dim] [bold]{cfg.folder}[/bold]")
    rprint(f"[dim]Indexed:[/dim] [green]{stats['documents']}[/green] docs, [green]{stats['chunks']}[/green] chunks")
    rprint(f"[dim]LLM:[/dim] [yellow]{cfg.llm_model}[/yellow] @ {cfg.llm_base_url}")
    console.rule()


@app.command()
def index(
    ctx: typer.Context,
    folder: Optional[Path] = typer.Option(None, "--folder", "-f", help="Folder containing documents (defaults to cwd or $DOCQA_FOLDER)"),
    force: bool = typer.Option(False, "--force", "-F", help="Reindex everything even if unchanged"),
):
    """Scan folder and (re)build the semantic index. Fast and incremental by default."""
    cfg = _get_cfg(folder, ctx)
    idx = _get_indexer(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning and indexing...", total=None)

        if force:
            # Clear collection for true force
            try:
                idx.client.delete_collection(idx.collection.name)
                idx.collection = idx.client.get_or_create_collection(
                    name=idx.collection.name, metadata={"hnsw:space": "cosine"}
                )
                idx.full_texts.clear()
                idx.doc_metas.clear()
            except Exception:
                pass

        def cb(p: Path, done: int, total: int):
            progress.update(task, description=f"Indexed {done}/{total} — {p.name}")

        n = idx.reindex_all(progress_cb=cb if not force else None)
        # If force we already re-added in the loop inside reindex_all? Wait, for force we cleared, reindex_all still walks.
        if force:
            n = idx.reindex_all(progress_cb=cb)

    stats = idx.stats()
    rprint(f"\n[green]Done.[/green] {stats['documents']} documents, {stats['chunks']} chunks.")


@app.command()
def ask(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Your question about the documents"),
    folder: Optional[Path] = typer.Option(None, "--folder", "-f"),
    k: int = typer.Option(12, "--k", help="How many chunks to consider for retrieval"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable token streaming"),
    show_context: bool = typer.Option(False, "--show-context", "-c", help="Print the full context block sent to LLM"),
):
    """Ask a question. Retrieves relevant full documents + chunks and streams a grounded answer."""
    cfg = _get_cfg(folder, ctx)
    if k:
        cfg.top_k = k

    idx = _get_indexer(cfg)
    stats = idx.stats()
    if stats["documents"] == 0:
        rprint("[yellow]No documents indexed yet. Run:[/yellow] [bold]docq index[/bold]")
        raise typer.Exit(1)

    _print_header(cfg, stats)

    with Progress(SpinnerColumn(), TextColumn("Retrieving with full context..."), transient=True, console=console) as p:
        p.add_task("retrieve", total=None)
        context, sources = idx.get_full_context(question, top_k=cfg.top_k)

    if show_context:
        console.print(Panel(context[:8000] + ("..." if len(context) > 8000 else ""), title="Full Context Sent to LLM", border_style="dim"))

    rprint("[bold blue]Sources considered:[/bold blue]")
    for s in sources:
        rprint(f"  • {s['source']} (score={s['score']:.3f}, matched={s['matched_chunks']}, len={s['full_length']})")

    llm = LLMClient(cfg)

    if not llm.is_available():
        rprint(
            "[yellow]Warning: Could not quickly reach the LLM at "
            f"{cfg.llm_base_url}. First answer may fail or hang.[/yellow]"
        )

    console.print("\n[bold green]Answer:[/bold green]\n")

    if no_stream:
        ans = llm.answer(question, context)
        console.print(Markdown(ans) if ans.strip().startswith("#") or "**" in ans[:100] else ans)
    else:
        # Clear "how long do I wait?" feedback for the common case of slow first token
        status = console.status(
            "[dim]Waiting for LLM... (slow? try a smaller model with $env:DOCQA_LLM_MODEL or use python -m docq search)[/dim]"
        )
        status.start()
        got_token = False
        buf = []
        try:
            for tok in llm.stream_answer(question, context):
                if not got_token:
                    status.stop()
                    got_token = True
                console.print(tok, end="")
                buf.append(tok)
                sys.stdout.flush()
            if not got_token:
                status.stop()
        except Exception:
            try:
                status.stop()
            except Exception:
                pass
            raise
        finally:
            try:
                status.stop()
            except Exception:
                pass
        console.print("\n")

    # Footer with quick tip
    console.rule()
    rprint("[dim]Tip: Use --show-context to inspect what the model saw.  `docq` for interactive mode.[/dim]")


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Semantic search query"),
    folder: Optional[Path] = typer.Option(None, "--folder", "-f"),
    k: int = typer.Option(8, "--k"),
    full: bool = typer.Option(True, "--full/--chunks", help="Show full docs (default) or just chunks"),
):
    """Pure retrieval: show the most relevant full documents or chunks. No LLM."""
    cfg = _get_cfg(folder, ctx)
    cfg.top_k = k
    idx = _get_indexer(cfg)

    context, sources = idx.get_full_context(query)
    if not sources:
        rprint("[yellow]Nothing relevant found.[/yellow]")
        return

    console.rule(f"[cyan]Top relevant for:[/cyan] {query}")
    for s in sources:
        rprint(f"[bold]{s['source']}[/bold] (score {s['score']:.3f})")
    console.rule()

    if full:
        console.print(context)
    else:
        # Re-do a chunk only view
        res = idx.collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
        for docs, metas in zip(res["documents"], res["metadatas"]):
            for d, m in zip(docs, metas):
                rprint(Panel(d, title=f"{m['source']}#{m.get('chunk_index', '')}", border_style="dim"))


def _interactive(cfg: DocQConfig, idx: Indexer, watcher: Optional[FolderWatcher] = None):
    _print_header(cfg, idx.stats())
    rprint("[dim]Type questions. Commands: /index /reindex /list /stats /sources <q> /model <name> /gist <q> /config /help /quit[/dim]")
    rprint("[dim]Tip for speed: /model qwen3.5:2b   or   /gist your question   (instant, no LLM)[/dim]\n")

    llm = LLMClient(cfg)

    if not llm.is_available():
        rprint(
            "[yellow]Warning: Could not quickly reach the LLM server at "
            f"{cfg.llm_base_url} with model '{cfg.llm_model}'.[/yellow]\n"
            "[yellow]You can still use /search or retrieval commands. "
            "For answers, start Ollama (`ollama serve`) and pull the model, or change DOCQA_LLM_* env vars.[/yellow]\n"
        )

    while True:
        try:
            q = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            rprint("\n[bold]bye[/bold]")
            break

        if not q:
            continue
        if q.startswith("/"):
            cmd = q.lower()
            if cmd in {"/q", "/quit", "/exit"}:
                break
            elif cmd == "/index":
                with Progress(SpinnerColumn(), TextColumn("Indexing changes..."), console=console, transient=True) as pr:
                    pr.add_task("", total=None)
                    n = idx.reindex_all()
                rprint(f"[green]Indexed/updated {n} files.[/green]")
                _print_header(cfg, idx.stats())
            elif cmd == "/reindex":
                rprint("[yellow]Full reindex...[/yellow]")
                try:
                    idx.client.delete_collection(idx.collection.name)
                    idx.collection = idx.client.get_or_create_collection(
                        name=idx.collection.name, metadata={"hnsw:space": "cosine"}
                    )
                except Exception:
                    pass
                idx.full_texts.clear()
                idx.doc_metas.clear()
                n = idx.reindex_all()
                rprint(f"[green]Full reindex complete: {n} files.[/green]")
            elif cmd == "/list":
                t = Table(title="Indexed Documents")
                t.add_column("File", style="cyan")
                t.add_column("Chunks", justify="right")
                t.add_column("Size", justify="right")
                for rel, meta in sorted(idx.doc_metas.items()):
                    t.add_row(rel, str(meta.chunk_count), f"{meta.size:,}")
                console.print(t)
            elif cmd == "/stats":
                s = idx.stats()
                rprint(s)
            elif cmd.startswith("/sources "):
                subq = cmd[len("/sources "):].strip()
                if subq:
                    _, srcs = idx.get_full_context(subq, max_docs=10)
                    for s in srcs:
                        rprint(f"• {s['source']} score={s['score']:.3f}")
            elif cmd == "/config":
                rprint(cfg)
            elif cmd.startswith("/model "):
                new_model = cmd[len("/model "):].strip()
                if new_model:
                    cfg.llm_model = new_model
                    llm = LLMClient(cfg)
                    rprint(f"[green]Switched LLM to: {new_model}[/green]")
                    if not llm.is_available():
                        rprint("[yellow]Warning: Could not reach this model right now.[/yellow]")
            elif cmd.startswith("/gist "):
                subq = cmd[len("/gist "):].strip()
                if subq:
                    context, srcs = idx.get_full_context(subq, max_docs=5)
                    rprint(f"[bold]Fast gist (full relevant context, no LLM):[/bold]\n")
                    for s in srcs:
                        rprint(f"• {s['source']}")
                    console.print("\n" + context[:15000] + ("\n... (truncated)" if len(context) > 15000 else ""))
            elif cmd in {"/h", "/help", "/?"}:
                rprint("Commands: /index /reindex /list /stats /sources <q> /model <name> /gist <q> /config /quit")
                rprint("[dim]/model qwen3.5:2b   → switch to a faster/smaller model[/dim]")
                rprint("[dim]/gist foo bar       → instant full context (no LLM wait)[/dim]")
            else:
                rprint("[red]Unknown command.[/red] Try /help")
            continue

        # Normal question
        try:
            context, sources = idx.get_full_context(q)
        except Exception as e:
            rprint(f"[red]Retrieval error:[/red] {e}")
            continue

        rprint("[dim]Sources:[/dim] " + ", ".join(s["source"] for s in sources[:4]))
        console.print("\n[bold green]Answer:[/bold green]")

        try:
            # Show a clear waiting indicator. First token from local LLM can easily
            # take 10-90 seconds the first time (model loading into RAM, CPU inference, etc.).
            status = console.status(
                "[dim]Waiting for LLM... (slow? try /model qwen3.5:2b or /gist for instant)[/dim]"
            )
            status.start()
            got_token = False
            try:
                # Use smaller max_tokens in interactive for faster responses
                for tok in llm.stream_answer(q, context, max_tokens=600):
                    if not got_token:
                        status.stop()
                        got_token = True
                    console.print(tok, end="")
                    sys.stdout.flush()
                if not got_token:
                    status.stop()
            except Exception:
                try:
                    status.stop()
                except Exception:
                    pass
                raise
            finally:
                try:
                    status.stop()
                except Exception:
                    pass
            console.print("\n")
        except Exception as e:
            rprint(f"[red]LLM error:[/red] {e}")

        console.rule(style="dim")


@app.command()
def watch(
    ctx: typer.Context,
    folder: Optional[Path] = typer.Option(None, "--folder", "-f"),
    no_llm_check: bool = typer.Option(False, "--no-llm-check"),
):
    """Start live watcher + interactive Q&A. New/changed/deleted docs are auto-indexed."""
    cfg = _get_cfg(folder, ctx)
    idx = _get_indexer(cfg)

    # Initial index if empty
    if idx.stats()["documents"] == 0:
        rprint("[yellow]First run — building initial index...[/yellow]")
        n = idx.reindex_all()
        rprint(f"[green]Indexed {n} documents.[/green]")

    watcher = FolderWatcher(
        idx,
        on_change=lambda action, p: console.print(f"[dim][watch] {action}: {p.name}[/dim]")
    )
    watcher.run_in_thread()

    rprint("[green]Watcher active.[/green] Changes to supported files will be indexed automatically.\n")

    if not no_llm_check:
        llm = LLMClient(cfg)
        # fire and forget - don't block interactive
        pass

    try:
        _interactive(cfg, idx, watcher)
    finally:
        watcher.stop_thread()


@app.command()
def serve(
    ctx: typer.Context,
    folder: Optional[Path] = typer.Option(None, "--folder", "-f"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
):
    """Run a minimal HTTP API (FastAPI if installed) for /ask. For advanced use, extend this."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        rprint("[red]serve requires fastapi + uvicorn.[/red] pip install fastapi uvicorn")
        raise typer.Exit(1)

    cfg = _get_cfg(folder, ctx)
    idx = _get_indexer(cfg)
    llm = LLMClient(cfg)

    api = FastAPI(title="docq", version="0.1")

    class AskReq(BaseModel):
        question: str
        k: int | None = None
        stream: bool = True

    @api.get("/stats")
    def stats():
        return idx.stats()

    @api.post("/ask")
    def ask_endpoint(req: AskReq):
        ctx, srcs = idx.get_full_context(req.question, top_k=req.k or cfg.top_k)
        if not req.stream:
            ans = llm.answer(req.question, ctx)
            return {"answer": ans, "sources": srcs}

        def gen():
            yield '{"answer":"'
            for tok in llm.stream_answer(req.question, ctx):
                # naive json escape for demo
                safe = tok.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                yield safe
            yield '","sources":' + __import__("json").dumps(srcs) + "}"
        return StreamingResponse(gen(), media_type="application/json")

    rprint(f"Starting API on http://{host}:{port}  (POST /ask)")
    uvicorn.run(api, host=host, port=port)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    folder: Optional[Path] = typer.Option(
        None, "--folder", "-f",
        is_eager=True,
        help="Documents folder (can be placed before any subcommand)"
    ),
    version: bool = typer.Option(False, "--version", help="Show version"),
):
    """docq — when run with no subcommand, starts interactive Q&A (with live watcher)."""
    if version:
        from . import __version__
        rprint(f"docq {__version__}")
        raise typer.Exit(0)

    if folder:
        ctx.obj = {"folder": folder}

    if ctx.invoked_subcommand is None:
        # Default behavior: interactive (like `docq watch` but clean)
        cfg = _get_cfg(folder, ctx)
        idx = _get_indexer(cfg)

        if idx.stats()["documents"] == 0:
            rprint("[yellow]No index yet. Building one... (use `docq index` to control)[/yellow]")
            idx.reindex_all()

        # Start watcher in background for nice UX
        watcher = FolderWatcher(
            idx,
            on_change=lambda a, p: None,  # silent in interactive
        )
        watcher.run_in_thread()

        try:
            _interactive(cfg, idx, watcher)
        finally:
            watcher.stop_thread()


# Allow `python -m docq`
if __name__ == "__main__":
    app()
