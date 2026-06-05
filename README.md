# docq — Fast local document Q&A with full context

Drop **any documents** into a folder. Ask questions. Get **fast, grounded answers** that see the **full relevant documents** (not tiny snippets).

- Semantic + keyword-friendly retrieval (via embeddings)
- **Full document context stuffing** for the most relevant files (the key to high-quality answers)
- Live folder watching — new, edited, or deleted files are auto-indexed
- Works with **any OpenAI-compatible LLM server** (Ollama, LM Studio, llama.cpp server, vLLM, Groq, OpenAI, xAI, etc.)
- Supports PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, source code, HTML, JSON, logs, and more
- Pure Python, fast incremental indexing, nice rich CLI + streaming answers

## Why "full context"?

Most RAG tools only feed the LLM small chunks. docq:

1. Retrieves the best chunks semantically.
2. Identifies the **most relevant source documents**.
3. Feeds the **entire document text** (or large prefix + the actual matching excerpts) for those top documents into the LLM prompt.

Result: the model sees the complete picture for the docs that matter → fewer hallucinations, better synthesis, accurate citations.

## Quick start

```bash
# 1. Clone / cd into your docs folder (or anywhere)
cd C:\Tools\document-question

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# On PowerShell:
.\venv\Scripts\Activate.ps1
# On CMD:
# venv\Scripts\activate.bat

# 3. Install dependencies + the `docq` command (the important step for the bare `docq` command)
pip install -r requirements.txt
pip install -e .

# 4. (Recommended) Start a fast local LLM (one-time)
# Install Ollama from https://ollama.com then:
ollama pull llama3.2:3b     # or llama3.2:1b for even faster / lower RAM
# Alternatives: phi3:mini, gemma2:2b, qwen2.5:3b etc.

# 5. Drop documents into this folder (or any folder)

# 6. Run interactive mode (auto-indexes + watches for changes)
docq

# or one-shot
docq ask "What were the key decisions in Q3?"

# force full reindex after adding lots of files
docq index --force
```

**Windows / PowerShell tip — "docq is not recognized as a name of a cmdlet"**

This is extremely common. The `pip install -e .` creates a `docq.exe` launcher in `venv\Scripts\`, but if the venv was already active, PowerShell may not see the new .exe until you re-activate the shell.

Fix:
```powershell
# after the pip install -e . above
deactivate
.\venv\Scripts\Activate.ps1
docq --help
```

**Works right now, no re-activation needed:**
```powershell
python -m docq ask "your question"
```

**"How long do I have to wait for an answer?" (LLM is slow / hanging at "Answer:")**

This is normal with local models:

- First token after `ollama pull` or after the model was unloaded can take **15–90+ seconds**.
- The new version of docq now shows an explicit spinner:  
  `Waiting for LLM response (first token can take 10-90s if the model is loading)...`

What to do:
1. Warm up the model once:
   ```powershell
   ollama run llama3.2:3b
   # type anything, wait for reply, then type /bye
   ```
2. Make sure the Ollama server is running (usually the tray app or leave one `ollama run` session open).
3. Restart `docq` completely (`/quit` then `docq` again).
4. For **instant** results with no LLM wait at all, use:
   ```powershell
   python -m docq search "your question"
   ```
   or inside the REPL: `/sources your question`

You can also change the model to something smaller/faster you have installed:
```powershell
$env:DOCQA_LLM_MODEL="qwen3.5:2b"
docq
```


## Usage

### Interactive (best experience)

```bash
docq                 # starts watcher + REPL
docq --folder /path/to/my/docs
```

Inside the REPL:

- Just type your question
- `/index` — incremental update
- `/reindex` — nuclear full rebuild
- `/list` — show indexed files + chunk counts
- `/stats`
- `/sources my query` — see which docs would be used
- `/config`
- `/quit`

Live watcher prints quiet updates when files change.

### Commands

- `docq ask "question here" --show-context` — one shot + see what context was sent
- `docq index` — build / update index (incremental)
- `docq watch` — watcher + interactive (explicit)
- `docq search "foo bar"` — retrieval only (great for copying context to other LLMs)
- `docq serve` — tiny FastAPI server on :8765 (needs `pip install fastapi uvicorn`)

## Configuration (env vars or CLI)

```bash
# Point at a different docs folder
export DOCQA_FOLDER="C:\My\Project\Notes"

# Use a different (faster or smarter) model / server
export DOCQA_LLM_MODEL="llama3.2:1b"
export DOCQA_LLM_BASE_URL="http://localhost:11434/v1"
export DOCQA_LLM_API_KEY="ollama"     # dummy value is fine for local

# Or point at LM Studio, llama.cpp, OpenAI, Groq, etc.
export DOCQA_LLM_BASE_URL="http://localhost:1234/v1"
export DOCQA_LLM_MODEL="local-model"

# Retrieval tuning
export DOCQA_TOP_K=15
export DOCQA_MAX_CONTEXT_CHARS=30000
```

You can also pass `--folder` to most commands.

## Supported documents (fast extractors)

- Text: `.txt .md .rst .log .csv .json .jsonl`
- Office: `.pdf .docx .pptx .xlsx .xls`
- Web/Code: `.html .htm` + 30+ source extensions (`.py .js .ts .go .rs ...`)
- Everything is extracted to clean text + page/slide/sheet markers where applicable.

Unknown extensions or binary files are ignored. Corrupt files are skipped gracefully.

## Performance & speed notes

- Indexing is incremental (hashes + mtime). Only changed files are re-parsed + re-embedded.
- Chroma + MiniLM embeddings are fast on CPU. First run downloads the ~90 MB embedding model.
- Answers feel instant because:
  - Retrieval is < 50 ms even for thousands of chunks
  - We stream tokens from the LLM as soon as they arrive
- For maximum speed on CPU use 1B–3B models (`llama3.2:1b`, `gemma2:2b`).
- "Full context" budget is generous (~24k chars of relevant docs by default) but stays inside most 8k–128k context windows.

## Architecture (for hackers)

```
docq/
  parsers.py     # robust extractors (no heavy OCR)
  chunking.py    # simple overlapping char chunks (excellent recall)
  indexer.py     # ChromaDB + full text store + doc meta + get_full_context()
  llm.py         # OpenAI client wrapper + streaming + strong grounding prompt
  watcher.py     # watchdog observer in thread
  cli.py         # typer + rich TUI/REPL
  config.py      # everything configurable via env
```

The secret sauce is in `Indexer.get_full_context()` — it turns top-k chunks into a short list of whole documents that get stuffed.

## Tips for best answers

1. Use reasonably sized source documents (5–50 pages ideal). Huge books still work but you may hit token limits.
2. Put related docs together in one folder (or use multiple `docq --folder` sessions).
3. For very large collections, increase `DOCQA_TOP_K` and rely on the ranking inside `get_full_context`.
4. If the model says "I don't have that information", use `/sources your question` or `docq search` to debug retrieval.
5. Want even higher quality? Use a stronger model (llama3.1:8b, qwen2.5:7b, or a fine-tune) — the context quality is already excellent.

## Troubleshooting

- **No LLM answers** — make sure Ollama/LMStudio is running and the model is pulled. `docq ask "hi"` will show the error.
- **Slow first run** — embedding model + first LLM download. Subsequent runs are fast.
- **Bad PDF text** — some PDFs are scanned images. docq does not do OCR (by design — keeps it fast and dep-light).
- **Windows paths** — everything uses pathlib; should just work.
- **Permission errors** — don't put the folder inside protected system dirs.

## Development

```bash
pip install -e .
# run from source
python -m docq --help
docq --help
```


### Speed tips (when Ollama feels slow)

Local LLMs on CPU can be slow for the first token (10-60s+). `docq` retrieval is instant.

In the interactive REPL use these **zero-wait** or fast commands:

- `/gist your question here` — instantly returns the full relevant document text(s) with no LLM call at all. Perfect for gists/summaries.
- `/model qwen3.5:2b` (or any model from `ollama list`) — switch to a smaller/faster model live.
- `/sources your question` — see what docs would be used.

From shell:
```powershell
python -m docq search "your question"     # always fast
```

Set a fast default model permanently:
```powershell
$env:DOCQA_LLM_MODEL="qwen3.5:2b"
docq
```

You have several small models (`qwen2.5-coder:1.5b-base` is tiny and very fast).

## License

MIT. Build cool things with your private data.

---

Made for people who want **their documents to answer questions instantly and accurately**, locally, with maximum context.
