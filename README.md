# Xkit

Token-efficient code retrieval for AI coding agents. Index a project once, then retrieve only the chunks relevant to each task — instead of letting the agent read file after file.

Works fully offline with zero dependencies (stdlib-only BM25). Optional extras add semantic embeddings (FAISS-backed), AST-aware chunking (tree-sitter), exact token counting (tiktoken), and an **MCP server** so agents like Claude Code, Cline, and Cursor can call Xkit natively.

## Why

Coding agents explore repos with grep/glob/read loops. That works, but every exploration round trip costs tokens and latency. Xkit precomputes a chunked, searchable index so one retrieval call returns a ranked, budget-capped context pack:

```
index once → retrieve per task → agent works from compact context → update changed files → repeat
```

Honest framing: the comparison that matters isn't "vs sending the whole repo" (nobody does that) — it's **fewer agent tool-call round trips and a hard token budget per task**. The built-in metrics track retrieved-vs-full-repo estimates so you can measure your own savings.

## Install

```bash
pip install -e .                 # core: BM25 retrieval, zero dependencies
pip install -e ".[tokens]"       # + tiktoken for accurate token estimates
pip install -e ".[treesitter]"   # + AST-aware chunking (prebuilt grammar wheels)
pip install -e ".[embeddings]"   # + semantic search (sentence-transformers + FAISS)
pip install -e ".[mcp]"          # + MCP server for coding agents
pip install -e ".[all]"          # everything
```

Requires Python 3.11+. Windows: use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Usage

```bash
xkit init  /path/to/project                  # write .xkit/config.toml (optional)
xkit index /path/to/project                  # build the index once
xkit retrieve /path/to/project "fix login redirect after refresh" --top-k 8 --budget 12000
xkit update /path/to/project                 # re-index only changed files
xkit metrics /path/to/project                # token savings & retrieval stats
```

Export a context pack for any agent:

```bash
xkit retrieve /path/to/project "fix Stripe subscription activation" --format markdown > context.md
```

## Use with coding agents

### Option A (recommended): MCP server

```bash
pip install -e ".[mcp]"
claude mcp add xkit -- xkit mcp /path/to/project     # Claude Code
```

The agent gets four native tools: `retrieve_context`, `update_index`, `index_project`, `get_metrics` — no prompt engineering required. Any MCP client (Cline, Cursor, ...) can register the same stdio command.

### Option B: shell-capable agents

Add to the agent's instructions:

> Before working on a task, run `xkit retrieve <project> "<task>" --format markdown` and work from the returned chunks. After making changes, run `xkit update <project>`.

### Option C: pipe into a prompt

```bash
xkit retrieve my-project "fix login redirect" | your-agent-cli --prompt-stdin
```

## Retrieval methods

Set in `.xkit/config.toml` (`retriever = "..."`):

| Method | Deps | Best for |
|---|---|---|
| `bm25` (default) | none | exact identifiers, fast, offline |
| `hybrid` | embeddings extra | best overall: BM25 + embeddings fused with RRF |
| `embeddings` | embeddings extra | conceptual queries ("where do we validate payment events") |
| `tfidf` | none | legacy |

The tokenizer splits `camelCase` and `snake_case`, so "stripe webhook" matches `handleStripeWebhook`. With the embeddings extra installed, vectors are persisted to a FAISS index at index time; queries encode **only the query string** — never the corpus.

## Design

Per-project state lives in `.xkit/`:

```
.xkit/
  config.toml          # optional overrides
  index.json           # file hashes + token estimates
  chunks.jsonl         # symbol-aware chunks
  metrics.json         # capped run history
  task_history.jsonl
  faiss/               # persisted vectors (embeddings extra)
```

Chunking is AST-aware when tree-sitter is available (one chunk per top-level symbol, oversized functions split by lines, configurable overlap) and falls back to regex symbol detection otherwise. Incremental `xkit update` re-chunks only files whose hash changed and removes stale vectors from FAISS.

## Development

```bash
pip install -e ".[test]"
pytest -q
ruff check xkit tests
```

CI runs the suite on Linux/macOS/Windows with both minimal (stdlib-only) and full installs.

## License

MIT
