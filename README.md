# Xkit

A local CLI tool that lowers coding-agent token usage by indexing a project once, chunking files intelligently, retrieving only relevant context per task, and updating only changed files after fixes.

It includes a metrics system so you can compare:

- full-repo token estimate
- retrieved-context token estimate
- estimated token savings
- indexed files/chunks
- changed-file reindexing
- retrieval time
- task history

No paid API is required for the default prototype. It uses a local TF-IDF retrieval index so you can test the workflow before connecting OpenAI embeddings or another vector database.

## Features

- **Config file support** — Per-project `.xkit/config.toml` for customizing chunking, retrieval, and ignored directories
- **Semantic search** — Optional sentence-transformers embeddings for concept-aware retrieval (falls back to TF-IDF if not installed)
- **AST-aware chunking** — Optional tree-sitter integration for language-aware code splitting (falls back to regex-based symbol detection)
- **Progress bars** — Real-time progress during indexing and updates
- **Test suite** — 37 pytest tests covering all modules
- **Zero dependencies** — Core functionality works with Python stdlib only

## Install

```bash
cd Xkit
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Index a project:

```bash
xkit index /path/to/project
```

Retrieve context for a task:

```bash
xkit retrieve /path/to/project "fix login redirect after refresh" --top-k 8 --budget 12000
```

Show metrics:

```bash
xkit metrics /path/to/project
```

Update only changed files:

```bash
xkit update /path/to/project
```

Export retrieved context for an AI agent:

```bash
xkit retrieve /path/to/project "fix Stripe subscription activation" --format markdown > context.md
```

## Design

The tool creates a `.xkit/` folder inside your project:

```text
.xkit/
  index.json
  chunks.jsonl
  metrics.json
  task_history.jsonl
```

The core loop is:

```text
index once → retrieve per task → send small context to AI agent → apply patch → update changed files only → track savings
```

## Why this lowers cost

Instead of sending the whole repo to an AI coding agent, you send only the top relevant chunks and summaries. This can greatly reduce input tokens and prevent context-window overflow.

## Workflow with a Coding Agent

Here's the practical workflow for using Xkit with AI coding agents (like Cline, Claude Code, Copilot, etc.):

### Option A: You manually feed context to the agent

```
1. xkit index my-project          # Index once
2. xkit retrieve my-project "..." > context.md   # Get relevant context
3. Paste context.md into your prompt to the agent
4. Agent applies changes
5. xkit update my-project         # Re-index only changed files
```

### Option B: The agent uses Xkit itself (recommended)

If your coding agent can run shell commands (like Cline does), you can instruct it to use Xkit as part of its workflow. Add something like this to your agent's instructions or system prompt:

> **For the AI agent:**
> Before working on a task, run `xkit retrieve /path/to/project "<task description>" --format markdown` to get relevant context. Use only the returned code chunks unless you truly need more files. After making changes, run `xkit update /path/to/project` to keep the index fresh.

The agent would then:
1. Call `xkit retrieve` to get a compact context pack (saving tokens vs. reading the whole repo)
2. Work with the returned chunks to understand the relevant code
3. Make changes
4. Call `xkit update` so future retrievals are accurate

### Option C: Pipe context directly into your prompt

```bash
xkit retrieve my-project "fix login redirect" | your-agent-cli --prompt-stdin
```

### Real-world token savings example

On the included demo project:
- Full repo: **424 tokens**
- Retrieved context for "fix login redirect after refresh": **328 tokens**
- **22.6% savings** on a tiny project

On a real codebase (e.g., 50,000+ tokens), the savings are much more dramatic — often **80-95%** since you only send the 5-10 most relevant chunks instead of every file.

