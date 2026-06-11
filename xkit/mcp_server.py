"""MCP server exposing Xkit to coding agents (Claude Code, Cline, Cursor, ...).

Run with:  xkit mcp /path/to/project
Requires:  pip install "xkit[mcp]"   (the official `mcp` Python SDK)

Example Claude Code registration:
  claude mcp add xkit -- xkit mcp /path/to/project

Tools exposed:
  - retrieve_context: get a compact, ranked context pack for a task
  - update_index:     re-index only changed files after edits
  - index_project:    full (re)index
  - get_metrics:      token-savings and retrieval metrics
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_context import format_context_markdown, retrieve_context
from .config import XkitConfig
from .indexer import build_index, update_changed_files
from .metrics import format_metrics_report, load_metrics


def _require_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError as e:
        raise ImportError(
            "The MCP server requires the `mcp` package. Install with: pip install 'xkit[mcp]'"
        ) from e


def create_server(project_root: Path):
    """Build the FastMCP server bound to a project root."""
    FastMCP = _require_mcp()
    root = project_root.resolve()
    mcp = FastMCP(
        "xkit",
        instructions=(
            "Xkit provides token-efficient code retrieval for this project. "
            "Before working on a task, call retrieve_context with a concise task "
            "description to get only the relevant code chunks instead of reading "
            "many files. After making changes, call update_index so future "
            "retrievals stay accurate."
        ),
    )

    def _config() -> XkitConfig:
        return XkitConfig.load(root)

    @mcp.tool(name="retrieve_context")
    def retrieve_context_tool(
        task: str,
        top_k: int = 0,
        budget_tokens: int = 0,
        format: str = "markdown",
    ) -> str:
        """Retrieve the most relevant code chunks for a coding task.

        Args:
            task: Natural-language description of the task (e.g. "fix login
                redirect after refresh" or "where is Stripe webhook handled").
            top_k: Max number of chunks to return (0 = project default).
            budget_tokens: Hard ceiling on estimated context tokens
                (0 = project default).
            format: "markdown" for an agent-ready context pack, "json" for
                structured chunk data.
        """
        config = _config()
        result = retrieve_context(
            root,
            task,
            config,
            top_k or config.default_top_k,
            budget_tokens or config.default_budget_tokens,
        )
        if format == "json":
            return json.dumps(result, indent=2)
        return format_context_markdown(result)

    @mcp.tool()
    def update_index(show_details: bool = False) -> str:
        """Incrementally re-index files that changed since the last index/update.

        Call this after applying code changes so retrieval stays accurate.
        """
        config = _config()
        index = update_changed_files(root, config)
        summary = {
            "status": "updated",
            "files": index["file_count"],
            "chunks": index["chunk_count"],
            "full_repo_token_estimate": index["full_repo_token_estimate"],
        }
        if show_details:
            summary["files_detail"] = index.get("files", {})
        return json.dumps(summary, indent=2)

    @mcp.tool()
    def index_project() -> str:
        """Build (or rebuild) the full project index from scratch."""
        config = _config()
        index = build_index(root, config)
        return json.dumps(
            {
                "status": "indexed",
                "files": index["file_count"],
                "chunks": index["chunk_count"],
                "full_repo_token_estimate": index["full_repo_token_estimate"],
            },
            indent=2,
        )

    @mcp.tool()
    def get_metrics(format: str = "markdown") -> str:
        """Show retrieval metrics: token estimates, savings, recent tasks."""
        config = _config()
        data = load_metrics(root, config)
        if format == "json":
            return json.dumps(data, indent=2)
        return format_metrics_report(data)

    return mcp


def serve(project_root: Path) -> int:
    """Start the MCP server on stdio (blocking)."""
    server = create_server(project_root)
    server.run(transport="stdio")
    return 0
