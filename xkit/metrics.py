from __future__ import annotations

from pathlib import Path
from statistics import mean
from .config import XkitConfig
from .store import ensure_store, read_json, read_jsonl


def load_metrics(project_root: Path, config: XkitConfig) -> dict:
    store = ensure_store(project_root, config)
    index = read_json(store / "index.json", {})
    metrics = read_json(store / "metrics.json", {})
    history = read_jsonl(store / "task_history.jsonl")

    retrievals = metrics.get("retrieval_runs", []) or history
    index_runs = metrics.get("index_runs", [])

    avg_savings = mean([r.get("estimated_savings_pct", 0) for r in retrievals]) if retrievals else 0
    avg_retrieved_tokens = mean([r.get("retrieved_token_estimate", 0) for r in retrievals]) if retrievals else 0
    avg_duration = mean([r.get("duration_sec", 0) for r in retrievals]) if retrievals else 0

    last_incremental = next((r for r in reversed(index_runs) if r.get("type") == "incremental"), None)

    return {
        "project": str(project_root),
        "file_count": index.get("file_count", 0),
        "chunk_count": index.get("chunk_count", 0),
        "full_repo_token_estimate": index.get("full_repo_token_estimate", 0),
        "index_runs": len(index_runs),
        "retrieval_runs": len(retrievals),
        "average_retrieved_tokens": round(avg_retrieved_tokens, 1),
        "average_estimated_savings_pct": round(avg_savings, 2),
        "average_retrieval_duration_sec": round(avg_duration, 3),
        "last_incremental_update": last_incremental,
        "recent_tasks": retrievals[-5:],
    }


def format_metrics_report(data: dict) -> str:
    lines = [
        "# Xkit Metrics",
        "",
        f"Project: `{data['project']}`",
        "",
        "## Index",
        f"- Files indexed: {data['file_count']:,}",
        f"- Chunks indexed: {data['chunk_count']:,}",
        f"- Full repo token estimate: {data['full_repo_token_estimate']:,}",
        f"- Index/update runs: {data['index_runs']:,}",
        "",
        "## Retrieval Performance",
        f"- Retrieval runs: {data['retrieval_runs']:,}",
        f"- Average retrieved tokens: {data['average_retrieved_tokens']:,}",
        f"- Average estimated token savings: {data['average_estimated_savings_pct']}%",
        f"- Average retrieval duration: {data['average_retrieval_duration_sec']} sec",
    ]
    inc = data.get("last_incremental_update")
    if inc:
        lines.extend([
            "",
            "## Last Incremental Update",
            f"- Changed files: {inc.get('changed_file_count', 0)}",
            f"- Deleted files: {inc.get('deleted_file_count', 0)}",
            f"- Re-indexed chunks: {inc.get('reembedded_chunk_count', 0)}",
            f"- Duration: {inc.get('duration_sec', 0)} sec",
        ])
    if data.get("recent_tasks"):
        lines.extend(["", "## Recent Tasks"])
        for task in data["recent_tasks"]:
            lines.append(
                f"- {task.get('task', '')[:80]} | retrieved {task.get('retrieved_token_estimate', 0):,} / "
                f"full {task.get('full_repo_token_estimate', 0):,} tokens | saved {task.get('estimated_savings_pct', 0)}%"
            )
    return "\n".join(lines) + "\n"
