from __future__ import annotations

import time
from pathlib import Path

from .config import XkitConfig
from .retrieval import create_retriever
from .store import append_jsonl, ensure_store, read_json, read_jsonl, write_json
from .token_estimator import estimate_tokens


def retrieve_context(project_root: Path, task: str, config: XkitConfig, top_k: int, budget_tokens: int) -> dict:
    started = time.time()
    store = ensure_store(project_root, config)
    index = read_json(store / "index.json", None)
    if not index:
        raise RuntimeError("Project is not indexed yet. Run: xkit index <project>")

    chunks = read_jsonl(store / "chunks.jsonl")
    retriever = create_retriever(
        chunks,
        method=config.retriever,
        model_name=config.embedding_model,
        device=config.embedding_device,
        project_root=project_root,
        index_dir_name=config.index_dir_name,
    )
    candidates = retriever.search(task, top_k=top_k * 3)

    def _entry(chunk: dict, score: float, text: str, chunk_tokens: int) -> dict:
        return {
            "score": round(score, 4),
            "file": chunk["file"],
            "symbol": chunk.get("symbol"),
            "kind": chunk.get("kind"),
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "token_estimate": chunk_tokens,
            "text": text,
        }

    # Pass 1: greedily add whole chunks that fit the budget, best score first.
    selected = []
    used_tokens = estimate_tokens(task)
    for result in candidates:
        chunk = result.chunk
        chunk_tokens = int(chunk.get("token_estimate", estimate_tokens(chunk.get("text", ""))))
        if used_tokens + chunk_tokens > budget_tokens:
            continue  # a smaller candidate further down may still fit
        selected.append(_entry(chunk, result.score, chunk["text"], chunk_tokens))
        used_tokens += chunk_tokens
        if len(selected) >= top_k:
            break

    # Pass 2: if nothing fit at all, truncate the single best chunk so callers
    # still get useful context — the budget remains a hard ceiling either way.
    if not selected and candidates:
        best = candidates[0]
        marker = "\n/* chunk truncated to fit token budget */"
        remaining = max(0, budget_tokens - used_tokens - estimate_tokens(marker))
        text = best.chunk["text"][: max(0, int(remaining * 3.5))]
        while text and estimate_tokens(text) > remaining:
            text = text[: int(len(text) * 0.8)]
        if text:
            text += marker
            chunk_tokens = estimate_tokens(text)
            selected.append(_entry(best.chunk, best.score, text, chunk_tokens))
            used_tokens += chunk_tokens

    full_repo_tokens = int(index.get("full_repo_token_estimate", 0))
    savings_tokens = max(0, full_repo_tokens - used_tokens)
    savings_pct = (savings_tokens / full_repo_tokens * 100) if full_repo_tokens else 0.0

    event = {
        "task": task,
        "top_k": top_k,
        "budget_tokens": budget_tokens,
        "selected_chunks": len(selected),
        "retrieved_token_estimate": used_tokens,
        "full_repo_token_estimate": full_repo_tokens,
        "estimated_saved_tokens": savings_tokens,
        "estimated_savings_pct": round(savings_pct, 2),
        "duration_sec": round(time.time() - started, 3),
        "timestamp": int(time.time()),
    }

    append_jsonl(store / "task_history.jsonl", event)
    metrics = read_json(store / "metrics.json", {})
    metrics.setdefault("retrieval_runs", [])
    metrics["retrieval_runs"].append(event)
    limit = getattr(config, "metrics_history_limit", 200)
    if len(metrics["retrieval_runs"]) > limit:
        metrics["retrieval_runs"] = metrics["retrieval_runs"][-limit:]
    write_json(store / "metrics.json", metrics)

    return {"event": event, "chunks": selected}


def format_context_markdown(result: dict) -> str:
    event = result["event"]
    lines = [
        "# AI Agent Context Pack",
        "",
        "## Task",
        event["task"],
        "",
        "## Token Metrics",
        f"- Full repo estimate: {event['full_repo_token_estimate']:,} tokens",
        f"- Retrieved context estimate: {event['retrieved_token_estimate']:,} tokens",
        f"- Estimated saved tokens: {event['estimated_saved_tokens']:,}",
        f"- Estimated savings: {event['estimated_savings_pct']}%",
        "",
        "## Instructions for Coding Agent",
        "- Use only the context below unless more files are truly required.",
        "- Request exact file line ranges if context is insufficient.",
        "- Return unified diffs only; do not rewrite full files unless necessary.",
        "- Keep explanations short.",
        "",
        "## Retrieved Chunks",
    ]
    for idx, chunk in enumerate(result["chunks"], start=1):
        title = f"{chunk['file']}:{chunk['start_line']}-{chunk['end_line']}"
        if chunk.get("symbol"):
            title += f" ({chunk['symbol']})"
        lines.extend([
            "",
            f"### Chunk {idx}: {title}",
            f"Score: {chunk['score']} | Estimated tokens: {chunk['token_estimate']}",
            "",
            "```",
            chunk["text"],
            "```",
        ])
    return "\n".join(lines) + "\n"
