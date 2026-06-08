from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

try:
    from prometheus_client import start_http_server, Gauge
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False


_gauges: dict[str, Gauge] = {}


def _ensure_gauge(name: str, documentation: str) -> Optional[Gauge]:
    if not _PROM_AVAILABLE:
        return None
    if name in _gauges:
        return _gauges[name]
    g = Gauge(name, documentation)
    _gauges[name] = g
    return g


def _load_metrics_file(path: Path) -> dict:
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def start_exporter(project_root: Path, port: int = 8000, interval: float = 5.0):
    """Start a simple Prometheus exporter that reads `metrics.json` periodically.

    This is intentionally lightweight: it reads the existing metrics file and
    exposes a few gauges useful for monitoring indexing/retrieval activity.
    """
    if not _PROM_AVAILABLE:
        raise RuntimeError("prometheus-client not installed")

    store = project_root / ".xkit"
    metrics_path = store / "metrics.json"

    start_http_server(port)

    def loop():
        while True:
            m = _load_metrics_file(metrics_path) if metrics_path.exists() else {}
            idx_runs = m.get("index_runs", [])
            retr_runs = m.get("retrieval_runs", [])
            emb_runs = m.get("embedding_runs", [])
            _ensure_gauge("xkit_index_runs_total", "Number of index runs").set(len(idx_runs) if idx_runs is not None else 0)
            _ensure_gauge("xkit_retrieval_runs_total", "Number of retrieval runs").set(len(retr_runs) if retr_runs is not None else 0)
            # Export last embedding upsert duration and last chunk count
            if emb_runs:
                last = emb_runs[-1]
                _ensure_gauge("xkit_embedding_last_duration_seconds", "Duration of last embedding upsert (s)").set(float(last.get("duration_sec", 0.0)))
                _ensure_gauge("xkit_embedding_last_chunk_count", "Chunk count in last embedding upsert").set(int(last.get("chunk_count", 0)))
            else:
                _ensure_gauge("xkit_embedding_last_duration_seconds", "Duration of last embedding upsert (s)").set(0.0)
                _ensure_gauge("xkit_embedding_last_chunk_count", "Chunk count in last embedding upsert").set(0)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


__all__ = ["start_exporter"]
