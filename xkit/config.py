from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class XkitConfig:
    index_dir_name: str = ".xkit"
    max_chunk_chars: int = 6000
    min_chunk_chars: int = 400
    overlap_lines: int = 8
    default_top_k: int = 8
    default_budget_tokens: int = 12000
    ignored_dirs: set[str] = field(default_factory=lambda: {
        ".git", ".xkit", "node_modules", "dist", "build", ".next", ".venv", "venv",
        "__pycache__", ".pytest_cache", ".mypy_cache", "target", "out", "coverage",
    })
    code_extensions: set[str] = field(default_factory=lambda: {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".swift", ".go",
        ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".scala",
        ".sql", ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".md", ".toml",
        ".xml", ".gradle", ".sh",
    })
    # Embedding / retrieval
    retriever: str = "tfidf"  # "tfidf" | "embeddings"
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model name
    embedding_device: str = "cpu"

    @classmethod
    def load(cls, project_root: Path) -> XkitConfig:
        """Load config from .xkit/config.toml if it exists, merging with defaults."""
        config_path = project_root / ".xkit" / "config.toml"
        if not config_path.exists():
            return cls()
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        return cls._merge(raw)

    @classmethod
    def _merge(cls, raw: dict) -> XkitConfig:
        """Merge a parsed TOML dict into a new XkitConfig, overriding defaults."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                val = raw[field_name]
                # Handle sets from TOML arrays
                if isinstance(val, list) and isinstance(getattr(cls(), field_name), set):
                    val = set(val)
                kwargs[field_name] = val
        return cls(**kwargs)

    def save_default(self, project_root: Path) -> Path:
        """Write a default config.toml to .xkit/ for user customization."""
        d = project_root / self.index_dir_name
        d.mkdir(exist_ok=True)
        path = d / "config.toml"
        if not path.exists():
            lines = [
                "# Xkit configuration",
                "# Uncomment and change values as needed.",
                "",
                "# Chunking",
                "# max_chunk_chars = 6000",
                "# min_chunk_chars = 400",
                "# overlap_lines = 8",
                "",
                "# Retrieval defaults",
                "# default_top_k = 8",
                "# default_budget_tokens = 12000",
                "",
                "# Retriever: 'tfidf' (no deps) or 'embeddings' (requires sentence-transformers)",
                '# retriever = "tfidf"',
                '# embedding_model = "all-MiniLM-L6-v2"',
                "",
                "# Ignored directories (list format)",
                '# ignored_dirs = [".git", "node_modules", "__pycache__"]',
                "",
                "# Code file extensions",
                '# code_extensions = [".py", ".js", ".ts", ".go", ".rs"]',
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def xkit_dir(project_root: Path, config: XkitConfig | None = None) -> Path:
    cfg = config or XkitConfig()
    return project_root / cfg.index_dir_name
