from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from .files import sha256_text
from .token_estimator import estimate_tokens


@dataclass
class Chunk:
    chunk_id: str
    file: str
    kind: str
    symbol: str | None
    start_line: int
    end_line: int
    text: str
    hash: str
    token_estimate: int

    def to_dict(self):
        return asdict(self)


# Regex-based symbol detection (works for all languages without extra deps)
SYMBOL_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][\w]*)\s*=\s*(?:async\s*)?\(?"),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*(?:public|private|protected|static|final|open|override|async|func|def)\s+.*?\b([A-Za-z_][\w]*)\s*\("),
    re.compile(r"^\s*def\s+([A-Za-z_][\w]*)\s*\("),
    re.compile(r"^\s*struct\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_][\w]*)\s*="),
    re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*impl\s+([A-Za-z_][\w]*(?:\s*<[^>]*>)?)\s+(?:for|$)"),
    re.compile(r"^\s*trait\s+([A-Za-z_][\w]*)"),
    re.compile(r"^\s*module\s+([A-Za-z_][\w]*)"),
]


def _find_symbol(line: str) -> str | None:
    for pattern in SYMBOL_PATTERNS:
        m = pattern.search(line)
        if m:
            return m.group(1)
    return None


# --- AST-aware chunking via tree-sitter (optional) ---

try:
    from tree_sitter import Language, Parser

    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False


def _get_tree_sitter_chunks(rel: str, text: str, max_chunk_chars: int, min_chunk_chars: int, overlap_lines: int) -> list[Chunk] | None:
    """Try to use tree-sitter for AST-aware chunking. Returns None if unavailable."""
    if not _TS_AVAILABLE:
        return None

    try:
        # Map file extensions to tree-sitter language grammars
        ext = Path(rel).suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
        }
        lang_name = lang_map.get(ext)
        if lang_name is None:
            return None

        # Try to load the language grammar
        try:
            lang = Language(f"build/{lang_name}.so", lang_name)
        except Exception:
            # Grammar not compiled — fall back to regex
            return None

        parser = Parser()
        parser.set_language(lang)
        tree = parser.parse(text.encode("utf-8"))

        lines = text.splitlines()
        chunks: list[Chunk] = []
        cursor = tree.walk()

        # Collect top-level named nodes (functions, classes, etc.)
        top_nodes = []
        if cursor.goto_first_child():
            while True:
                if cursor.node.is_named:
                    top_nodes.append(cursor.node)
                if not cursor.goto_next_sibling():
                    break
            cursor.goto_parent()

        if not top_nodes:
            return None

        # Group nodes into chunks respecting max_chunk_chars
        current_group: list = []
        current_chars = 0

        def flush_group(end_line: int):
            nonlocal current_group, current_chars
            if not current_group:
                return
            start_line = current_group[0].start_point[0]  # 0-based
            chunk_text = "\n".join(lines[start_line:end_line]).strip("\n")
            if not chunk_text:
                current_group = []
                current_chars = 0
                return
            # Find the first symbol name
            symbol = None
            for node in current_group:
                for child in node.children:
                    if child.type in ("identifier", "name") and child.start_point[0] >= start_line:
                        symbol = text[child.start_byte:child.end_byte]
                        break
                if symbol:
                    break
            cid_base = f"{rel}:{symbol or 'block'}:{start_line + 1}-{end_line}"
            chunks.append(Chunk(
                chunk_id=f"{cid_base}:{sha256_text(chunk_text)[:8]}",
                file=rel,
                kind="symbol" if symbol else "block",
                symbol=symbol,
                start_line=start_line + 1,
                end_line=end_line,
                text=chunk_text,
                hash=sha256_text(chunk_text),
                token_estimate=estimate_tokens(chunk_text),
            ))
            current_group = []
            current_chars = 0

        for node in top_nodes:
            node_start = node.start_point[0]
            node_end = node.end_point[0]
            node_text = "\n".join(lines[node_start:node_end])
            node_chars = len(node_text)

            if current_group and current_chars + node_chars > max_chunk_chars and current_chars >= min_chunk_chars:
                flush_group(node_start)
                # Add overlap
                if overlap_lines > 0 and chunks:
                    prev_lines = chunks[-1].text.splitlines()
                    overlap = prev_lines[-overlap_lines:] if len(prev_lines) >= overlap_lines else prev_lines
                    for ol in overlap:
                        current_group.append(type("node", (), {"start_point": (node_start - len(overlap) + overlap.index(ol), 0)})())
                    current_chars = sum(len(l) + 1 for l in overlap)

            if not current_group:
                current_group = [node]
                current_chars = node_chars
            else:
                current_group.append(node)
                current_chars += node_chars

        if current_group:
            flush_group(len(lines))

        return chunks

    except Exception:
        return None


def chunk_file(project_root: Path, path: Path, text: str, max_chunk_chars: int = 6000, min_chunk_chars: int = 400, overlap_lines: int = 8) -> list[Chunk]:
    rel = str(path.relative_to(project_root))
    lines = text.splitlines()
    if not lines:
        return []

    # Try AST-aware chunking first
    ts_chunks = _get_tree_sitter_chunks(rel, text, max_chunk_chars, min_chunk_chars, overlap_lines)
    if ts_chunks is not None:
        return ts_chunks

    # Fall back to regex-based chunking
    chunks: list[Chunk] = []
    start = 0
    current: list[str] = []
    current_symbol: str | None = None

    def flush(end_idx: int):
        nonlocal current, start, current_symbol
        if not current:
            return
        chunk_text = "\n".join(current).strip("\n")
        if not chunk_text:
            current = []
            return
        symbol = current_symbol
        cid_base = f"{rel}:{symbol or 'chunk'}:{start + 1}-{end_idx}"
        chunks.append(Chunk(
            chunk_id=f"{cid_base}:{sha256_text(chunk_text)[:8]}",
            file=rel,
            kind="symbol" if symbol else "block",
            symbol=symbol,
            start_line=start + 1,
            end_line=end_idx,
            text=chunk_text,
            hash=sha256_text(chunk_text),
            token_estimate=estimate_tokens(chunk_text),
        ))
        current = []
        current_symbol = None
        start = end_idx

    for idx, line in enumerate(lines, start=1):
        symbol = _find_symbol(line)
        current_len = sum(len(x) + 1 for x in current)
        starts_new_symbol = symbol is not None and current_len >= min_chunk_chars
        too_large = current_len + len(line) + 1 > max_chunk_chars

        if current and (starts_new_symbol or too_large):
            flush(idx - 1)
            # Add overlap lines from the end of the previous chunk for context continuity
            if overlap_lines > 0 and chunks:
                prev_chunk_lines = chunks[-1].text.splitlines()
                overlap = prev_chunk_lines[-overlap_lines:] if len(prev_chunk_lines) >= overlap_lines else prev_chunk_lines
                current = list(overlap)
                current_len = sum(len(x) + 1 for x in current)

        if not current:
            start = idx - 1
            current_symbol = symbol
        elif symbol and not current_symbol:
            current_symbol = symbol

        current.append(line)

    flush(len(lines))
    return chunks
