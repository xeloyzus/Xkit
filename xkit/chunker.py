from __future__ import annotations

import re
from dataclasses import asdict, dataclass
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
    re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|open|override|async)\s+)*(?:func|fun|def|fn)\s+([A-Za-z_][\w]*)\s*[(<]"),
    re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|synchronized)\s+)+[\w<>\[\], ]+\s+([A-Za-z_][\w]*)\s*\("),
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

_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".scala": "scala", ".cs": "csharp",
}

_PARSER_CACHE: dict[str, object] = {}


def _get_parser(lang_name: str):
    """Load a tree-sitter parser, trying (in order):

    1. tree-sitter-language-pack (one dep, many languages)
    2. tree-sitter-languages (legacy bundled package)
    3. official per-language wheels (tree_sitter_python, tree_sitter_go, ...),
       which ship compiled grammars inside the wheel — no runtime downloads.

    Returns None when no source is available so callers fall back to regex.
    """
    if lang_name in _PARSER_CACHE:
        return _PARSER_CACHE[lang_name]
    parser = None
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang_name)
    except Exception:
        parser = None
    if parser is None:
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser(lang_name)
        except Exception:
            parser = None
    if parser is None:
        try:
            import importlib

            from tree_sitter import Language, Parser
            mod = importlib.import_module(f"tree_sitter_{lang_name}")
            parser = Parser(Language(mod.language()))
        except Exception:
            parser = None
    _PARSER_CACHE[lang_name] = parser
    return parser


def _get_tree_sitter_chunks(rel: str, text: str, max_chunk_chars: int, min_chunk_chars: int, overlap_lines: int) -> list[Chunk] | None:
    """AST-aware chunking: group top-level named nodes (functions, classes, ...)
    into chunks that respect max_chunk_chars. Returns None if tree-sitter is
    unavailable for this language so callers fall back to regex chunking."""
    lang_name = _LANG_MAP.get(Path(rel).suffix.lower())
    if lang_name is None:
        return None
    parser = _get_parser(lang_name)
    if parser is None:
        return None

    try:
        data = text.encode("utf-8")
        tree = parser.parse(data)
        top_nodes = [n for n in tree.root_node.children if n.is_named]
        if not top_nodes:
            return None

        lines = text.splitlines()
        chunks: list[Chunk] = []

        def node_symbol(node) -> str | None:
            for child in node.children:
                if child.type in ("identifier", "name", "type_identifier", "field_identifier"):
                    return data[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
            return None

        def emit_lines(symbol: str | None, start_line: int, end_line: int):
            """Emit one chunk, splitting by lines if it exceeds max_chunk_chars."""
            seg_start = start_line
            seg: list[str] = []
            seg_chars = 0
            for li in range(start_line, end_line):
                line = lines[li]
                if seg and seg_chars + len(line) + 1 > max_chunk_chars:
                    _push(symbol if seg_start == start_line else None, seg_start, li, seg)
                    keep = seg[-overlap_lines:] if overlap_lines > 0 else []
                    seg = list(keep)
                    seg_chars = sum(len(s) + 1 for s in seg)
                    seg_start = li - len(keep)
                seg.append(line)
                seg_chars += len(line) + 1
            _push(symbol if seg_start == start_line else None, seg_start, end_line, seg)

        def _push(symbol: str | None, start_line: int, end_line: int, seg: list[str]):
            chunk_text = "\n".join(seg).strip("\n")
            if not chunk_text:
                return
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

        def emit(group: list, end_line: int):
            if not group:
                return
            start_line = group[0].start_point[0]
            symbol = next((s for s in (node_symbol(n) for n in group) if s), None)
            # Prepend a few trailing lines of the previous chunk for continuity.
            ctx_start = start_line
            if overlap_lines > 0 and chunks:
                ctx_start = max(0, start_line - overlap_lines)
            emit_lines(symbol, ctx_start, end_line)

        # Group consecutive top-level nodes: flush at symbol boundaries once the
        # group reaches min_chunk_chars (mirrors the regex chunker's semantics),
        # or earlier if adding the node would exceed max_chunk_chars.
        group: list = []
        group_chars = 0
        for node in top_nodes:
            node_chars = node.end_byte - node.start_byte
            boundary = node_symbol(node) is not None and group_chars >= min_chunk_chars
            too_large = group_chars + node_chars > max_chunk_chars
            if group and (boundary or too_large):
                emit(group, node.start_point[0])
                group, group_chars = [], 0
            group.append(node)
            group_chars += node_chars
        emit(group, len(lines))

        return chunks or None
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
