#!/usr/bin/env python3
"""Helper to build tree-sitter grammars for common languages.

This script expects `tree-sitter` CLI and language repos available.
It's a convenience helper; building grammars is environment-specific.
"""
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

LANGS = {
    "python": "https://github.com/tree-sitter/tree-sitter-python",
    "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
    "typescript": "https://github.com/tree-sitter/tree-sitter-typescript",
    "rust": "https://github.com/tree-sitter/tree-sitter-rust",
}

OUT = Path("build")
OUT.mkdir(exist_ok=True)

def build(lang_name, repo):
    target = OUT / f"{lang_name}.so"
    print(f"Building {lang_name} -> {target}")
    try:
        # Prefer using the Python tree_sitter helper if available
        tmp = tempfile.mkdtemp(prefix="tree-sitter-")
        try:
            from tree_sitter import Language
            # ensure tmp is clean
            shutil.rmtree(tmp, ignore_errors=True)
            subprocess.check_call(["git", "clone", "--depth", "1", repo, tmp], stdout=subprocess.DEVNULL)
            src = Path(tmp)
            # Build library containing this single language
            Language.build_library(str(target), [str(src)])
        except Exception:
            # Fallback: try a basic gcc compile of parser sources (best-effort)
            # Ensure repo is cloned into tmp
            shutil.rmtree(tmp, ignore_errors=True)
            subprocess.check_call(["git", "clone", "--depth", "1", repo, tmp], stdout=subprocess.DEVNULL)
            # Attempt to find parser.c and scanner.c in common locations
            parser_c = Path(tmp) / "src" / "parser.c"
            scanner_c = Path(tmp) / "src" / "scanner.c"
            cmd = ["gcc", "-shared", "-fPIC", "-o", str(target)]
            if parser_c.exists():
                cmd.append(str(parser_c))
            if scanner_c.exists():
                cmd.append(str(scanner_c))
            if len(cmd) > 4:
                subprocess.check_call(cmd)
            else:
                raise RuntimeError("No parser sources found for fallback build")
    except Exception as e:
        print("Build failed:", e)
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

if __name__ == '__main__':
    for k, v in LANGS.items():
        build(k, v)
