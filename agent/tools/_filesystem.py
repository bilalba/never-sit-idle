"""File-system tool implementations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _read_file(root: Path, path: str, max_lines: int) -> dict[str, Any]:
    target = (root / path).resolve()
    # Safety: don't escape codebase root
    if not str(target).startswith(str(root.resolve())):
        return {"error": "Path escapes codebase root"}
    if not target.exists():
        return {"error": f"File not found: {path}"}
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        lines = target.read_text(errors="replace").splitlines()
        truncated = len(lines) > max_lines
        content = "\n".join(lines[:max_lines])
        return {
            "path": path,
            "content": content,
            "lines": len(lines),
            "truncated": truncated,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _glob_files(root: Path, pattern: str) -> list[str]:
    matches = sorted(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
    return matches[:200]  # cap results


def _grep_files(root: Path, pattern: str, file_glob: str) -> list[dict[str, Any]]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return [{"error": f"Invalid regex: {exc}"}]

    results: list[dict[str, Any]] = []
    for fpath in root.rglob(file_glob):
        if not fpath.is_file():
            continue
        # Skip binary / large files
        if fpath.stat().st_size > 1_000_000:
            continue
        try:
            text = fpath.read_text(errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append({
                    "file": str(fpath.relative_to(root)),
                    "line": i,
                    "text": line.strip()[:300],
                })
                if len(results) >= 100:
                    return results
    return results
