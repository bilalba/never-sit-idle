"""Knowledge base manager — organizes codebase knowledge in nested MD files."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agent import config
from agent.llm import count_tokens

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Manages a nested markdown knowledge base on disk.

    Structure example:
        knowledge_base/
            _index.json
            overview.md
            architecture/
                overview.md
                data_flow.md
            modules/
                auth/
                    overview.md
                    endpoints.md
                database/
                    schema.md
            patterns/
                error_handling.md
    """

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or config.KNOWLEDGE_BASE_DIR)
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self.base_dir / "_index.json"

    def _load_index(self) -> dict[str, Any]:
        ip = self._index_path()
        if ip.exists():
            try:
                return json.loads(ip.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self, index: dict[str, Any]) -> None:
        self._index_path().write_text(json.dumps(index, indent=2))

    # --- Write ---

    def write_entry(self, path: str, content: str, *, tags: list[str] | None = None) -> dict[str, Any]:
        """Write or overwrite a knowledge base entry.

        Args:
            path: Slash-separated path like "architecture/data_flow" (no .md suffix needed).
            content: Markdown content.
            tags: Optional tags for searchability.

        Returns metadata dict.
        """
        clean_path = path.strip("/")
        if not clean_path.endswith(".md"):
            clean_path += ".md"

        full_path = self.base_dir / clean_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        tokens = count_tokens(content)
        meta = {
            "tokens": tokens,
            "updated": time.time(),
            "tags": tags or [],
            "summary": content[:200].replace("\n", " "),
        }

        index = self._load_index()
        index[clean_path] = meta
        self._save_index(index)

        logger.info("Wrote KB entry %s (%d tokens)", clean_path, tokens)
        return meta

    # --- Read ---

    def read_entry(self, path: str) -> str | None:
        clean_path = path.strip("/")
        if not clean_path.endswith(".md"):
            clean_path += ".md"
        full_path = self.base_dir / clean_path
        if full_path.exists():
            return full_path.read_text()
        return None

    # --- Delete ---

    def delete_entry(self, path: str) -> bool:
        clean_path = path.strip("/")
        if not clean_path.endswith(".md"):
            clean_path += ".md"
        full_path = self.base_dir / clean_path
        if full_path.exists():
            full_path.unlink()
            index = self._load_index()
            index.pop(clean_path, None)
            self._save_index(index)
            # Clean up empty parent dirs
            parent = full_path.parent
            while parent != self.base_dir:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
            logger.info("Deleted KB entry %s", clean_path)
            return True
        return False

    # --- List / tree ---

    def list_entries(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """List all entries, optionally filtered by path prefix."""
        index = self._load_index()
        results = []
        for key, meta in index.items():
            if prefix and not key.startswith(prefix.strip("/")):
                continue
            results.append({"path": key, **meta})
        return sorted(results, key=lambda x: x["path"])

    def tree(self) -> str:
        """Return a tree-style string representation of the knowledge base."""
        index = self._load_index()
        if not index:
            return "(empty knowledge base)"

        paths = sorted(index.keys())
        lines: list[str] = ["knowledge_base/"]
        for p in paths:
            parts = p.split("/")
            indent = "  " * len(parts)
            lines.append(f"{indent}{parts[-1]}")
        return "\n".join(lines)

    # --- Search ---

    def search(self, query: str, *, tags: list[str] | None = None) -> list[dict[str, Any]]:
        """Search knowledge base entries by keyword and/or tags."""
        query_lower = query.lower() if query else ""
        index = self._load_index()
        results = []

        for key, meta in index.items():
            # Tag filter
            if tags:
                entry_tags = set(meta.get("tags", []))
                if not entry_tags.intersection(tags):
                    continue

            # Keyword filter
            if query_lower:
                full_path = self.base_dir / key
                if not full_path.exists():
                    continue
                content = full_path.read_text()
                if query_lower not in content.lower() and query_lower not in key.lower():
                    continue
                snippet = _snippet(content, query_lower)
            else:
                snippet = meta.get("summary", "")

            results.append({
                "path": key,
                "snippet": snippet,
                "tags": meta.get("tags", []),
                "tokens": meta.get("tokens", 0),
            })

        return results

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        index = self._load_index()
        total_tokens = sum(m.get("tokens", 0) for m in index.values())
        return {
            "entry_count": len(index),
            "total_tokens": total_tokens,
            "categories": list({k.split("/")[0] for k in index if "/" in k}),
        }


def _snippet(content: str, query: str, window: int = 150) -> str:
    idx = content.lower().find(query)
    if idx == -1:
        return content[:window]
    start = max(0, idx - window // 2)
    end = min(len(content), idx + len(query) + window // 2)
    s = content[start:end]
    if start > 0:
        s = "..." + s
    if end < len(content):
        s = s + "..."
    return s
