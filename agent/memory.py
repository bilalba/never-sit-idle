"""Three-tier memory system: system prompt, long-term memory, working memory."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agent import config
from agent.llm import count_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

class SystemPrompt:
    """Static system prompt. Loaded once, never mutated at runtime."""

    def __init__(self, text: str = ""):
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value

    @property
    def token_count(self) -> int:
        return count_tokens(self._text)

    def to_message(self) -> dict[str, str]:
        return {"role": "system", "content": self._text}


# ---------------------------------------------------------------------------
# Long-Term Memory  (persistent, ≤50k tokens, stored as MD + index)
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Persistent memory backed by markdown files in a directory.

    Structure:
        memory_dir/
            _index.json        ← metadata index
            <category>/
                <entry>.md     ← individual memory entries
    """

    def __init__(self, memory_dir: str | Path | None = None, max_tokens: int | None = None):
        self.memory_dir = Path(memory_dir or config.KNOWLEDGE_BASE_DIR) / ".memory"
        self.max_tokens = max_tokens or config.LONG_TERM_MEMORY_MAX_TOKENS
        self._index: dict[str, dict[str, Any]] = {}
        self._ensure_dir()
        self._load_index()

    # --- Directory / index ---

    def _ensure_dir(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self.memory_dir / "_index.json"

    def _load_index(self) -> None:
        ip = self._index_path()
        if ip.exists():
            try:
                self._index = json.loads(ip.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt memory index, rebuilding")
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _save_index(self) -> None:
        self._index_path().write_text(json.dumps(self._index, indent=2))

    def _rebuild_index(self) -> None:
        """Walk memory_dir and rebuild the index from file contents."""
        self._index = {}
        for md in self.memory_dir.rglob("*.md"):
            rel = md.relative_to(self.memory_dir)
            key = str(rel)
            content = md.read_text()
            self._index[key] = {
                "tokens": count_tokens(content),
                "updated": md.stat().st_mtime,
                "summary": content[:200],
            }
        self._save_index()

    # --- Token accounting ---

    @property
    def total_tokens(self) -> int:
        return sum(e["tokens"] for e in self._index.values())

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    # --- CRUD ---

    def store(self, category: str, name: str, content: str) -> dict[str, Any]:
        """Store a memory entry. Returns metadata dict.

        Raises ValueError if the entry would exceed the token budget.
        """
        tokens = count_tokens(content)
        key = f"{category}/{name}.md"

        # If updating, reclaim old tokens first
        old_tokens = self._index.get(key, {}).get("tokens", 0)
        net_new = tokens - old_tokens

        if net_new > self.remaining_tokens:
            raise ValueError(
                f"Memory budget exceeded: need {net_new} tokens, "
                f"only {self.remaining_tokens} remaining "
                f"(total {self.total_tokens}/{self.max_tokens})"
            )

        path = self.memory_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        meta = {
            "tokens": tokens,
            "updated": time.time(),
            "summary": content[:200],
        }
        self._index[key] = meta
        self._save_index()
        logger.info("Stored memory %s (%d tokens)", key, tokens)
        return meta

    def retrieve(self, category: str, name: str) -> str | None:
        """Read a memory entry. Returns None if not found."""
        key = f"{category}/{name}.md"
        path = self.memory_dir / key
        if path.exists():
            return path.read_text()
        return None

    def delete(self, category: str, name: str) -> bool:
        """Delete a memory entry. Returns True if it existed."""
        key = f"{category}/{name}.md"
        path = self.memory_dir / key
        if path.exists():
            path.unlink()
            self._index.pop(key, None)
            self._save_index()
            logger.info("Deleted memory %s", key)
            return True
        return False

    def list_entries(self, category: str | None = None) -> list[dict[str, Any]]:
        """List memory entries, optionally filtered by category."""
        results = []
        for key, meta in self._index.items():
            if category and not key.startswith(f"{category}/"):
                continue
            results.append({"key": key, **meta})
        return sorted(results, key=lambda x: x.get("updated", 0), reverse=True)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Simple keyword search across memory entries."""
        query_lower = query.lower()
        results = []
        for key in self._index:
            path = self.memory_dir / key
            if not path.exists():
                continue
            content = path.read_text()
            if query_lower in content.lower() or query_lower in key.lower():
                results.append({
                    "key": key,
                    "snippet": _extract_snippet(content, query_lower),
                    "tokens": self._index[key]["tokens"],
                })
        return results

    def get_context_block(self, max_tokens: int | None = None) -> str:
        """Build a single text block of all memories for injection into prompt.

        Respects max_tokens budget (defaults to self.max_tokens).
        """
        budget = max_tokens or self.max_tokens
        parts: list[str] = []
        used = 0
        for entry in self.list_entries():
            key = entry["key"]
            path = self.memory_dir / key
            if not path.exists():
                continue
            content = path.read_text()
            entry_tokens = count_tokens(content)
            if used + entry_tokens > budget:
                continue
            parts.append(f"## [{key}]\n{content}")
            used += entry_tokens
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Working Memory  (ephemeral, per-session)
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Ephemeral scratchpad for the current agent session.

    Entries are key-value pairs that live only in RAM.
    """

    def __init__(self, max_tokens: int | None = None):
        self.max_tokens = max_tokens or config.WORKING_MEMORY_MAX_TOKENS
        self._store: dict[str, str] = {}

    @property
    def total_tokens(self) -> int:
        return sum(count_tokens(v) for v in self._store.values())

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    def set(self, key: str, value: str) -> None:
        """Set a working-memory entry. Raises ValueError on budget overflow."""
        old_tokens = count_tokens(self._store[key]) if key in self._store else 0
        new_tokens = count_tokens(value)
        net = new_tokens - old_tokens
        if net > self.remaining_tokens:
            raise ValueError(
                f"Working memory budget exceeded: need {net}, "
                f"have {self.remaining_tokens}"
            )
        self._store[key] = value

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def list_keys(self) -> list[str]:
        return list(self._store.keys())

    def get_context_block(self) -> str:
        if not self._store:
            return ""
        parts = [f"### {k}\n{v}" for k, v in self._store.items()]
        return "\n\n".join(parts)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_snippet(content: str, query: str, window: int = 150) -> str:
    idx = content.lower().find(query)
    if idx == -1:
        return content[:window]
    start = max(0, idx - window // 2)
    end = min(len(content), idx + len(query) + window // 2)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet
