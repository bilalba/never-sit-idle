"""Telegram conversation memory with automatic compaction.

Keeps recent exchanges verbatim and periodically summarizes older ones
into a rolling summary using the LLM.  Persists to disk so context
survives daemon restarts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agent.llm import chat_completion, extract_assistant_message, count_tokens

logger = logging.getLogger(__name__)

COMPACT_SYSTEM = (
    "You are a conversation summarizer. Output ONLY the summary, "
    "no preamble or explanation."
)

COMPACT_PROMPT = (
    "Summarize this conversation concisely. Preserve:\n"
    "- Key facts and information exchanged\n"
    "- User preferences and requests\n"
    "- Decisions made and topics discussed\n"
    "- Any pending questions or ongoing threads\n\n"
)


class TelegramMemory:
    """Rolling conversation memory for Telegram chat.

    Stores full user/assistant exchanges.  When the exchange count
    exceeds ``RECENT_LIMIT``, older exchanges are summarized into a
    compact text block via the LLM and the originals are discarded.

    Attributes:
        summary:   Compacted text of older conversation.
        exchanges: Recent verbatim exchanges ``[{"user": …, "assistant": …}]``.
    """

    RECENT_LIMIT = 4           # keep last N exchanges verbatim
    FORCE_COMPACT_LIMIT = 8    # force-compact before responding if over this
    IDLE_COMPACT_SECONDS = 120 # compact after this many seconds idle

    def __init__(self, persist_path: str | Path | None = None):
        self.summary: str = ""
        self.exchanges: list[dict[str, str]] = []
        self.last_activity: float = time.time()
        self.persist_path = Path(persist_path) if persist_path else None
        self._load()

    # ── Public API ────────────────────────────────────────────────────

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        """Record a completed user/assistant exchange."""
        self.exchanges.append({"user": user_text, "assistant": assistant_text})
        self.last_activity = time.time()
        self._save()

    def get_context_messages(self) -> list[dict[str, str]]:
        """Build message list to inject into the agent before a new run."""
        msgs: list[dict[str, str]] = []
        if self.summary:
            msgs.append({
                "role": "system",
                "content": (
                    "# Conversation History (summary of earlier messages)\n\n"
                    + self.summary
                ),
            })
        for ex in self.exchanges:
            msgs.append({"role": "user", "content": ex["user"]})
            msgs.append({"role": "assistant", "content": ex["assistant"]})
        return msgs

    def needs_compaction(self) -> bool:
        return len(self.exchanges) > self.RECENT_LIMIT

    def needs_force_compaction(self) -> bool:
        return len(self.exchanges) > self.FORCE_COMPACT_LIMIT

    def idle_long_enough(self) -> bool:
        return time.time() - self.last_activity > self.IDLE_COMPACT_SECONDS

    def compact(self) -> None:
        """Summarize older exchanges into ``self.summary``, keep recent ones."""
        if len(self.exchanges) <= self.RECENT_LIMIT:
            return

        to_summarize = self.exchanges[:-self.RECENT_LIMIT]
        keep = self.exchanges[-self.RECENT_LIMIT:]

        # Build conversation text
        parts: list[str] = []
        for ex in to_summarize:
            parts.append(f"User: {ex['user']}")
            parts.append(f"Assistant: {ex['assistant']}")
        new_text = "\n".join(parts)

        prompt = COMPACT_PROMPT
        if self.summary:
            prompt += f"Existing summary:\n{self.summary}\n\n"
        prompt += f"New conversation to incorporate:\n{new_text}"

        try:
            response = chat_completion(
                [
                    {"role": "system", "content": COMPACT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            msg = extract_assistant_message(response)
            self.summary = msg.get("content", "").strip()
            self.exchanges = keep
            logger.info(
                "Compacted %d exchanges into summary (%d tokens), %d recent kept",
                len(to_summarize),
                count_tokens(self.summary),
                len(self.exchanges),
            )
        except Exception as exc:
            logger.warning("Compaction failed, keeping all messages: %s", exc)
            return

        self._save()

    def clear(self) -> None:
        """Wipe all conversation context."""
        self.summary = ""
        self.exchanges.clear()
        self._save()

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary,
            "exchanges": self.exchanges,
            "last_activity": self.last_activity,
        }
        self.persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            self.summary = data.get("summary", "")
            self.exchanges = data.get("exchanges", [])
            self.last_activity = data.get("last_activity", time.time())
            logger.info(
                "Loaded telegram memory: %d exchanges, summary=%d chars",
                len(self.exchanges), len(self.summary),
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load telegram memory, starting fresh")
