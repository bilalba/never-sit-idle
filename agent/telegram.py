"""Telegram bot for agent notifications and job intake.

Uses the Bot API directly via requests — no extra dependencies.
The bot can:
  - Send notifications (job done, queue empty, shutting down)
  - Long-poll for user replies to queue new jobs
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from agent import config

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    """Thin wrapper around the Telegram Bot API."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ):
        self.token = token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self._update_offset: int = 0  # tracks last seen update

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _url(self, method: str) -> str:
        return TELEGRAM_API.format(token=self.token, method=method)

    def _call(self, method: str, **params: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            resp = requests.post(self._url(method), json=params, timeout=60)
            data = resp.json()
            if not data.get("ok"):
                logger.warning("Telegram API error: %s", data.get("description", data))
                return None
            return data.get("result")
        except requests.RequestException as exc:
            logger.warning("Telegram request failed: %s", exc)
            return None

    # ── Sending ───────────────────────────────────────────────────────

    def send(self, text: str, parse_mode: str = "Markdown") -> dict[str, Any] | None:
        """Send a message to the configured chat."""
        return self._call(
            "sendMessage",
            chat_id=self.chat_id,
            text=text,
            parse_mode=parse_mode,
        )

    def notify_idle(self, completed_count: int = 0) -> None:
        """Tell the user the queue is empty and ask for work."""
        msg = "\U0001f4ed *Queue empty.*"
        if completed_count:
            msg += f" Completed {completed_count} job{'s' if completed_count != 1 else ''}."
        msg += (
            "\n\nWhat should I research next? "
            "Reply with a topic and I'll queue it up."
        )
        self.send(msg)

    def notify_job_started(self, job: dict[str, Any]) -> None:
        subject = job.get("subject", "unknown")
        job_type = job.get("type", "job")
        self.send(f"\u2699\ufe0f Starting {job_type}: *{subject}*")

    def notify_job_done(self, job: dict[str, Any]) -> None:
        subject = job.get("subject", "unknown")
        duration = ""
        if job.get("started_at") and job.get("completed_at"):
            secs = int(job["completed_at"] - job["started_at"])
            mins, secs = divmod(secs, 60)
            duration = f" ({mins}m {secs}s)" if mins else f" ({secs}s)"
        self.send(f"\u2705 Done: *{subject}*{duration}")

    def notify_job_failed(self, job: dict[str, Any]) -> None:
        subject = job.get("subject", "unknown")
        error = job.get("error", "unknown error")
        if len(error) > 200:
            error = error[:200] + "..."
        self.send(f"\u274c Failed: *{subject}*\n`{error}`")

    def notify_shutdown(self, stats: dict[str, Any] | None = None) -> None:
        msg = "\U0001f6d1 *Agent shutting down.*"
        if stats:
            msg += f"\nJobs completed: {stats.get('jobs_completed', 0)}"
            msg += f"\nJobs in queue: {stats.get('jobs_queued', 0)}"
        self.send(msg)

    # ── Receiving ─────────────────────────────────────────────────────

    def poll_reply(self, timeout: int = 30) -> str | None:
        """Long-poll for a single text reply from the user.

        Returns the message text, or None if no message arrives
        within the timeout window. Only returns messages from the
        configured chat_id.
        """
        result = self._call(
            "getUpdates",
            offset=self._update_offset,
            timeout=timeout,
            allowed_updates=["message"],
        )
        if not result:
            return None

        for update in result:
            update_id = update.get("update_id", 0)
            self._update_offset = max(self._update_offset, update_id + 1)

            msg = update.get("message", {})
            chat = msg.get("chat", {})
            text = msg.get("text", "")

            # Only accept messages from our chat
            if str(chat.get("id")) == str(self.chat_id) and text.strip():
                return text.strip()

        return None

    def wait_for_reply(self, timeout: int = 30, max_wait: int = 0) -> str | None:
        """Poll for a reply, optionally up to max_wait seconds total.

        If max_wait is 0, polls once (with long-poll timeout).
        If max_wait > 0, keeps polling until a reply or max_wait elapsed.
        Returns the reply text or None.
        """
        if max_wait <= 0:
            return self.poll_reply(timeout=timeout)

        deadline = time.time() + max_wait
        while time.time() < deadline:
            remaining = int(deadline - time.time())
            poll_timeout = min(timeout, max(1, remaining))
            reply = self.poll_reply(timeout=poll_timeout)
            if reply:
                return reply

        return None
