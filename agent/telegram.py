"""Telegram bot for agent notifications and conversational interaction.

Uses the Bot API directly via requests — no extra dependencies.
The bot can:
  - Send notifications (job done, queue empty, shutting down)
  - Long-poll for user messages, batch-draining all pending
  - Handle commands: /jobs, /status, /help, /queue, /cancel
  - Route plain-text messages through the agent for conversational replies
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from agent import config

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
TELEGRAM_MAX_LENGTH = 4096


def _esc(text: str) -> str:
    """Escape text for safe inclusion in Telegram HTML messages."""
    return _html.escape(text, quote=False)


def _md_to_html(text: str) -> str:
    """Convert standard Markdown to Telegram-compatible HTML.

    Handles: **bold**, *italic*, `code`, ```code blocks```,
    [text](url), ~~strikethrough~~.
    """
    placeholders: list[str] = []

    def _save(match: re.Match) -> str:
        idx = len(placeholders)
        placeholders.append(match.group(0))
        return f"\x00{idx}\x00"

    # Protect code blocks (```lang\n...\n```) then inline code (`...`)
    text = re.sub(r"```(?:\w*\n)?(.*?)```", _save, text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", _save, text)

    # Escape HTML in remaining (non-code) text
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic: *text* or _text_ (not mid-word underscores like snake_case)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # Restore code placeholders with HTML escaping inside
    def _restore(match: re.Match) -> str:
        idx = int(match.group(1))
        original = placeholders[idx]
        if original.startswith("```"):
            inner = re.sub(r"^```\w*\n?", "", original)
            inner = inner.removesuffix("```")
            inner = _html.escape(inner, quote=False)
            return f"<pre>{inner}</pre>"
        else:
            inner = original[1:-1]
            inner = _html.escape(inner, quote=False)
            return f"<code>{inner}</code>"

    text = re.sub(r"\x00(\d+)\x00", _restore, text)
    return text


def _split_message(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    """Split text into chunks that fit within Telegram's message limit.

    Tries to break at newlines, falls back to hard split.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to break at last newline within limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            # No good newline — hard split
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


@dataclass
class IncomingMessage:
    """A parsed message from Telegram."""
    text: str
    message_id: int
    is_command: bool
    command: str  # e.g. "jobs", "status" — empty if not a command
    args: str     # everything after the command


class TelegramBot:
    """Thin wrapper around the Telegram Bot API."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ):
        self.token = token if token is not None else config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id if chat_id is not None else config.TELEGRAM_CHAT_ID
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

    def send(
        self,
        text: str,
        parse_mode: str = "HTML",
        reply_to: int | None = None,
    ) -> dict[str, Any] | None:
        """Send a message to the configured chat.

        If HTML parsing fails, retries as plain text.
        """
        params: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
        }
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_to:
            params["reply_to_message_id"] = reply_to
        result = self._call("sendMessage", **params)
        if result is None and parse_mode:
            # Markdown parse error — retry without formatting
            logger.debug("Retrying send without parse_mode")
            params.pop("parse_mode", None)
            result = self._call("sendMessage", **params)
        return result

    def reply(self, msg: IncomingMessage, text: str) -> dict[str, Any] | None:
        """Reply to a specific incoming message."""
        return self.send(text, reply_to=msg.message_id)

    def send_long(
        self,
        text: str,
        reply_to: int | None = None,
        parse_mode: str = "HTML",
        limit: int = 4096,
    ) -> None:
        """Send a message, splitting into chunks if it exceeds Telegram's limit."""
        for chunk in _split_message(text, limit):
            self.send(chunk, parse_mode=parse_mode, reply_to=reply_to)
            reply_to = None  # only first chunk is a reply

    def notify_idle(self, completed_count: int = 0) -> None:
        """Tell the user the queue is empty and ask for work."""
        msg = "\U0001f4ed <b>Queue empty.</b>"
        if completed_count:
            msg += f" Completed {completed_count} job{'s' if completed_count != 1 else ''}."
        msg += (
            "\n\nSend me topics to research, or use commands:"
            "\n/help — show all commands"
        )
        self.send(msg)

    def notify_job_started(self, job: dict[str, Any]) -> None:
        subject = _esc(job.get("subject", "unknown"))
        job_type = _esc(job.get("type", "job"))
        self.send(f"\u2699\ufe0f Starting {job_type}: <b>{subject}</b>")

    def notify_job_done(self, job: dict[str, Any]) -> None:
        subject = _esc(job.get("subject", "unknown"))
        duration = ""
        if job.get("started_at") and job.get("completed_at"):
            secs = int(job["completed_at"] - job["started_at"])
            mins, secs = divmod(secs, 60)
            duration = f" ({mins}m {secs}s)" if mins else f" ({secs}s)"
        self.send(f"\u2705 Done: <b>{subject}</b>{duration}")

    def notify_job_failed(self, job: dict[str, Any]) -> None:
        subject = _esc(job.get("subject", "unknown"))
        error = _esc(job.get("error", "unknown error"))
        if len(error) > 200:
            error = error[:200] + "..."
        self.send(f"\u274c Failed: <b>{subject}</b>\n<code>{error}</code>")

    def notify_shutdown(self, stats: dict[str, Any] | None = None) -> None:
        msg = "\U0001f6d1 <b>Agent shutting down.</b>"
        if stats:
            msg += f"\nJobs completed: {stats.get('jobs_completed', 0)}"
            msg += f"\nJobs in queue: {stats.get('jobs_queued', 0)}"
        self.send(msg)

    def send_typing(self) -> None:
        """Send a 'typing' chat action (lasts ~5 seconds)."""
        self._call("sendChatAction", chat_id=self.chat_id, action="typing")

    def edit_message(
        self,
        message_id: int,
        text: str,
        parse_mode: str = "",
    ) -> dict[str, Any] | None:
        """Edit an existing message by ID."""
        params: dict[str, Any] = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode:
            params["parse_mode"] = parse_mode
        return self._call("editMessageText", **params)

    # ── Receiving ─────────────────────────────────────────────────────

    def _parse_message(self, text: str, message_id: int) -> IncomingMessage:
        """Parse raw text into an IncomingMessage."""
        text = text.strip()
        if text.startswith("/"):
            parts = text.split(None, 1)
            # Strip @botname from command (e.g. /jobs@mybot -> jobs)
            cmd = parts[0][1:].split("@")[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return IncomingMessage(
                text=text, message_id=message_id,
                is_command=True, command=cmd, args=args,
            )
        return IncomingMessage(
            text=text, message_id=message_id,
            is_command=False, command="", args="",
        )

    def poll_messages(self, timeout: int = 30) -> list[IncomingMessage]:
        """Long-poll and return ALL pending messages from our chat.

        Returns a list of IncomingMessage (may be empty).
        Advances the offset past all returned updates.
        """
        result = self._call(
            "getUpdates",
            offset=self._update_offset,
            timeout=timeout,
            allowed_updates=["message"],
        )
        if not result:
            return []

        messages: list[IncomingMessage] = []
        for update in result:
            update_id = update.get("update_id", 0)
            self._update_offset = max(self._update_offset, update_id + 1)

            msg = update.get("message", {})
            chat = msg.get("chat", {})
            text = msg.get("text", "")
            message_id = msg.get("message_id", 0)

            # Only accept messages from our chat
            if str(chat.get("id")) == str(self.chat_id) and text.strip():
                messages.append(self._parse_message(text, message_id))

        return messages

    # Keep poll_reply for backwards compat / simple cases
    def poll_reply(self, timeout: int = 30) -> str | None:
        """Long-poll for a single text reply. Returns text or None."""
        msgs = self.poll_messages(timeout=timeout)
        for m in msgs:
            if not m.is_command:
                return m.text
        return None

    # ── Command handlers ──────────────────────────────────────────────

    def handle_commands(
        self,
        messages: list[IncomingMessage],
        queue_module: Any,
        agent_stats: dict[str, Any] | None = None,
    ) -> list[IncomingMessage]:
        """Process command messages, reply inline, return non-command messages.

        Takes the full batch from poll_messages(). Commands are handled
        and replied to immediately. Plain-text messages (topics to queue)
        are returned for the caller to process.
        """
        topics: list[IncomingMessage] = []

        for msg in messages:
            if not msg.is_command:
                topics.append(msg)
                continue

            handler = {
                "help": self._cmd_help,
                "start": self._cmd_help,
                "jobs": lambda m: self._cmd_jobs(m, queue_module),
                "status": lambda m: self._cmd_status(m, queue_module, agent_stats),
                "queue": lambda m: self._cmd_queue(m, queue_module),
                "cancel": lambda m: self._cmd_cancel(m, queue_module),
                "clear": lambda m: self._cmd_clear(m, queue_module),
            }.get(msg.command)

            if handler:
                handler(msg)
            else:
                self.reply(msg, f"Unknown command: /{msg.command}\nSend /help for available commands.")

        return topics

    def _cmd_help(self, msg: IncomingMessage) -> None:
        self.reply(msg, (
            "<b>Commands:</b>\n"
            "/jobs — list recent jobs\n"
            "/status — daemon &amp; KB status\n"
            "/queue <code>topic</code> — queue a research job\n"
            "/cancel <code>job-id</code> — cancel a queued job\n"
            "/clear — remove done/failed jobs\n"
            "/help — this message\n"
            "\nOr just send me a message — I'll check the KB, "
            "answer questions, or queue research as needed."
        ))

    def _cmd_jobs(self, msg: IncomingMessage, Q: Any) -> None:
        jobs = Q.list_jobs()
        if not jobs:
            self.reply(msg, "No jobs.")
            return

        lines = []
        status_emoji = {
            "queued": "\U0001f552",   # clock
            "running": "\u2699\ufe0f", # gear
            "done": "\u2705",          # check
            "failed": "\u274c",        # cross
        }
        for j in jobs[:15]:  # cap at 15 to avoid message length limits
            emoji = status_emoji.get(j["status"], "\u2753")
            subject = _esc(j["subject"])
            if len(subject) > 45:
                subject = subject[:42] + "..."
            lines.append(f"{emoji} <code>{j['status']:7s}</code> {subject}")

        total = len(jobs)
        text = "\n".join(lines)
        if total > 15:
            text += f"\n\n<i>...and {total - 15} more</i>"
        self.reply(msg, text)

    def _cmd_status(
        self, msg: IncomingMessage, Q: Any,
        agent_stats: dict[str, Any] | None,
    ) -> None:
        queued = Q.queue_size()
        total = len(Q.list_jobs())
        lines = [
            f"<b>Queue:</b> {queued} pending, {total} total",
        ]
        if agent_stats:
            lines.append(f"<b>Turns:</b> {agent_stats.get('total_turns', 0)}")
            lines.append(f"<b>Tool calls:</b> {agent_stats.get('total_tool_calls', 0)}")
            kb = agent_stats.get("kb_stats", {})
            if kb:
                lines.append(f"<b>KB:</b> {kb.get('entry_count', 0)} entries, {kb.get('total_tokens', 0)} tokens")
        self.reply(msg, "\n".join(lines))

    def _cmd_queue(self, msg: IncomingMessage, Q: Any) -> None:
        topic = msg.args.strip()
        if not topic:
            self.reply(msg, "Usage: /queue <code>topic to research</code>")
            return
        job = Q.add("research", topic)
        self.reply(msg, f"\U0001f4cb Queued: <b>{_esc(topic)}</b>\n<code>{_esc(job['id'])}</code>")

    def _cmd_cancel(self, msg: IncomingMessage, Q: Any) -> None:
        job_id = msg.args.strip()
        if not job_id:
            self.reply(msg, "Usage: /cancel <code>job-id</code>")
            return
        # Find matching job(s) — allow prefix match
        jobs = Q.list_jobs(status="queued")
        match = [j for j in jobs if j["id"] == job_id or j["id"].startswith(job_id)]
        if not match:
            self.reply(msg, f"No queued job matching <code>{_esc(job_id)}</code>")
            return
        for j in match:
            Q.mark_failed(j["id"], error="Cancelled via Telegram")
        names = ", ".join(_esc(j["subject"]) for j in match)
        self.reply(msg, f"\U0001f6ab Cancelled: {names}")

    def _cmd_clear(self, msg: IncomingMessage, Q: Any) -> None:
        removed = Q.clear_done()
        self.reply(msg, f"Removed {removed} completed/failed job{'s' if removed != 1 else ''}.")
