"""Tests for the Telegram bot integration."""

import json
import pytest
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace

from agent.telegram import TelegramBot, IncomingMessage


def _ok_response(result=None):
    resp = MagicMock()
    resp.json.return_value = {"ok": True, "result": result or {}}
    return resp


def _error_response(desc="Bad Request"):
    resp = MagicMock()
    resp.json.return_value = {"ok": False, "description": desc}
    return resp


def _make_update(update_id, chat_id, text, message_id=1):
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "chat": {"id": chat_id},
            "text": text,
        },
    }


class TestEnabled:
    def test_enabled_when_both_set(self):
        bot = TelegramBot(token="tok", chat_id="123")
        assert bot.enabled is True

    def test_disabled_when_no_token(self):
        bot = TelegramBot(token="", chat_id="123")
        assert bot.enabled is False

    def test_disabled_when_no_chat_id(self):
        bot = TelegramBot(token="tok", chat_id="")
        assert bot.enabled is False


class TestSend:
    @patch("agent.telegram.requests.post")
    def test_sends_message(self, mock_post):
        mock_post.return_value = _ok_response({"message_id": 1})
        bot = TelegramBot(token="tok", chat_id="123")
        result = bot.send("hello")
        assert result == {"message_id": 1}

        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs["json"]
        assert body["chat_id"] == "123"
        assert body["text"] == "hello"

    @patch("agent.telegram.requests.post")
    def test_send_with_reply_to(self, mock_post):
        mock_post.return_value = _ok_response({"message_id": 2})
        bot = TelegramBot(token="tok", chat_id="123")
        bot.send("reply", reply_to=42)

        body = mock_post.call_args.kwargs["json"]
        assert body["reply_to_message_id"] == 42

    @patch("agent.telegram.requests.post")
    def test_reply_to_message(self, mock_post):
        mock_post.return_value = _ok_response({"message_id": 3})
        bot = TelegramBot(token="tok", chat_id="123")
        msg = IncomingMessage(text="hello", message_id=55, is_command=False, command="", args="")
        bot.reply(msg, "got it")

        body = mock_post.call_args.kwargs["json"]
        assert body["reply_to_message_id"] == 55
        assert body["text"] == "got it"

    @patch("agent.telegram.requests.post")
    def test_returns_none_on_api_error(self, mock_post):
        mock_post.return_value = _error_response("Unauthorized")
        bot = TelegramBot(token="bad", chat_id="123")
        assert bot.send("hello") is None

    def test_returns_none_when_disabled(self):
        bot = TelegramBot(token="", chat_id="")
        assert bot.send("hello") is None

    @patch("agent.telegram.requests.post")
    def test_returns_none_on_network_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.ConnectionError("offline")
        bot = TelegramBot(token="tok", chat_id="123")
        assert bot.send("test") is None


class TestNotifications:
    @patch("agent.telegram.requests.post")
    def test_notify_idle(self, mock_post):
        mock_post.return_value = _ok_response()
        bot = TelegramBot(token="tok", chat_id="123")
        bot.notify_idle(completed_count=3)

        body = mock_post.call_args.kwargs["json"]
        assert "Queue empty" in body["text"]
        assert "3 jobs" in body["text"]
        assert "/help" in body["text"]

    @patch("agent.telegram.requests.post")
    def test_notify_job_done(self, mock_post):
        mock_post.return_value = _ok_response()
        bot = TelegramBot(token="tok", chat_id="123")
        bot.notify_job_done({
            "subject": "React hooks",
            "started_at": 1000,
            "completed_at": 1095,
        })
        body = mock_post.call_args.kwargs["json"]
        assert "React hooks" in body["text"]
        assert "1m 35s" in body["text"]

    @patch("agent.telegram.requests.post")
    def test_notify_shutdown(self, mock_post):
        mock_post.return_value = _ok_response()
        bot = TelegramBot(token="tok", chat_id="123")
        bot.notify_shutdown({"jobs_completed": 5, "jobs_queued": 2})
        body = mock_post.call_args.kwargs["json"]
        assert "shutting down" in body["text"].lower()
        assert "5" in body["text"]


class TestParseMessage:
    def test_plain_text(self):
        bot = TelegramBot(token="tok", chat_id="123")
        msg = bot._parse_message("WebSocket performance", 10)
        assert not msg.is_command
        assert msg.text == "WebSocket performance"
        assert msg.command == ""
        assert msg.message_id == 10

    def test_command_no_args(self):
        bot = TelegramBot(token="tok", chat_id="123")
        msg = bot._parse_message("/jobs", 11)
        assert msg.is_command
        assert msg.command == "jobs"
        assert msg.args == ""

    def test_command_with_args(self):
        bot = TelegramBot(token="tok", chat_id="123")
        msg = bot._parse_message("/queue React hooks best practices", 12)
        assert msg.is_command
        assert msg.command == "queue"
        assert msg.args == "React hooks best practices"

    def test_command_strips_botname(self):
        bot = TelegramBot(token="tok", chat_id="123")
        msg = bot._parse_message("/jobs@my_cool_bot", 13)
        assert msg.is_command
        assert msg.command == "jobs"


class TestPollMessages:
    @patch("agent.telegram.requests.post")
    def test_returns_all_messages(self, mock_post):
        mock_post.return_value = _ok_response([
            _make_update(100, 123, "topic one", message_id=1),
            _make_update(101, 123, "topic two", message_id=2),
            _make_update(102, 123, "/jobs", message_id=3),
        ])
        bot = TelegramBot(token="tok", chat_id="123")
        msgs = bot.poll_messages(timeout=1)

        assert len(msgs) == 3
        assert msgs[0].text == "topic one"
        assert msgs[1].text == "topic two"
        assert msgs[2].is_command
        assert msgs[2].command == "jobs"
        assert bot._update_offset == 103

    @patch("agent.telegram.requests.post")
    def test_ignores_other_chats(self, mock_post):
        mock_post.return_value = _ok_response([
            _make_update(100, 999, "spam"),
            _make_update(101, 123, "real message", message_id=5),
        ])
        bot = TelegramBot(token="tok", chat_id="123")
        msgs = bot.poll_messages(timeout=1)

        assert len(msgs) == 1
        assert msgs[0].text == "real message"
        assert bot._update_offset == 102  # still advanced past both

    @patch("agent.telegram.requests.post")
    def test_returns_empty_on_timeout(self, mock_post):
        mock_post.return_value = _ok_response([])
        bot = TelegramBot(token="tok", chat_id="123")
        assert bot.poll_messages(timeout=1) == []

    @patch("agent.telegram.requests.post")
    def test_returns_empty_when_disabled(self, mock_post):
        bot = TelegramBot(token="", chat_id="")
        assert bot.poll_messages(timeout=1) == []
        mock_post.assert_not_called()


class TestHandleCommands:
    """Test command handling with a mock queue module."""

    def _make_bot_and_mock(self):
        bot = TelegramBot(token="tok", chat_id="123")
        bot.send = MagicMock(return_value={"message_id": 1})
        bot.reply = MagicMock(return_value={"message_id": 1})
        return bot

    def _mock_queue(self, jobs=None, queue_size=0):
        Q = MagicMock()
        Q.list_jobs = MagicMock(return_value=jobs or [])
        Q.queue_size = MagicMock(return_value=queue_size)
        Q.clear_done = MagicMock(return_value=0)
        Q.add = MagicMock(return_value={"id": "123-test", "subject": "test"})
        Q.mark_failed = MagicMock()
        return Q

    def test_plain_text_returned_as_topics(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()

        msgs = [
            IncomingMessage("React hooks", 1, False, "", ""),
            IncomingMessage("FastAPI tips", 2, False, "", ""),
        ]
        topics = bot.handle_commands(msgs, Q)
        assert len(topics) == 2
        assert topics[0].text == "React hooks"
        assert topics[1].text == "FastAPI tips"
        # No replies for plain text (caller handles those)
        bot.reply.assert_not_called()

    def test_help_command(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()

        msgs = [IncomingMessage("/help", 1, True, "help", "")]
        topics = bot.handle_commands(msgs, Q)
        assert topics == []
        bot.reply.assert_called_once()
        reply_text = bot.reply.call_args.args[1]
        assert "/jobs" in reply_text
        assert "/status" in reply_text

    def test_jobs_command_no_jobs(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue(jobs=[])

        msgs = [IncomingMessage("/jobs", 1, True, "jobs", "")]
        bot.handle_commands(msgs, Q)
        reply_text = bot.reply.call_args.args[1]
        assert "No jobs" in reply_text

    def test_jobs_command_with_jobs(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue(jobs=[
            {"status": "running", "subject": "React hooks"},
            {"status": "queued", "subject": "FastAPI"},
            {"status": "done", "subject": "Python async"},
        ])

        msgs = [IncomingMessage("/jobs", 1, True, "jobs", "")]
        bot.handle_commands(msgs, Q)
        reply_text = bot.reply.call_args.args[1]
        assert "React hooks" in reply_text
        assert "FastAPI" in reply_text

    def test_status_command(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue(queue_size=3, jobs=[{}, {}, {}, {}, {}])

        stats = {"total_turns": 10, "total_tool_calls": 25, "kb_stats": {"entry_count": 5, "total_tokens": 1000}}
        msgs = [IncomingMessage("/status", 1, True, "status", "")]
        bot.handle_commands(msgs, Q, agent_stats=stats)
        reply_text = bot.reply.call_args.args[1]
        assert "3 pending" in reply_text
        assert "5 total" in reply_text
        assert "25" in reply_text  # tool calls

    def test_queue_command(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()

        msgs = [IncomingMessage("/queue React hooks", 1, True, "queue", "React hooks")]
        bot.handle_commands(msgs, Q)
        Q.add.assert_called_once_with("research", "React hooks")
        reply_text = bot.reply.call_args.args[1]
        assert "Queued" in reply_text

    def test_queue_command_no_args(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()

        msgs = [IncomingMessage("/queue", 1, True, "queue", "")]
        bot.handle_commands(msgs, Q)
        Q.add.assert_not_called()
        reply_text = bot.reply.call_args.args[1]
        assert "Usage" in reply_text

    def test_cancel_command(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue(jobs=[])
        Q.list_jobs = MagicMock(return_value=[
            {"id": "123-react-hooks", "status": "queued", "subject": "React hooks"},
        ])

        msgs = [IncomingMessage("/cancel 123-react", 1, True, "cancel", "123-react")]
        bot.handle_commands(msgs, Q)
        Q.mark_failed.assert_called_once()
        reply_text = bot.reply.call_args.args[1]
        assert "Cancelled" in reply_text

    def test_cancel_no_match(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue(jobs=[])
        Q.list_jobs = MagicMock(return_value=[])

        msgs = [IncomingMessage("/cancel bogus", 1, True, "cancel", "bogus")]
        bot.handle_commands(msgs, Q)
        Q.mark_failed.assert_not_called()
        reply_text = bot.reply.call_args.args[1]
        assert "No queued job" in reply_text

    def test_unknown_command(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()

        msgs = [IncomingMessage("/foo", 1, True, "foo", "")]
        bot.handle_commands(msgs, Q)
        reply_text = bot.reply.call_args.args[1]
        assert "Unknown command" in reply_text

    def test_mixed_commands_and_topics(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()

        msgs = [
            IncomingMessage("React hooks", 1, False, "", ""),
            IncomingMessage("/jobs", 2, True, "jobs", ""),
            IncomingMessage("FastAPI tips", 3, False, "", ""),
        ]
        topics = bot.handle_commands(msgs, Q)
        assert len(topics) == 2
        assert topics[0].text == "React hooks"
        assert topics[1].text == "FastAPI tips"
        # /jobs was handled
        assert bot.reply.call_count == 1

    def test_clear_command(self):
        bot = self._make_bot_and_mock()
        Q = self._mock_queue()
        Q.clear_done = MagicMock(return_value=4)

        msgs = [IncomingMessage("/clear", 1, True, "clear", "")]
        bot.handle_commands(msgs, Q)
        Q.clear_done.assert_called_once()
        reply_text = bot.reply.call_args.args[1]
        assert "4" in reply_text
