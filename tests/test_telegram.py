"""Tests for the Telegram bot integration."""

import json
import logging
import pytest
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace

from agent.telegram import TelegramBot, IncomingMessage, _split_message


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


class TestSplitMessage:
    def test_short_message_not_split(self):
        assert _split_message("hello") == ["hello"]

    def test_exact_limit_not_split(self):
        text = "a" * 4096
        assert _split_message(text) == [text]

    def test_splits_at_newline(self):
        # Two blocks separated by a newline near the limit
        block1 = "a" * 100
        block2 = "b" * 100
        text = block1 + "\n" + block2
        chunks = _split_message(text, limit=105)
        assert len(chunks) == 2
        assert chunks[0] == block1
        assert chunks[1] == block2

    def test_hard_split_when_no_newline(self):
        text = "a" * 200
        chunks = _split_message(text, limit=80)
        assert len(chunks) == 3
        assert chunks[0] == "a" * 80
        assert chunks[1] == "a" * 80
        assert chunks[2] == "a" * 40

    def test_empty_string(self):
        assert _split_message("") == [""]

    def test_multiple_chunks(self):
        lines = [f"Line {i}" for i in range(100)]
        text = "\n".join(lines)
        chunks = _split_message(text, limit=100)
        assert len(chunks) > 1
        # All original content preserved
        reassembled = "\n".join(chunks)
        assert reassembled == text


class TestSendLong:
    @patch("agent.telegram.requests.post")
    def test_short_message_single_send(self, mock_post):
        mock_post.return_value = _ok_response({"message_id": 1})
        bot = TelegramBot(token="tok", chat_id="123")
        bot.send_long("short message", reply_to=42)
        assert mock_post.call_count == 1
        body = mock_post.call_args.kwargs["json"]
        assert body["reply_to_message_id"] == 42

    @patch("agent.telegram.requests.post")
    def test_long_message_multiple_sends(self, mock_post):
        mock_post.return_value = _ok_response({"message_id": 1})
        bot = TelegramBot(token="tok", chat_id="123")
        text = "a" * 100 + "\n" + "b" * 100
        bot.send_long(text, reply_to=42, limit=105)
        assert mock_post.call_count == 2
        # First call has reply_to, second doesn't
        first_body = mock_post.call_args_list[0].kwargs["json"]
        assert first_body["reply_to_message_id"] == 42
        second_body = mock_post.call_args_list[1].kwargs["json"]
        assert "reply_to_message_id" not in second_body


class TestHandleTelegramMessage:
    """Tests for the _handle_telegram_message helper in cli.py."""

    def test_routes_message_through_agent(self):
        from agent.cli import _handle_telegram_message
        from agent.prompts import TELEGRAM_CHAT_SYSTEM

        agent = MagicMock()
        agent.run.return_value = "Here's what I know about React hooks..."
        agent.system_prompt = MagicMock()

        bot = MagicMock()
        bot.send.return_value = None
        msg = IncomingMessage("what do we know about React hooks?", 10, False, "", "")
        logger = logging.getLogger("test")

        _handle_telegram_message(agent, bot, msg, logger)

        # Agent called with message text and streaming callback
        agent.run.assert_called_once()
        assert agent.run.call_args.args[0] == "what do we know about React hooks?"
        assert agent.run.call_args.kwargs.get("on_text_delta") is not None
        # Typing indicator sent
        bot.send_typing.assert_called_once()
        # Final response sent (via streamer)
        bot.send.assert_called()
        # Should have set the Telegram system prompt
        assert agent.system_prompt.text == TELEGRAM_CHAT_SYSTEM

    def test_handles_agent_error_gracefully(self):
        from agent.cli import _handle_telegram_message

        agent = MagicMock()
        agent.run.side_effect = RuntimeError("LLM exploded")
        agent.system_prompt = MagicMock()

        bot = MagicMock()
        bot.send.return_value = None
        msg = IncomingMessage("test", 11, False, "", "")
        logger = logging.getLogger("test")

        _handle_telegram_message(agent, bot, msg, logger)

        # Error sent via streamer (bot.send with error text)
        bot.send.assert_called()
        send_text = bot.send.call_args.args[0]
        assert "something went wrong" in send_text

    def test_handles_empty_response(self):
        from agent.cli import _handle_telegram_message

        agent = MagicMock()
        agent.run.return_value = "  "
        agent.system_prompt = MagicMock()

        bot = MagicMock()
        bot.send.return_value = None
        msg = IncomingMessage("test", 12, False, "", "")
        logger = logging.getLogger("test")

        _handle_telegram_message(agent, bot, msg, logger)

        # "(No response)" sent via streamer
        bot.send.assert_called()
        send_text = bot.send.call_args.args[0]
        assert "No response" in send_text

    def test_does_not_reset_conversation(self):
        """Telegram messages should keep conversation context."""
        from agent.cli import _handle_telegram_message

        agent = MagicMock()
        agent.run.return_value = "answer"
        agent.system_prompt = MagicMock()

        bot = MagicMock()
        bot.send.return_value = None
        msg = IncomingMessage("test", 13, False, "", "")
        logger = logging.getLogger("test")

        _handle_telegram_message(agent, bot, msg, logger)

        agent.reset_conversation.assert_not_called()


class TestTelegramMemory:
    """Tests for TelegramMemory conversation context management."""

    def test_add_and_get_context(self, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        mem.add_exchange("hello", "hi there")
        mem.add_exchange("what is 2+2?", "4")

        msgs = mem.get_context_messages()
        assert len(msgs) == 4  # 2 user + 2 assistant
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_persistence(self, tmp_path):
        from agent.telegram_memory import TelegramMemory

        path = tmp_path / "mem.json"
        mem1 = TelegramMemory(persist_path=path)
        mem1.add_exchange("q1", "a1")
        mem1.add_exchange("q2", "a2")

        # Load from same path
        mem2 = TelegramMemory(persist_path=path)
        assert len(mem2.exchanges) == 2
        assert mem2.exchanges[0]["user"] == "q1"

    def test_needs_compaction(self, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        for i in range(3):
            mem.add_exchange(f"q{i}", f"a{i}")
        assert not mem.needs_compaction()  # 3 <= RECENT_LIMIT (4)

        mem.add_exchange("q3", "a3")
        mem.add_exchange("q4", "a4")
        assert mem.needs_compaction()  # 5 > 4

    @patch("agent.telegram_memory.chat_completion")
    @patch("agent.telegram_memory.extract_assistant_message")
    def test_compact_summarizes_older(self, mock_extract, mock_chat, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mock_extract.return_value = {"content": "Summary: user asked several questions."}
        mock_chat.return_value = {}

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        for i in range(6):
            mem.add_exchange(f"question {i}", f"answer {i}")

        mem.compact()

        # Should keep last RECENT_LIMIT exchanges
        assert len(mem.exchanges) == TelegramMemory.RECENT_LIMIT
        assert mem.exchanges[0]["user"] == "question 2"
        assert mem.summary == "Summary: user asked several questions."
        mock_chat.assert_called_once()

    @patch("agent.telegram_memory.chat_completion")
    @patch("agent.telegram_memory.extract_assistant_message")
    def test_compact_includes_existing_summary(self, mock_extract, mock_chat, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mock_extract.return_value = {"content": "Updated summary."}
        mock_chat.return_value = {}

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        mem.summary = "Old summary from earlier."
        for i in range(6):
            mem.add_exchange(f"q{i}", f"a{i}")

        mem.compact()

        # Prompt should include the existing summary
        prompt = mock_chat.call_args[0][0][1]["content"]
        assert "Old summary from earlier." in prompt

    def test_context_includes_summary(self, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        mem.summary = "User likes Python."
        mem.add_exchange("what's new?", "not much")

        msgs = mem.get_context_messages()
        assert len(msgs) == 3  # summary system msg + 1 user + 1 assistant
        assert msgs[0]["role"] == "system"
        assert "User likes Python." in msgs[0]["content"]

    @patch("agent.telegram_memory.chat_completion", side_effect=RuntimeError("LLM down"))
    def test_compact_failure_preserves_messages(self, mock_chat, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        for i in range(6):
            mem.add_exchange(f"q{i}", f"a{i}")

        mem.compact()

        # All messages should be preserved on failure
        assert len(mem.exchanges) == 6

    def test_clear(self, tmp_path):
        from agent.telegram_memory import TelegramMemory

        mem = TelegramMemory(persist_path=tmp_path / "mem.json")
        mem.summary = "some history"
        mem.add_exchange("q", "a")
        mem.clear()

        assert mem.summary == ""
        assert len(mem.exchanges) == 0

    def test_handle_message_with_telegram_mem(self):
        """_handle_telegram_message should inject context and record exchange."""
        from agent.cli import _handle_telegram_message
        from agent.telegram_memory import TelegramMemory

        mem = TelegramMemory()
        mem.add_exchange("earlier question", "earlier answer")

        agent = MagicMock()
        agent.run.return_value = "new answer"
        agent.system_prompt = MagicMock()

        bot = MagicMock()
        bot.send.return_value = None
        msg = IncomingMessage("new question", 20, False, "", "")
        logger = logging.getLogger("test")

        _handle_telegram_message(agent, bot, msg, logger, telegram_mem=mem)

        # Agent should have gotten context messages injected
        assert len(agent.messages) == 2  # earlier user + earlier assistant
        # Exchange should be recorded
        assert len(mem.exchanges) == 2
        assert mem.exchanges[1]["user"] == "new question"
        assert mem.exchanges[1]["assistant"] == "new answer"
