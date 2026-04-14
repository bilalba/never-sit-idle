"""Tests for the Telegram bot integration."""

import json
import pytest
from unittest.mock import patch, MagicMock

from agent.telegram import TelegramBot


def _ok_response(result=None):
    resp = MagicMock()
    resp.json.return_value = {"ok": True, "result": result or {}}
    return resp


def _error_response(desc="Bad Request"):
    resp = MagicMock()
    resp.json.return_value = {"ok": False, "description": desc}
    return resp


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
        call_kwargs = mock_post.call_args
        assert "sendMessage" in call_kwargs.args[0]
        body = call_kwargs.kwargs["json"]
        assert body["chat_id"] == "123"
        assert body["text"] == "hello"

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


class TestPollReply:
    @patch("agent.telegram.requests.post")
    def test_returns_text_from_correct_chat(self, mock_post):
        mock_post.return_value = _ok_response([
            {
                "update_id": 100,
                "message": {
                    "chat": {"id": 123},
                    "text": "look into WebSockets",
                },
            }
        ])
        bot = TelegramBot(token="tok", chat_id="123")
        reply = bot.poll_reply(timeout=1)
        assert reply == "look into WebSockets"
        assert bot._update_offset == 101

    @patch("agent.telegram.requests.post")
    def test_ignores_other_chats(self, mock_post):
        mock_post.return_value = _ok_response([
            {
                "update_id": 200,
                "message": {
                    "chat": {"id": 999},
                    "text": "spam",
                },
            }
        ])
        bot = TelegramBot(token="tok", chat_id="123")
        reply = bot.poll_reply(timeout=1)
        assert reply is None
        # Still advances offset
        assert bot._update_offset == 201

    @patch("agent.telegram.requests.post")
    def test_returns_none_on_empty(self, mock_post):
        mock_post.return_value = _ok_response([])
        bot = TelegramBot(token="tok", chat_id="123")
        assert bot.poll_reply(timeout=1) is None

    @patch("agent.telegram.requests.post")
    def test_returns_none_when_disabled(self, mock_post):
        bot = TelegramBot(token="", chat_id="")
        assert bot.poll_reply(timeout=1) is None
        mock_post.assert_not_called()
