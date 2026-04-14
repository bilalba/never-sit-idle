"""Tests for the LLM client: token counting, retries, response parsing."""

import json
import pytest
from unittest.mock import patch, MagicMock

from agent.llm import (
    count_tokens,
    count_messages_tokens,
    chat_completion,
    extract_assistant_message,
    extract_tool_calls,
    _parse_tool_calls_from_content,
    LLMError,
    LLMRetryExhausted,
)


class TestTokenCounting:
    def test_count_tokens_basic(self):
        count = count_tokens("hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self):
        text = "word " * 1000
        count = count_tokens(text)
        assert count > 500

    def test_count_messages_tokens(self):
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Hello!"},
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_count_messages_with_tool_calls(self):
        messages = [
            {"role": "assistant", "content": "I'll help.",
             "tool_calls": [{"id": "1", "function": {"name": "test", "arguments": "{}"}}]},
        ]
        count = count_messages_tokens(messages)
        assert count > 0


class TestChatCompletion:
    def _mock_response(self, status_code=200, json_data=None, text=""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = text
        return resp

    @patch("agent.llm.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = self._mock_response(200, {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        })
        result = chat_completion(
            [{"role": "user", "content": "Hi"}],
            api_key="test-key",
            max_retries=1,
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @patch("agent.llm.config.OPENROUTER_API_KEY", "")
    @patch("agent.llm.requests.post")
    def test_missing_api_key(self, mock_post):
        with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
            chat_completion(
                [{"role": "user", "content": "Hi"}],
                api_key="",
                max_retries=1,
            )

    @patch("agent.llm._backoff")
    @patch("agent.llm.requests.post")
    def test_retry_on_429(self, mock_post, mock_backoff):
        # First call returns 429, second succeeds
        mock_post.side_effect = [
            self._mock_response(429, text="rate limited"),
            self._mock_response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
        result = chat_completion(
            [{"role": "user", "content": "Hi"}],
            api_key="test-key",
            max_retries=3,
        )
        assert result["choices"][0]["message"]["content"] == "ok"
        assert mock_post.call_count == 2

    @patch("agent.llm._backoff")
    @patch("agent.llm.requests.post")
    def test_retry_on_500(self, mock_post, mock_backoff):
        mock_post.side_effect = [
            self._mock_response(500, text="server error"),
            self._mock_response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
        result = chat_completion(
            [{"role": "user", "content": "Hi"}],
            api_key="test-key",
            max_retries=3,
        )
        assert mock_post.call_count == 2

    @patch("agent.llm._backoff")
    @patch("agent.llm.requests.post")
    def test_exhaust_retries(self, mock_post, mock_backoff):
        mock_post.return_value = self._mock_response(503, text="unavailable")
        with pytest.raises(LLMRetryExhausted) as exc_info:
            chat_completion(
                [{"role": "user", "content": "Hi"}],
                api_key="test-key",
                max_retries=2,
            )
        assert exc_info.value.attempts == 2

    @patch("agent.llm.requests.post")
    def test_non_retryable_error(self, mock_post):
        mock_post.return_value = self._mock_response(401, text="unauthorized")
        with pytest.raises(LLMError, match="401"):
            chat_completion(
                [{"role": "user", "content": "Hi"}],
                api_key="test-key",
                max_retries=3,
            )

    @patch("agent.llm.requests.post")
    def test_api_error_in_200(self, mock_post):
        mock_post.return_value = self._mock_response(200, {
            "error": {"message": "model not found"}
        })
        with pytest.raises(LLMError, match="model not found"):
            chat_completion(
                [{"role": "user", "content": "Hi"}],
                api_key="test-key",
                max_retries=1,
            )

    @patch("agent.llm._backoff")
    @patch("agent.llm.requests.post")
    def test_retry_on_timeout(self, mock_post, mock_backoff):
        import requests as req
        mock_post.side_effect = [
            req.exceptions.Timeout("timed out"),
            self._mock_response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
        result = chat_completion(
            [{"role": "user", "content": "Hi"}],
            api_key="test-key",
            max_retries=3,
        )
        assert mock_post.call_count == 2

    @patch("agent.llm._backoff")
    @patch("agent.llm.requests.post")
    def test_retry_on_connection_error(self, mock_post, mock_backoff):
        import requests as req
        mock_post.side_effect = [
            req.exceptions.ConnectionError("refused"),
            self._mock_response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
        result = chat_completion(
            [{"role": "user", "content": "Hi"}],
            api_key="test-key",
            max_retries=3,
        )
        assert mock_post.call_count == 2

    @patch("agent.llm.requests.post")
    def test_tools_passed_in_body(self, mock_post):
        mock_post.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "ok"}}]
        })
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        chat_completion(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
            api_key="test-key",
            max_retries=1,
        )
        call_body = mock_post.call_args[1]["json"]
        assert "tools" in call_body
        assert call_body["tool_choice"] == "auto"


class TestExtractAssistantMessage:
    def test_normal(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        msg = extract_assistant_message(resp)
        assert msg["content"] == "hi"

    def test_no_choices(self):
        with pytest.raises(LLMError, match="No choices"):
            extract_assistant_message({"choices": []})


class TestExtractToolCalls:
    def test_structured_tool_calls(self):
        msg = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "main.py"}'
                    },
                }
            ],
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"] == {"path": "main.py"}
        assert calls[0]["id"] == "call_1"

    def test_multiple_tool_calls(self):
        msg = {
            "content": "",
            "tool_calls": [
                {"id": "1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"id": "2", "function": {"name": "glob_files", "arguments": '{"pattern": "*.py"}'}},
            ],
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 2

    def test_fallback_json_in_content(self):
        msg = {
            "content": 'I\'ll read the file: {"tool": "read_file", "arguments": {"path": "main.py"}}',
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"

    def test_fallback_array_in_content(self):
        msg = {
            "content": 'Here are my actions: [{"tool": "read_file", "arguments": {"path": "a.py"}}, {"tool": "glob_files", "arguments": {"pattern": "*.py"}}]',
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        msg = {"content": "Just a normal response."}
        calls = extract_tool_calls(msg)
        assert calls == []

    def test_malformed_arguments(self):
        msg = {
            "content": "",
            "tool_calls": [
                {"id": "1", "function": {"name": "test", "arguments": "not json"}},
            ],
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"_raw": "not json"}

    def test_dict_arguments(self):
        msg = {
            "content": "",
            "tool_calls": [
                {"id": "1", "function": {"name": "test", "arguments": {"key": "val"}}},
            ],
        }
        calls = extract_tool_calls(msg)
        assert calls[0]["arguments"] == {"key": "val"}

    def test_gemma_tool_call_format(self):
        """Test Gemma's native <|tool_call> format with <|"|> delimiters."""
        msg = {
            "content": (
                'Some text before\n'
                '<|tool_call>call:kb_write{content:<|"|>Hello world<|"|>,'
                'path:<|"|>test/doc<|"|>,'
                'tags:[<|"|>python<|"|>,<|"|>test<|"|>]}<tool_call|>'
            ),
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "kb_write"
        assert calls[0]["arguments"]["path"] == "test/doc"
        assert "Hello world" in calls[0]["arguments"]["content"]

    def test_gemma_tool_call_multiline_content(self):
        """Test Gemma format with multiline content."""
        msg = {
            "content": (
                '<|tool_call>call:kb_write{content:<|"|># Title\n\nBody text here.<|"|>,'
                'path:<|"|>overview<|"|>}<tool_call|>'
            ),
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "kb_write"
        assert "# Title" in calls[0]["arguments"]["content"]

    def test_gemma_tool_call_with_inner_quotes(self):
        """Content with literal " chars inside <|"|> delimiters must not truncate."""
        msg = {
            "content": (
                '<|tool_call>call:kb_write{content:<|"|>'
                '# Overview\n\n'
                'Moving away from the traditional "all-on-the-client" model '
                'towards a "hybrid" approach.\n\n'
                '| Feature | "Old" | "New" |\n'
                '<|"|>,'
                'path:<|"|>react/overview<|"|>}<tool_call|>'
            ),
        }
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        args = calls[0]["arguments"]
        assert args["path"] == "react/overview"
        # The full content must survive, including the quoted phrases
        assert '"all-on-the-client"' in args["content"]
        assert '"hybrid"' in args["content"]
        assert '| Feature |' in args["content"]
