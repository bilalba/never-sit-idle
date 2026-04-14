"""Tests for the core agent loop."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agent.agent import Agent
from agent.llm import LLMError, LLMRetryExhausted


def _make_llm_response(content="", tool_calls=None):
    """Build a mock OpenRouter response."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _tool_call(tc_id, name, arguments):
    return {
        "id": tc_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class TestAgent:
    @pytest.fixture
    def agent(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "main.py").write_text("print('hello')\n")
        (code_dir / "lib").mkdir()
        (code_dir / "lib" / "utils.py").write_text("def add(a, b): return a + b\n")

        return Agent(
            codebase_root=code_dir,
            kb_dir=str(tmp_path / "kb"),
            max_turns=10,
        )

    @patch("agent.agent.chat_completion")
    def test_simple_response(self, mock_llm, agent):
        mock_llm.return_value = _make_llm_response("Hello! I'm the agent.")
        result = agent.run("Hi")
        assert "Hello" in result

    @patch("agent.agent.chat_completion")
    def test_tool_call_cycle(self, mock_llm, agent):
        # Turn 1: LLM calls a tool
        # Turn 2: LLM responds with text
        mock_llm.side_effect = [
            _make_llm_response(
                content="Let me read the file.",
                tool_calls=[_tool_call("call_1", "read_file", {"path": "main.py"})],
            ),
            _make_llm_response("The file contains a print statement."),
        ]
        result = agent.run("What's in main.py?")
        assert "print" in result
        assert mock_llm.call_count == 2

    @patch("agent.agent.chat_completion")
    def test_multiple_tool_calls(self, mock_llm, agent):
        mock_llm.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[
                    _tool_call("call_1", "read_file", {"path": "main.py"}),
                    _tool_call("call_2", "glob_files", {"pattern": "**/*.py"}),
                ],
            ),
            _make_llm_response("Found 2 Python files."),
        ]
        result = agent.run("Show me the codebase")
        assert agent.total_tool_calls == 2

    @patch("agent.agent.chat_completion")
    def test_kb_write_via_tool(self, mock_llm, agent):
        mock_llm.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[_tool_call("call_1", "kb_write", {
                    "path": "overview",
                    "content": "# Overview\nThis is a test project.",
                })],
            ),
            _make_llm_response("I've written the overview."),
        ]
        agent.run("Document this project")
        # Verify KB was written
        content = agent.kb.read_entry("overview")
        assert content is not None
        assert "test project" in content

    @patch("agent.agent.chat_completion")
    def test_memory_store_via_tool(self, mock_llm, agent):
        mock_llm.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[_tool_call("call_1", "memory_store", {
                    "category": "facts",
                    "name": "language",
                    "content": "Project uses Python",
                })],
            ),
            _make_llm_response("Noted."),
        ]
        agent.run("Remember this uses Python")
        assert agent.ltm.retrieve("facts", "language") == "Project uses Python"

    @patch("agent.agent.chat_completion")
    def test_working_memory(self, mock_llm, agent):
        mock_llm.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[_tool_call("call_1", "wm_set", {
                    "key": "plan", "value": "Step 1: read files",
                })],
            ),
            _make_llm_response("Plan stored."),
        ]
        agent.run("Plan the exploration")
        assert agent.wm.get("plan") == "Step 1: read files"

    @patch("agent.agent.chat_completion")
    def test_max_turns_limit(self, mock_llm, agent):
        # Always return tool calls — agent should stop at max_turns
        mock_llm.return_value = _make_llm_response(
            content="still working",
            tool_calls=[_tool_call("call_1", "think", {"thought": "thinking..."})],
        )
        agent.max_turns = 3
        result = agent.run("Do something")
        assert agent.total_turns == 3
        assert "maximum turns" in result.lower()

    @patch("agent.agent.chat_completion")
    def test_llm_error_handled(self, mock_llm, agent):
        mock_llm.side_effect = LLMError("API key invalid")
        result = agent.run("Hi")
        assert "error" in result.lower()

    @patch("agent.agent.chat_completion")
    def test_llm_retry_exhausted_handled(self, mock_llm, agent):
        mock_llm.side_effect = LLMRetryExhausted(
            last_error=Exception("timeout"), attempts=3
        )
        result = agent.run("Hi")
        assert "error" in result.lower()
        assert "3 retries" in result

    @patch("agent.agent.chat_completion")
    def test_stats(self, mock_llm, agent):
        mock_llm.return_value = _make_llm_response("Done.")
        agent.run("Hi")
        stats = agent.stats()
        assert stats["total_turns"] == 1
        assert stats["conversation_messages"] == 2  # user + assistant

    @patch("agent.agent.chat_completion")
    def test_reset_conversation(self, mock_llm, agent):
        mock_llm.return_value = _make_llm_response("Done.")
        agent.run("Hi")
        agent.wm.set("key", "val")
        agent.task_dir = "some-task"
        agent.reset_conversation()
        assert len(agent.messages) == 0
        assert agent.wm.list_keys() == []
        assert agent.task_dir is None

    @patch("agent.agent.chat_completion")
    def test_memory_injected_in_messages(self, mock_llm, agent):
        # Store some LTM
        agent.ltm.store("facts", "test", "Important fact")
        agent.wm.set("plan", "Current plan")

        mock_llm.return_value = _make_llm_response("Done.")
        agent.run("Hi")

        # Check that LTM and WM were in the messages sent to the LLM
        call_args = mock_llm.call_args
        messages = call_args[0][0]
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "Important fact" in all_content
        assert "Current plan" in all_content


class TestTaskNaming:
    @pytest.fixture
    def agent(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        return Agent(codebase_root=code_dir, kb_dir=str(tmp_path / "kb"), max_turns=5)

    @patch("agent.agent.chat_completion")
    def test_name_task_basic(self, mock_llm, agent):
        mock_llm.return_value = _make_llm_response("react-hooks-research")
        slug = agent.name_task("Research: React hooks")
        assert slug == "react-hooks-research"
        assert agent.task_dir == "react-hooks-research"
        # Directory should be created
        assert (Path(agent.kb.base_dir) / "react-hooks-research").is_dir()

    @patch("agent.agent.chat_completion")
    def test_name_task_sanitizes(self, mock_llm, agent):
        mock_llm.return_value = _make_llm_response("My Task Name!!! (v2)")
        slug = agent.name_task("Some task")
        assert slug == "my-task-name-v2"
        assert " " not in slug
        assert "!" not in slug

    @patch("agent.agent.chat_completion")
    def test_name_task_empty_fallback(self, mock_llm, agent):
        mock_llm.return_value = _make_llm_response("")
        slug = agent.name_task("Some task")
        assert slug == "task"

    @patch("agent.agent.chat_completion")
    def test_task_dir_injected_in_messages(self, mock_llm, agent):
        agent.task_dir = "my-research"
        mock_llm.return_value = _make_llm_response("Done.")
        agent.run("Hi")
        call_args = mock_llm.call_args
        messages = call_args[0][0]
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "my-research/" in all_content
        assert "memory_store" in all_content

    @patch("agent.agent.chat_completion")
    def test_no_task_dir_no_injection(self, mock_llm, agent):
        """When task_dir is None, no task directory context is injected."""
        mock_llm.return_value = _make_llm_response("Done.")
        agent.run("Hi")
        call_args = mock_llm.call_args
        messages = call_args[0][0]
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "Current Task Directory" not in all_content


class TestAgentStreaming:
    @pytest.fixture
    def agent(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "test.py").write_text("pass\n")
        return Agent(codebase_root=code_dir, kb_dir=str(tmp_path / "kb"), max_turns=5)

    @patch("agent.agent.chat_completion")
    def test_streaming_events(self, mock_llm, agent):
        mock_llm.side_effect = [
            _make_llm_response(
                content="Reading file.",
                tool_calls=[_tool_call("c1", "read_file", {"path": "test.py"})],
            ),
            _make_llm_response("File contains 'pass'."),
        ]

        events = list(agent.run_streaming("Read test.py"))
        types = [e["type"] for e in events]
        assert "text" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "done" in types

    @patch("agent.agent.chat_completion")
    def test_streaming_error(self, mock_llm, agent):
        mock_llm.side_effect = LLMError("bad")
        events = list(agent.run_streaming("Hi"))
        assert any(e["type"] == "error" for e in events)
