"""Tests for the tool registry and file-system tools."""

import json
import pytest
from pathlib import Path

from agent.knowledge_base import KnowledgeBase
from agent.memory import LongTermMemory, WorkingMemory
from agent.tools import ToolRegistry, build_registry, _read_file, _glob_files, _grep_files


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        schema = {
            "type": "function",
            "function": {"name": "test_tool", "description": "A test", "parameters": {}},
        }
        reg.register(schema, lambda: "ok")
        assert "test_tool" in reg.list_names()

    def test_execute(self):
        reg = ToolRegistry()
        schema = {
            "type": "function",
            "function": {"name": "greet", "description": "Greet", "parameters": {}},
        }
        reg.register(schema, lambda name="world": f"Hello {name}")
        result = reg.execute("greet", {"name": "Alice"})
        assert result == "Hello Alice"

    def test_execute_returns_json_for_dicts(self):
        reg = ToolRegistry()
        schema = {
            "type": "function",
            "function": {"name": "info", "description": "Info", "parameters": {}},
        }
        reg.register(schema, lambda: {"key": "value"})
        result = reg.execute("info", {})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = json.loads(reg.execute("nonexistent", {}))
        assert "error" in result

    def test_execute_handles_exception(self):
        reg = ToolRegistry()
        schema = {
            "type": "function",
            "function": {"name": "boom", "description": "Boom", "parameters": {}},
        }
        reg.register(schema, lambda: 1 / 0)
        result = json.loads(reg.execute("boom", {}))
        assert "error" in result
        assert "ZeroDivisionError" in result["error"]

    def test_get_schemas(self):
        reg = ToolRegistry()
        schema = {
            "type": "function",
            "function": {"name": "test", "description": "Test", "parameters": {}},
        }
        reg.register(schema, lambda: None)
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test"


class TestFileTools:
    @pytest.fixture
    def codebase(self, tmp_path):
        # Create a mock codebase
        (tmp_path / "main.py").write_text("print('hello')\nprint('world')\n")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "utils.py").write_text("def add(a, b):\n    return a + b\n")
        (tmp_path / "lib" / "__init__.py").write_text("")
        (tmp_path / "README.md").write_text("# My Project\n\nA cool project.\n")
        (tmp_path / "data.json").write_text('{"key": "value"}')
        return tmp_path

    def test_read_file(self, codebase):
        result = _read_file(codebase, "main.py", 200)
        assert result["content"] == "print('hello')\nprint('world')"
        assert result["lines"] == 2
        assert result["truncated"] is False

    def test_read_file_truncated(self, codebase):
        result = _read_file(codebase, "main.py", 1)
        assert result["truncated"] is True
        assert "hello" in result["content"]

    def test_read_file_not_found(self, codebase):
        result = _read_file(codebase, "nope.py", 200)
        assert "error" in result

    def test_read_file_path_escape(self, codebase):
        result = _read_file(codebase, "../../etc/passwd", 200)
        assert "error" in result

    def test_read_file_directory(self, codebase):
        result = _read_file(codebase, "lib", 200)
        assert "error" in result

    def test_glob_files(self, codebase):
        result = _glob_files(codebase, "**/*.py")
        assert any("main.py" in r for r in result)
        assert any("utils.py" in r for r in result)

    def test_glob_files_specific(self, codebase):
        result = _glob_files(codebase, "*.md")
        assert len(result) == 1
        assert "README.md" in result[0]

    def test_grep_files(self, codebase):
        result = _grep_files(codebase, "def add", "*")
        assert len(result) >= 1
        assert any("utils.py" in r["file"] for r in result)

    def test_grep_files_regex(self, codebase):
        result = _grep_files(codebase, r"print\(.+\)", "*")
        assert len(result) == 2

    def test_grep_invalid_regex(self, codebase):
        result = _grep_files(codebase, "[invalid", "*")
        assert len(result) == 1
        assert "error" in result[0]


class TestBuiltRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        kb = KnowledgeBase(base_dir=str(tmp_path / "kb"))
        ltm = LongTermMemory(memory_dir=str(tmp_path / "kb"), max_tokens=5000)
        wm = WorkingMemory(max_tokens=2000)
        # Create a small codebase
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "app.py").write_text("# Main app\nprint('running')\n")
        return build_registry(kb, ltm, wm, code_dir)

    def test_has_all_expected_tools(self, registry):
        names = registry.list_names()
        expected = [
            "read_file", "glob_files", "grep_files",
            "kb_write", "kb_read", "kb_delete", "kb_list", "kb_search", "kb_tree",
            "memory_store", "memory_retrieve", "memory_delete", "memory_list", "memory_search",
            "wm_set", "wm_get", "wm_delete", "wm_list",
            "reddit_search", "reddit_top", "reddit_comments",
            "wikipedia_search", "wikipedia_article",
            "hackernews_search", "hackernews_top",
            "stackexchange_search", "stackexchange_answers",
            "github_search_repos", "github_readme",
            "web_fetch",
            "rate_limit_stats", "think",
        ]
        for name in expected:
            assert name in names, f"Missing tool: {name}"

    def test_schemas_are_valid(self, registry):
        schemas = registry.get_schemas()
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_read_file_tool(self, registry):
        result = json.loads(registry.execute("read_file", {"path": "app.py"}))
        assert "Main app" in result["content"]

    def test_kb_write_and_read(self, registry):
        registry.execute("kb_write", {"path": "test_entry", "content": "Hello KB"})
        result = registry.execute("kb_read", {"path": "test_entry"})
        assert "Hello KB" in result

    def test_memory_store_and_retrieve(self, registry):
        registry.execute("memory_store", {
            "category": "test", "name": "fact1", "content": "Test fact",
        })
        result = registry.execute("memory_retrieve", {"category": "test", "name": "fact1"})
        assert "Test fact" in result

    def test_wm_set_and_get(self, registry):
        registry.execute("wm_set", {"key": "plan", "value": "Step 1: test"})
        result = registry.execute("wm_get", {"key": "plan"})
        assert "Step 1" in result

    def test_think_tool(self, registry):
        result = json.loads(registry.execute("think", {"thought": "I should research this"}))
        assert result["acknowledged"] is True

    def test_rate_limit_stats(self, registry):
        result = json.loads(registry.execute("rate_limit_stats", {}))
        assert isinstance(result, list)
        names = [s["name"] for s in result]
        assert "reddit" in names
        assert "wikipedia" in names
