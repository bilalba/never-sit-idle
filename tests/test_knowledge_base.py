"""Tests for the knowledge base manager."""

import json
import pytest
from pathlib import Path

from agent.knowledge_base import KnowledgeBase


class TestKnowledgeBase:
    @pytest.fixture
    def kb(self, tmp_path):
        return KnowledgeBase(base_dir=str(tmp_path / "kb"))

    def test_write_and_read(self, kb):
        kb.write_entry("overview", "# Project Overview\nThis is a test.")
        content = kb.read_entry("overview")
        assert content == "# Project Overview\nThis is a test."

    def test_read_nonexistent(self, kb):
        assert kb.read_entry("nope") is None

    def test_nested_path(self, kb):
        kb.write_entry("architecture/data_flow", "# Data Flow\nA -> B -> C")
        content = kb.read_entry("architecture/data_flow")
        assert "A -> B -> C" in content

    def test_deeply_nested(self, kb):
        kb.write_entry("modules/auth/endpoints/login", "# Login Endpoint")
        content = kb.read_entry("modules/auth/endpoints/login")
        assert content == "# Login Endpoint"

    def test_md_suffix_handling(self, kb):
        # Should work with or without .md
        kb.write_entry("test.md", "with suffix")
        assert kb.read_entry("test") == "with suffix"
        assert kb.read_entry("test.md") == "with suffix"

    def test_overwrite(self, kb):
        kb.write_entry("doc", "version 1")
        kb.write_entry("doc", "version 2")
        assert kb.read_entry("doc") == "version 2"

    def test_delete(self, kb):
        kb.write_entry("to_delete", "temp")
        assert kb.delete_entry("to_delete") is True
        assert kb.read_entry("to_delete") is None

    def test_delete_nonexistent(self, kb):
        assert kb.delete_entry("nope") is False

    def test_delete_cleans_empty_dirs(self, kb):
        kb.write_entry("deep/nested/entry", "content")
        kb.delete_entry("deep/nested/entry")
        # Parent dirs should be cleaned up
        assert not (kb.base_dir / "deep" / "nested").exists()

    def test_list_entries(self, kb):
        kb.write_entry("overview", "ov")
        kb.write_entry("arch/data", "data")
        kb.write_entry("arch/deploy", "deploy")
        kb.write_entry("modules/auth", "auth")

        all_entries = kb.list_entries()
        assert len(all_entries) == 4

        arch_entries = kb.list_entries("arch")
        assert len(arch_entries) == 2

    def test_list_returns_metadata(self, kb):
        kb.write_entry("doc", "hello world", tags=["test"])
        entries = kb.list_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert "path" in entry
        assert "tokens" in entry
        assert entry["tokens"] > 0
        assert "tags" in entry

    def test_tree(self, kb):
        kb.write_entry("overview", "ov")
        kb.write_entry("architecture/overview", "arch")
        kb.write_entry("modules/auth/overview", "auth")

        tree = kb.tree()
        assert "knowledge_base/" in tree
        assert "overview.md" in tree

    def test_tree_empty(self, kb):
        assert "empty" in kb.tree().lower()

    def test_search_keyword(self, kb):
        kb.write_entry("python_guide", "Python is great for scripting")
        kb.write_entry("rust_guide", "Rust is great for systems")

        results = kb.search("Python")
        assert len(results) == 1
        assert "python" in results[0]["path"].lower()

    def test_search_in_content(self, kb):
        kb.write_entry("guide", "The frobnicator processes widgets")
        results = kb.search("frobnicator")
        assert len(results) == 1

    def test_search_by_tags(self, kb):
        kb.write_entry("doc1", "first", tags=["important"])
        kb.write_entry("doc2", "second", tags=["draft"])
        kb.write_entry("doc3", "third", tags=["important", "draft"])

        results = kb.search("", tags=["important"])
        assert len(results) == 2

    def test_search_keyword_and_tags(self, kb):
        kb.write_entry("doc1", "python rocks", tags=["lang"])
        kb.write_entry("doc2", "python sucks", tags=["rant"])

        results = kb.search("python", tags=["lang"])
        assert len(results) == 1

    def test_stats(self, kb):
        kb.write_entry("overview", "hello")
        kb.write_entry("arch/design", "design doc")
        stats = kb.stats()
        assert stats["entry_count"] == 2
        assert stats["total_tokens"] > 0
        assert "arch" in stats["categories"]

    def test_write_returns_metadata(self, kb):
        meta = kb.write_entry("doc", "hello world", tags=["test"])
        assert "tokens" in meta
        assert meta["tokens"] > 0
        assert "updated" in meta
        assert meta["tags"] == ["test"]

    def test_index_persistence(self, tmp_path):
        dir_path = str(tmp_path / "kb")
        kb1 = KnowledgeBase(base_dir=dir_path)
        kb1.write_entry("persistent", "I survive restarts")

        kb2 = KnowledgeBase(base_dir=dir_path)
        assert kb2.read_entry("persistent") == "I survive restarts"
        entries = kb2.list_entries()
        assert len(entries) == 1
