"""Tests for the three-tier memory system."""

import json
import pytest
from pathlib import Path

from agent.memory import SystemPrompt, LongTermMemory, WorkingMemory


class TestSystemPrompt:
    def test_basic(self):
        sp = SystemPrompt("You are a helpful agent.")
        assert sp.text == "You are a helpful agent."
        assert sp.token_count > 0

    def test_to_message(self):
        sp = SystemPrompt("Hello")
        msg = sp.to_message()
        assert msg["role"] == "system"
        assert msg["content"] == "Hello"

    def test_set_text(self):
        sp = SystemPrompt("first")
        sp.text = "second"
        assert sp.text == "second"

    def test_empty(self):
        sp = SystemPrompt("")
        assert sp.token_count == 0


class TestLongTermMemory:
    @pytest.fixture
    def ltm(self, tmp_path):
        return LongTermMemory(memory_dir=str(tmp_path / "kb"), max_tokens=1000)

    def test_store_and_retrieve(self, ltm):
        ltm.store("facts", "test_entry", "This is a test fact.")
        result = ltm.retrieve("facts", "test_entry")
        assert result == "This is a test fact."

    def test_retrieve_nonexistent(self, ltm):
        assert ltm.retrieve("nope", "nada") is None

    def test_delete(self, ltm):
        ltm.store("facts", "to_delete", "temp")
        assert ltm.delete("facts", "to_delete") is True
        assert ltm.retrieve("facts", "to_delete") is None

    def test_delete_nonexistent(self, ltm):
        assert ltm.delete("nope", "nada") is False

    def test_token_budget(self, ltm):
        # Store something
        ltm.store("facts", "entry1", "hello " * 50)
        tokens_used = ltm.total_tokens
        assert tokens_used > 0
        assert ltm.remaining_tokens == ltm.max_tokens - tokens_used

    def test_budget_exceeded(self, ltm):
        # Fill up with a large entry
        big_content = "word " * 500  # Should be ~500 tokens
        ltm.store("facts", "big", big_content)
        # Try to store another large entry that exceeds budget
        with pytest.raises(ValueError, match="Memory budget exceeded"):
            ltm.store("facts", "big2", big_content)

    def test_update_entry_reuses_budget(self, ltm):
        ltm.store("facts", "entry", "hello " * 50)
        tokens_before = ltm.total_tokens
        # Update with same-ish content should not double-count
        ltm.store("facts", "entry", "world " * 50)
        tokens_after = ltm.total_tokens
        assert abs(tokens_after - tokens_before) < 10  # roughly same

    def test_list_entries(self, ltm):
        ltm.store("facts", "a", "aaa")
        ltm.store("facts", "b", "bbb")
        ltm.store("context", "c", "ccc")

        all_entries = ltm.list_entries()
        assert len(all_entries) == 3

        facts_only = ltm.list_entries("facts")
        assert len(facts_only) == 2

    def test_search(self, ltm):
        ltm.store("facts", "python", "Python is a programming language")
        ltm.store("facts", "rust", "Rust is a systems language")
        results = ltm.search("Python")
        assert len(results) == 1
        assert "python" in results[0]["key"].lower()

    def test_get_context_block(self, ltm):
        ltm.store("facts", "one", "First fact")
        ltm.store("facts", "two", "Second fact")
        block = ltm.get_context_block()
        assert "First fact" in block
        assert "Second fact" in block

    def test_index_persistence(self, tmp_path):
        dir_path = str(tmp_path / "kb")
        ltm1 = LongTermMemory(memory_dir=dir_path, max_tokens=5000)
        ltm1.store("facts", "persist", "I should persist")

        # New instance reads from disk
        ltm2 = LongTermMemory(memory_dir=dir_path, max_tokens=5000)
        assert ltm2.retrieve("facts", "persist") == "I should persist"
        assert ltm2.total_tokens > 0

    def test_rebuild_index(self, ltm):
        ltm.store("facts", "entry", "content")
        # Corrupt the index
        index_path = ltm.memory_dir / "_index.json"
        index_path.write_text("not json")
        # Reload — should rebuild
        ltm._load_index()
        assert ltm.total_tokens > 0


class TestWorkingMemory:
    @pytest.fixture
    def wm(self):
        return WorkingMemory(max_tokens=500)

    def test_set_and_get(self, wm):
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_nonexistent(self, wm):
        assert wm.get("nope") is None

    def test_delete(self, wm):
        wm.set("key", "val")
        assert wm.delete("key") is True
        assert wm.get("key") is None

    def test_delete_nonexistent(self, wm):
        assert wm.delete("nope") is False

    def test_list_keys(self, wm):
        wm.set("a", "1")
        wm.set("b", "2")
        keys = wm.list_keys()
        assert set(keys) == {"a", "b"}

    def test_budget_exceeded(self, wm):
        # Fill up (500 token budget)
        wm.set("big", "word " * 400)  # ~401 tokens
        with pytest.raises(ValueError, match="Working memory budget exceeded"):
            wm.set("big2", "word " * 400)  # would push over 500

    def test_update_reuses_budget(self, wm):
        wm.set("key", "hello " * 30)
        before = wm.total_tokens
        wm.set("key", "world " * 30)
        after = wm.total_tokens
        assert abs(after - before) < 5

    def test_clear(self, wm):
        wm.set("a", "1")
        wm.set("b", "2")
        wm.clear()
        assert wm.list_keys() == []
        assert wm.total_tokens == 0

    def test_get_context_block(self, wm):
        wm.set("plan", "Step 1: do thing")
        wm.set("notes", "Some notes")
        block = wm.get_context_block()
        assert "plan" in block
        assert "Step 1" in block

    def test_context_block_empty(self, wm):
        assert wm.get_context_block() == ""
