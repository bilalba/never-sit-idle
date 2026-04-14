"""Tool registry and implementations for the agent.

Tools available to the LLM:
    File system:      read_file, glob_files, grep_files
    Knowledge base:   kb_write, kb_read, kb_delete, kb_list, kb_search, kb_tree
    Memory:           memory_store, memory_retrieve, memory_delete, memory_list, memory_search
    Working memory:   wm_set, wm_get, wm_delete, wm_list
    Data sources:     reddit_search, reddit_top, reddit_comments,
                      wikipedia_search, wikipedia_article,
                      hackernews_search, hackernews_top,
                      stackexchange_search, stackexchange_answers,
                      github_search_repos, github_readme,
                      web_fetch
    Utility:          rate_limit_stats, think
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from agent.knowledge_base import KnowledgeBase
from agent.memory import LongTermMemory, WorkingMemory
from agent.sources import (
    GitHubClient,
    HackerNewsClient,
    RedditClient,
    StackExchangeClient,
    WebFetcher,
    WikipediaClient,
    rate_limiter,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Tool schema type (OpenAI-compatible function calling)
# ═══════════════════════════════════════════════════════════════════════════

ToolDef = dict[str, Any]  # {"type": "function", "function": {...}}


def _func_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> ToolDef:
    """Build an OpenAI-compatible tool definition."""
    props = parameters
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required or [],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════════════════════════


class ToolRegistry:
    """Maps tool names → (schema, handler) pairs."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDef, Callable[..., Any]]] = {}

    def register(self, schema: ToolDef, handler: Callable[..., Any]) -> None:
        name = schema["function"]["name"]
        self._tools[name] = (schema, handler)

    def get_schemas(self) -> list[ToolDef]:
        return [schema for schema, _ in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name. Returns JSON string result.

        Strips any unexpected keyword arguments that the LLM may inject
        (e.g. Gemma adding 'id' fields) to prevent TypeErrors.
        """
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        schema, handler = self._tools[name]
        try:
            # Filter arguments to only those declared in the schema
            declared_params = set(
                schema.get("function", {})
                .get("parameters", {})
                .get("properties", {})
                .keys()
            )
            if declared_params:
                filtered = {k: v for k, v in arguments.items() if k in declared_params}
            else:
                filtered = arguments
            result = handler(**filtered)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Build the default registry
# ═══════════════════════════════════════════════════════════════════════════


def build_registry(
    kb: KnowledgeBase,
    ltm: LongTermMemory,
    wm: WorkingMemory,
    codebase_root: str | Path | None = None,
) -> ToolRegistry:
    """Construct a fully-wired ToolRegistry."""

    registry = ToolRegistry()
    codebase = Path(codebase_root) if codebase_root else Path.cwd()

    # Instantiate source clients
    reddit = RedditClient()
    wikipedia = WikipediaClient()
    hackernews = HackerNewsClient()
    stackexchange = StackExchangeClient()
    github = GitHubClient()
    web = WebFetcher()

    # ── File system ───────────────────────────────────────────────────

    registry.register(
        _func_tool("read_file", "Read a file from the codebase", {
            "path": {"type": "string", "description": "Relative path from codebase root"},
            "max_lines": {"type": "integer", "description": "Max lines to read (default 200)"},
        }, ["path"]),
        lambda path, max_lines=200: _read_file(codebase, path, max_lines),
    )

    registry.register(
        _func_tool("glob_files", "Find files matching a glob pattern", {
            "pattern": {"type": "string", "description": "Glob pattern like '**/*.py'"},
        }, ["pattern"]),
        lambda pattern: _glob_files(codebase, pattern),
    )

    registry.register(
        _func_tool("grep_files", "Search file contents with regex", {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "glob": {"type": "string", "description": "Optional glob to filter files (e.g. '*.py')"},
        }, ["pattern"]),
        lambda pattern, glob="*": _grep_files(codebase, pattern, glob),
    )

    # ── Knowledge base ────────────────────────────────────────────────

    registry.register(
        _func_tool("kb_write", "Write or update a knowledge base entry", {
            "path": {"type": "string", "description": "Path like 'architecture/data_flow' (no .md needed)"},
            "content": {"type": "string", "description": "Markdown content"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for searchability"},
        }, ["path", "content"]),
        lambda path, content, tags=None: kb.write_entry(path, content, tags=tags),
    )

    registry.register(
        _func_tool("kb_read", "Read a knowledge base entry", {
            "path": {"type": "string", "description": "Path to the entry"},
        }, ["path"]),
        lambda path: kb.read_entry(path) or {"error": "Entry not found"},
    )

    registry.register(
        _func_tool("kb_delete", "Delete a knowledge base entry", {
            "path": {"type": "string", "description": "Path to delete"},
        }, ["path"]),
        lambda path: {"deleted": kb.delete_entry(path)},
    )

    registry.register(
        _func_tool("kb_list", "List knowledge base entries", {
            "prefix": {"type": "string", "description": "Optional path prefix filter"},
        }),
        lambda prefix=None: kb.list_entries(prefix),
    )

    registry.register(
        _func_tool("kb_search", "Search knowledge base by keyword and/or tags", {
            "query": {"type": "string", "description": "Search query"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
        }),
        lambda query="", tags=None: kb.search(query, tags=tags),
    )

    registry.register(
        _func_tool("kb_tree", "Show knowledge base tree structure", {}),
        lambda: kb.tree(),
    )

    # ── Long-term memory ──────────────────────────────────────────────

    registry.register(
        _func_tool("memory_store", "Store a long-term memory entry (persistent)", {
            "category": {"type": "string", "description": "Category (e.g. 'facts', 'decisions', 'context')"},
            "name": {"type": "string", "description": "Entry name"},
            "content": {"type": "string", "description": "Content to remember"},
        }, ["category", "name", "content"]),
        lambda category, name, content: ltm.store(category, name, content),
    )

    registry.register(
        _func_tool("memory_retrieve", "Retrieve a long-term memory entry", {
            "category": {"type": "string", "description": "Category"},
            "name": {"type": "string", "description": "Entry name"},
        }, ["category", "name"]),
        lambda category, name: ltm.retrieve(category, name) or {"error": "Not found"},
    )

    registry.register(
        _func_tool("memory_delete", "Delete a long-term memory entry", {
            "category": {"type": "string", "description": "Category"},
            "name": {"type": "string", "description": "Entry name"},
        }, ["category", "name"]),
        lambda category, name: {"deleted": ltm.delete(category, name)},
    )

    registry.register(
        _func_tool("memory_list", "List long-term memory entries", {
            "category": {"type": "string", "description": "Optional category filter"},
        }),
        lambda category=None: ltm.list_entries(category),
    )

    registry.register(
        _func_tool("memory_search", "Search long-term memory", {
            "query": {"type": "string", "description": "Search query"},
        }, ["query"]),
        lambda query: ltm.search(query),
    )

    # ── Working memory ────────────────────────────────────────────────

    registry.register(
        _func_tool("wm_set", "Set a working memory entry (session-only scratchpad)", {
            "key": {"type": "string", "description": "Key"},
            "value": {"type": "string", "description": "Value to store"},
        }, ["key", "value"]),
        lambda key, value: (wm.set(key, value), {"stored": key})[1],
    )

    registry.register(
        _func_tool("wm_get", "Get a working memory entry", {
            "key": {"type": "string", "description": "Key"},
        }, ["key"]),
        lambda key: wm.get(key) or {"error": "Not found"},
    )

    registry.register(
        _func_tool("wm_delete", "Delete a working memory entry", {
            "key": {"type": "string", "description": "Key"},
        }, ["key"]),
        lambda key: {"deleted": wm.delete(key)},
    )

    registry.register(
        _func_tool("wm_list", "List all working memory keys", {}),
        lambda: wm.list_keys(),
    )

    # ── Reddit ────────────────────────────────────────────────────────

    registry.register(
        _func_tool("reddit_search", "Search Reddit for posts", {
            "query": {"type": "string", "description": "Search query"},
            "subreddit": {"type": "string", "description": "Subreddit to search (omit for all)"},
            "limit": {"type": "integer", "description": "Max results (default 10, max 25)"},
        }, ["query"]),
        lambda query, subreddit=None, limit=10: (
            reddit.search_subreddit(subreddit, query, limit) if subreddit
            else reddit.search_all(query, limit)
        ),
    )

    registry.register(
        _func_tool("reddit_top", "Get top posts from a subreddit", {
            "subreddit": {"type": "string", "description": "Subreddit name"},
            "time_filter": {"type": "string", "description": "Time filter: hour, day, week, month, year, all"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        }, ["subreddit"]),
        lambda subreddit, time_filter="month", limit=10: reddit.get_subreddit_top(subreddit, time_filter, limit),
    )

    registry.register(
        _func_tool("reddit_comments", "Get post and its comments", {
            "subreddit": {"type": "string", "description": "Subreddit name"},
            "post_id": {"type": "string", "description": "Post ID"},
            "limit": {"type": "integer", "description": "Max comments (default 20)"},
        }, ["subreddit", "post_id"]),
        lambda subreddit, post_id, limit=20: reddit.get_post_comments(subreddit, post_id, limit),
    )

    # ── Wikipedia ─────────────────────────────────────────────────────

    registry.register(
        _func_tool("wikipedia_search", "Search Wikipedia articles", {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        }, ["query"]),
        lambda query, limit=5: wikipedia.search(query, limit),
    )

    registry.register(
        _func_tool("wikipedia_article", "Get Wikipedia article content", {
            "title": {"type": "string", "description": "Article title"},
            "summary_only": {"type": "boolean", "description": "Get summary only (default true)"},
        }, ["title"]),
        lambda title, summary_only=True: (
            wikipedia.get_summary(title) if summary_only
            else {"title": title, "content": wikipedia.get_content(title)[:5000]}
        ),
    )

    # ── Hacker News ───────────────────────────────────────────────────

    registry.register(
        _func_tool("hackernews_search", "Search Hacker News", {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        }, ["query"]),
        lambda query, limit=10: hackernews.search(query, limit),
    )

    registry.register(
        _func_tool("hackernews_top", "Get top HN stories", {
            "limit": {"type": "integer", "description": "Max stories (default 10)"},
        }),
        lambda limit=10: hackernews.get_top_stories(limit),
    )

    # ── StackExchange ─────────────────────────────────────────────────

    registry.register(
        _func_tool("stackexchange_search", "Search StackOverflow/StackExchange", {
            "query": {"type": "string", "description": "Search query"},
            "site": {"type": "string", "description": "Site (default 'stackoverflow')"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        }, ["query"]),
        lambda query, site="stackoverflow", limit=10: stackexchange.search(query, site, limit),
    )

    registry.register(
        _func_tool("stackexchange_answers", "Get answers for a StackExchange question", {
            "question_id": {"type": "integer", "description": "Question ID"},
            "site": {"type": "string", "description": "Site (default 'stackoverflow')"},
            "limit": {"type": "integer", "description": "Max answers (default 5)"},
        }, ["question_id"]),
        lambda question_id, site="stackoverflow", limit=5: stackexchange.get_answers(question_id, site, limit),
    )

    # ── GitHub ────────────────────────────────────────────────────────

    registry.register(
        _func_tool("github_search_repos", "Search GitHub repositories", {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        }, ["query"]),
        lambda query, limit=5: github.search_repos(query, limit),
    )

    registry.register(
        _func_tool("github_readme", "Get a GitHub repo's README", {
            "owner": {"type": "string", "description": "Repo owner"},
            "repo": {"type": "string", "description": "Repo name"},
        }, ["owner", "repo"]),
        lambda owner, repo: github.get_readme(owner, repo),
    )

    # ── Web ───────────────────────────────────────────────────────────

    registry.register(
        _func_tool("web_fetch", "Fetch and extract text from a web page", {
            "url": {"type": "string", "description": "URL to fetch"},
        }, ["url"]),
        lambda url: web.fetch(url),
    )

    # ── Utility ───────────────────────────────────────────────────────

    registry.register(
        _func_tool("rate_limit_stats", "Show rate limit stats for all sources", {}),
        lambda: rate_limiter.all_stats(),
    )

    registry.register(
        _func_tool("think", "Think step-by-step (scratchpad, not stored)", {
            "thought": {"type": "string", "description": "Your reasoning"},
        }, ["thought"]),
        lambda thought: {"acknowledged": True},
    )

    return registry


# ═══════════════════════════════════════════════════════════════════════════
# File-system tool implementations
# ═══════════════════════════════════════════════════════════════════════════


def _read_file(root: Path, path: str, max_lines: int) -> dict[str, Any]:
    target = (root / path).resolve()
    # Safety: don't escape codebase root
    if not str(target).startswith(str(root.resolve())):
        return {"error": "Path escapes codebase root"}
    if not target.exists():
        return {"error": f"File not found: {path}"}
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        lines = target.read_text(errors="replace").splitlines()
        truncated = len(lines) > max_lines
        content = "\n".join(lines[:max_lines])
        return {
            "path": path,
            "content": content,
            "lines": len(lines),
            "truncated": truncated,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _glob_files(root: Path, pattern: str) -> list[str]:
    matches = sorted(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
    return matches[:200]  # cap results


def _grep_files(root: Path, pattern: str, file_glob: str) -> list[dict[str, Any]]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return [{"error": f"Invalid regex: {exc}"}]

    results: list[dict[str, Any]] = []
    for fpath in root.rglob(file_glob):
        if not fpath.is_file():
            continue
        # Skip binary / large files
        if fpath.stat().st_size > 1_000_000:
            continue
        try:
            text = fpath.read_text(errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append({
                    "file": str(fpath.relative_to(root)),
                    "line": i,
                    "text": line.strip()[:300],
                })
                if len(results) >= 100:
                    return results
    return results
