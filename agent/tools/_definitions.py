"""Tool definitions: registers all tools into a ToolRegistry."""

from __future__ import annotations

from pathlib import Path

from agent import queue as Q
from agent.config import ALPHA_VANTAGE_API_KEY
from agent.knowledge_base import KnowledgeBase
from agent.memory import LongTermMemory, WorkingMemory
from agent.sources import (
    AlphaVantageClient,
    FeedsearchClient,
    GDELTClient,
    GitHubClient,
    GoogleNewsClient,
    HackerNewsClient,
    RedditClient,
    StackExchangeClient,
    WebFetcher,
    WikipediaClient,
    YFinanceClient,
    rate_limiter,
)

from ._registry import ToolRegistry, _func_tool
from ._filesystem import _read_file, _glob_files, _grep_files


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
    google_news = GoogleNewsClient()
    gdelt = GDELTClient()
    feedsearch = FeedsearchClient()
    yfinance = YFinanceClient()
    alphavantage = AlphaVantageClient(api_key=ALPHA_VANTAGE_API_KEY)

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

    # ── Google News ───────────────────────────────────────────────────

    registry.register(
        _func_tool("google_news_search", "Search Google News for articles (returns titles, links, sources)", {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        }, ["query"]),
        lambda query, limit=10: google_news.search(query, limit),
    )

    registry.register(
        _func_tool("google_news_topic", "Get Google News for a topic (WORLD, BUSINESS, TECHNOLOGY, etc.)", {
            "topic": {"type": "string", "description": "Topic: WORLD, NATION, BUSINESS, TECHNOLOGY, SCIENCE, HEALTH, SPORTS, ENTERTAINMENT"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        }, ["topic"]),
        lambda topic, limit=10: google_news.topic(topic, limit),
    )

    # ── GDELT ────────────────────────────────────────────────────────

    registry.register(
        _func_tool("gdelt_search", "Search GDELT global news (includes tone/sentiment scores)", {
            "query": {"type": "string", "description": "Search keywords"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
            "timespan": {"type": "string", "description": "Time span: '7d', '30d', '1y' (default '7d')"},
        }, ["query"]),
        lambda query, limit=10, timespan="7d": gdelt.search(query, limit=limit, timespan=timespan),
    )

    registry.register(
        _func_tool("gdelt_tone", "Get GDELT tone/sentiment timeline for a topic", {
            "query": {"type": "string", "description": "Search keywords"},
            "timespan": {"type": "string", "description": "Time span: '7d', '30d', '1y' (default '30d')"},
        }, ["query"]),
        lambda query, timespan="30d": gdelt.tone_chart(query, timespan),
    )

    # ── Feed discovery ───────────────────────────────────────────────

    registry.register(
        _func_tool("feed_discover", "Discover RSS/Atom feeds for a website", {
            "url": {"type": "string", "description": "Website URL to discover feeds for"},
        }, ["url"]),
        lambda url: feedsearch.discover(url),
    )

    registry.register(
        _func_tool("feed_fetch", "Fetch and parse an RSS/Atom feed", {
            "feed_url": {"type": "string", "description": "RSS/Atom feed URL"},
            "limit": {"type": "integer", "description": "Max items (default 10)"},
        }, ["feed_url"]),
        lambda feed_url, limit=10: feedsearch.fetch_feed(feed_url, limit),
    )

    # ── Yahoo Finance ────────────────────────────────────────────────

    registry.register(
        _func_tool("yfinance_search", "Search Yahoo Finance for tickers/companies", {
            "query": {"type": "string", "description": "Search query (company name or ticker)"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        }, ["query"]),
        lambda query, limit=5: yfinance.search(query, limit),
    )

    registry.register(
        _func_tool("yfinance_quote", "Get current stock quote from Yahoo Finance", {
            "symbol": {"type": "string", "description": "Ticker symbol (e.g. AAPL, MSFT)"},
        }, ["symbol"]),
        lambda symbol: yfinance.quote(symbol),
    )

    registry.register(
        _func_tool("yfinance_news", "Get news for a stock ticker via Yahoo Finance RSS", {
            "symbol": {"type": "string", "description": "Ticker symbol"},
            "limit": {"type": "integer", "description": "Max articles (default 10)"},
        }, ["symbol"]),
        lambda symbol, limit=10: yfinance.news(symbol, limit),
    )

    # ── Alpha Vantage ────────────────────────────────────────────────

    registry.register(
        _func_tool("alphavantage_search", "Search Alpha Vantage for tickers/companies", {
            "query": {"type": "string", "description": "Search keywords"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        }, ["query"]),
        lambda query, limit=5: alphavantage.search(query, limit),
    )

    registry.register(
        _func_tool("alphavantage_quote", "Get stock quote from Alpha Vantage (with change data)", {
            "symbol": {"type": "string", "description": "Ticker symbol"},
        }, ["symbol"]),
        lambda symbol: alphavantage.quote(symbol),
    )

    registry.register(
        _func_tool("alphavantage_news", "Get news with sentiment scores from Alpha Vantage", {
            "tickers": {"type": "string", "description": "Comma-separated tickers (e.g. 'AAPL,MSFT'). Optional."},
            "topics": {"type": "string", "description": "Comma-separated topics (e.g. 'technology,earnings'). Optional."},
            "limit": {"type": "integer", "description": "Max articles (default 10)"},
        }),
        lambda tickers="", topics="", limit=10: alphavantage.news_sentiment(tickers, topics, limit),
    )

    # ── Job queue ─────────────────────────────────────────────────────

    registry.register(
        _func_tool("queue_research", "Queue a topic for deep background research (runs asynchronously)", {
            "topic": {"type": "string", "description": "The topic to research"},
        }, ["topic"]),
        lambda topic: {
            "queued": True,
            "job_id": Q.add(Q.RESEARCH, topic)["id"],
            "topic": topic,
        },
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
