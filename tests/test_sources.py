"""Tests for data sources and rate limiter."""

import time
import pytest
from unittest.mock import patch, MagicMock

from agent.sources import (
    RateLimitState,
    RateLimiter,
    rate_limiter,
    _rate_limited_get,
    _parse_rss_items,
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
    _strip_html,
)


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimitState:
    def test_record_request(self):
        state = RateLimitState(name="test", max_requests_per_minute=10)
        state.record_request()
        assert state.total_requests == 1

    def test_stats(self):
        state = RateLimitState(name="test", max_requests_per_minute=10)
        state.record_request()
        stats = state.stats()
        assert stats["name"] == "test"
        assert stats["total_requests"] == 1
        assert stats["requests_in_window"] == 1

    def test_update_from_headers(self):
        state = RateLimitState(name="test", max_requests_per_minute=10)
        state.update_from_headers({
            "x-ratelimit-remaining": "5",
            "x-ratelimit-reset": "30",
        })
        assert state.remaining == 5
        assert state.reset_at is not None

    def test_timestamps_pruned(self):
        state = RateLimitState(name="test", max_requests_per_minute=100, window_seconds=1)
        for _ in range(5):
            state.record_request()
        # All should be in window
        assert len(state.timestamps) == 5


class TestRateLimiter:
    def test_register_and_get(self):
        rl = RateLimiter()
        rl.register("test_source", max_requests_per_minute=10)
        state = rl.get("test_source")
        assert state.name == "test_source"

    def test_get_unregistered(self):
        rl = RateLimiter()
        with pytest.raises(KeyError):
            rl.get("nonexistent")

    def test_record(self):
        rl = RateLimiter()
        rl.register("test_source", max_requests_per_minute=10)
        rl.record("test_source")
        stats = rl.get("test_source").stats()
        assert stats["total_requests"] == 1

    def test_all_stats(self):
        rl = RateLimiter()
        rl.register("a", max_requests_per_minute=10)
        rl.register("b", max_requests_per_minute=20)
        stats = rl.all_stats()
        assert len(stats) == 2

    def test_global_rate_limiter_has_all_sources(self):
        stats = rate_limiter.all_stats()
        names = {s["name"] for s in stats}
        assert "reddit" in names
        assert "wikipedia" in names
        assert "hackernews" in names
        assert "stackexchange" in names
        assert "github" in names
        assert "web" in names
        assert "googlenews" in names
        assert "gdelt" in names
        assert "feedsearch" in names
        assert "yfinance" in names
        assert "alphavantage" in names


# ═══════════════════════════════════════════════════════════════════════════
# Rate-limited GET tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimitedGet:
    @patch("agent.sources._base.requests.get")
    def test_success(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        mock_get.return_value = resp

        result = _rate_limited_get("wikipedia", "https://example.com")
        assert result == resp

    @patch("agent.sources._base.requests.get")
    def test_retry_on_429(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "1"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.headers = {}

        mock_get.side_effect = [resp_429, resp_ok]
        result = _rate_limited_get("wikipedia", "https://example.com", max_retries=3)
        assert result.status_code == 200

    @patch("agent.sources._base.requests.get")
    def test_retry_on_500(self, mock_get):
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.headers = {}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.headers = {}

        mock_get.side_effect = [resp_500, resp_ok]
        result = _rate_limited_get("wikipedia", "https://example.com", max_retries=3)
        assert result.status_code == 200

    @patch("agent.sources._base.requests.get")
    def test_raises_on_client_error(self, mock_get):
        import requests
        resp = MagicMock()
        resp.status_code = 404
        resp.headers = {}
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = resp

        with pytest.raises(requests.exceptions.HTTPError):
            _rate_limited_get("wikipedia", "https://example.com", max_retries=1)


# ═══════════════════════════════════════════════════════════════════════════
# Source client tests (mocked HTTP)
# ═══════════════════════════════════════════════════════════════════════════


class TestRedditClient:
    @patch("agent.sources._reddit._rate_limited_get")
    def test_search_all(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "data": {"children": [
                {"kind": "t3", "data": {
                    "id": "abc123", "title": "Test Post",
                    "selftext": "Body text", "subreddit": "python",
                    "author": "user1", "score": 42, "num_comments": 10,
                    "url": "https://reddit.com/r/python/abc123",
                    "permalink": "/r/python/comments/abc123",
                    "created_utc": 1700000000,
                }}
            ]}
        })
        client = RedditClient()
        results = client.search_all("python async")
        assert len(results) == 1
        assert results[0]["title"] == "Test Post"
        assert results[0]["score"] == 42

    @patch("agent.sources._reddit._rate_limited_get")
    def test_search_subreddit(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "data": {"children": [
                {"kind": "t3", "data": {"id": "x", "title": "Found", "selftext": "",
                    "subreddit": "test", "author": "a", "score": 1,
                    "num_comments": 0, "url": "", "permalink": "", "created_utc": 0}}
            ]}
        })
        client = RedditClient()
        results = client.search_subreddit("test", "query")
        assert len(results) == 1

    @patch("agent.sources._reddit._rate_limited_get")
    def test_get_post_comments(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: [
            {"data": {"children": [
                {"kind": "t3", "data": {"id": "post1", "title": "Post",
                    "selftext": "", "subreddit": "test", "author": "a",
                    "score": 1, "num_comments": 1, "url": "", "permalink": "",
                    "created_utc": 0}}
            ]}},
            {"data": {"children": [
                {"kind": "t1", "data": {"id": "c1", "author": "b",
                    "body": "Great post!", "score": 5, "created_utc": 0}}
            ]}}
        ])
        client = RedditClient()
        result = client.get_post_comments("test", "post1")
        assert result["post"]["title"] == "Post"
        assert len(result["comments"]) == 1
        assert result["comments"][0]["body"] == "Great post!"


class TestWikipediaClient:
    @patch("agent.sources._wikipedia._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "query": {"search": [
                {"title": "Python (programming language)",
                 "snippet": "A <b>programming</b> language",
                 "pageid": 23862}
            ]}
        })
        client = WikipediaClient()
        results = client.search("Python programming")
        assert len(results) == 1
        assert results[0]["title"] == "Python (programming language)"

    @patch("agent.sources._wikipedia._rate_limited_get")
    def test_get_summary(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "title": "Python",
            "extract": "Python is a programming language.",
            "description": "General-purpose programming language",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
        })
        client = WikipediaClient()
        summary = client.get_summary("Python")
        assert summary["title"] == "Python"
        assert "programming" in summary["extract"]


class TestHackerNewsClient:
    @patch("agent.sources._hackernews._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "hits": [
                {"title": "Show HN: Cool thing", "url": "https://example.com",
                 "author": "user", "points": 100, "num_comments": 50,
                 "objectID": "123", "created_at": "2024-01-01"}
            ]
        })
        client = HackerNewsClient()
        results = client.search("cool thing")
        assert len(results) == 1
        assert results[0]["title"] == "Show HN: Cool thing"


class TestStackExchangeClient:
    @patch("agent.sources._stackexchange._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "items": [
                {"question_id": 123, "title": "How to Python?",
                 "body": "<p>Help me</p>", "tags": ["python"],
                 "score": 10, "answer_count": 3, "is_answered": True,
                 "link": "https://stackoverflow.com/q/123"}
            ]
        })
        client = StackExchangeClient()
        results = client.search("how to python")
        assert len(results) == 1
        assert results[0]["title"] == "How to Python?"
        assert results[0]["is_answered"] is True


class TestGitHubClient:
    @patch("agent.sources._github._rate_limited_get")
    def test_search_repos(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "items": [
                {"full_name": "python/cpython", "description": "The Python interpreter",
                 "stargazers_count": 50000, "language": "Python",
                 "html_url": "https://github.com/python/cpython", "topics": ["python"]}
            ]
        })
        client = GitHubClient()
        results = client.search_repos("python")
        assert len(results) == 1
        assert results[0]["full_name"] == "python/cpython"


class TestWebFetcher:
    @patch("agent.sources._web._rate_limited_get")
    def test_fetch_html(self, mock_get):
        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body><p>Hello world</p></body></html>"
        mock_get.return_value = resp

        client = WebFetcher()
        result = client.fetch("https://example.com")
        assert "Hello world" in result["content"]
        assert result["type"] == "html"

    @patch("agent.sources._web._rate_limited_get")
    def test_fetch_json(self, mock_get):
        resp = MagicMock()
        resp.headers = {"Content-Type": "application/json"}
        resp.json.return_value = {"key": "value"}
        mock_get.return_value = resp

        client = WebFetcher()
        result = client.fetch("https://api.example.com/data")
        assert result["type"] == "json"
        assert "key" in result["content"]


# ═══════════════════════════════════════════════════════════════════════════
# HTML stripping
# ═══════════════════════════════════════════════════════════════════════════


class TestStripHtml:
    def test_basic(self):
        assert _strip_html("<p>Hello</p>") == "Hello"

    def test_nested(self):
        result = _strip_html("<div><p>Hello <b>world</b></p></div>")
        assert "Hello" in result
        assert "world" in result

    def test_script_tags_removed(self):
        result = _strip_html("<p>Hi</p><script>alert('xss')</script><p>Bye</p>")
        assert "alert" not in result
        assert "Hi" in result
        assert "Bye" in result

    def test_empty(self):
        assert _strip_html("") == ""
        assert _strip_html(None) == ""


# ═══════════════════════════════════════════════════════════════════════════
# RSS parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestParseRssItems:
    def test_basic_rss(self):
        xml = """<?xml version="1.0"?>
        <rss><channel>
            <item>
                <title>Breaking News</title>
                <link>https://example.com/1</link>
                <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
                <source>Example</source>
            </item>
            <item>
                <title>Other News</title>
                <link>https://example.com/2</link>
            </item>
        </channel></rss>"""
        items = _parse_rss_items(xml, 10)
        assert len(items) == 2
        assert items[0]["title"] == "Breaking News"
        assert items[0]["link"] == "https://example.com/1"
        assert items[1]["title"] == "Other News"

    def test_limit(self):
        xml = """<?xml version="1.0"?>
        <rss><channel>
            <item><title>A</title></item>
            <item><title>B</title></item>
            <item><title>C</title></item>
        </channel></rss>"""
        items = _parse_rss_items(xml, 2)
        assert len(items) == 2

    def test_bad_xml(self):
        assert _parse_rss_items("not xml at all", 10) == []

    def test_html_description_stripped(self):
        xml = """<?xml version="1.0"?>
        <rss><channel>
            <item>
                <title>Test</title>
                <description>&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description>
            </item>
        </channel></rss>"""
        items = _parse_rss_items(xml, 10)
        assert len(items) == 1
        assert "<" not in items[0].get("description", "")


# ═══════════════════════════════════════════════════════════════════════════
# Google News
# ═══════════════════════════════════════════════════════════════════════════


class TestGoogleNewsClient:
    @patch("agent.sources._news._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(text="""<?xml version="1.0"?>
        <rss><channel>
            <item>
                <title>AI takes over - CNN</title>
                <link>https://news.google.com/123</link>
                <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                <source url="https://cnn.com">CNN</source>
            </item>
        </channel></rss>""")
        client = GoogleNewsClient()
        results = client.search("AI")
        assert len(results) == 1
        assert results[0]["title"] == "AI takes over - CNN"
        assert results[0]["source"] == "CNN"

    @patch("agent.sources._news._rate_limited_get")
    def test_topic(self, mock_get):
        mock_get.return_value = MagicMock(text="""<?xml version="1.0"?>
        <rss><channel>
            <item><title>Tech News</title><link>https://example.com</link></item>
        </channel></rss>""")
        client = GoogleNewsClient()
        results = client.topic("TECHNOLOGY")
        assert len(results) == 1

    @patch("agent.sources._news._rate_limited_get")
    def test_empty_response(self, mock_get):
        mock_get.return_value = MagicMock(text="<?xml version='1.0'?><rss><channel></channel></rss>")
        client = GoogleNewsClient()
        results = client.search("nonexistent topic xyzzy")
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# GDELT
# ═══════════════════════════════════════════════════════════════════════════


class TestGDELTClient:
    @patch("agent.sources._news._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "articles": [
                {"title": "AI article", "url": "https://example.com/ai",
                 "domain": "example.com", "language": "English",
                 "seendate": "20240101T120000Z", "tone": -2.5,
                 "socialimage": ""},
            ]
        })
        client = GDELTClient()
        results = client.search("artificial intelligence")
        assert len(results) == 1
        assert results[0]["title"] == "AI article"
        assert results[0]["tone"] == -2.5

    @patch("agent.sources._news._rate_limited_get")
    def test_tone_chart(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "timeline": [{"date": "2024-01-01", "tone": 1.5}]
        })
        client = GDELTClient()
        result = client.tone_chart("bitcoin")
        assert "timeline" in result

    @patch("agent.sources._news._rate_limited_get")
    def test_search_bad_json(self, mock_get):
        mock_get.return_value = MagicMock(json=MagicMock(side_effect=ValueError))
        client = GDELTClient()
        results = client.search("test")
        assert results == []

    @patch("agent.sources._news._rate_limited_get")
    def test_tone_chart_bad_json(self, mock_get):
        mock_get.return_value = MagicMock(json=MagicMock(side_effect=ValueError))
        client = GDELTClient()
        result = client.tone_chart("test")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# Feedsearch
# ═══════════════════════════════════════════════════════════════════════════


class TestFeedsearchClient:
    def test_extract_link_feeds(self):
        html = '''<html><head>
            <link rel="alternate" type="application/rss+xml" title="Blog" href="/feed.xml">
            <link rel="alternate" type="application/atom+xml" title="Atom" href="https://example.com/atom.xml">
        </head></html>'''
        client = FeedsearchClient()
        feeds = client._extract_link_feeds(html, "https://example.com")
        assert len(feeds) == 2
        assert feeds[0]["url"] == "https://example.com/feed.xml"
        assert feeds[0]["title"] == "Blog"
        assert feeds[1]["url"] == "https://example.com/atom.xml"

    def test_base_url(self):
        assert FeedsearchClient._base_url("https://example.com/blog/page") == "https://example.com"
        assert FeedsearchClient._base_url("http://sub.example.com/a/b") == "http://sub.example.com"

    @patch("agent.sources._news._rate_limited_get")
    def test_discover_from_html(self, mock_get):
        html_resp = MagicMock()
        html_resp.text = '<html><head><link rel="alternate" type="application/rss+xml" href="/rss"></head></html>'
        html_resp.headers = {"Content-Type": "text/html"}

        # The common-path probes will fail
        mock_get.side_effect = [html_resp] + [Exception("not found")] * 10
        client = FeedsearchClient()
        feeds = client.discover("https://example.com")
        assert len(feeds) >= 1
        assert feeds[0]["url"] == "https://example.com/rss"

    @patch("agent.sources._news._rate_limited_get")
    def test_fetch_feed(self, mock_get):
        mock_get.return_value = MagicMock(text="""<?xml version="1.0"?>
        <rss><channel>
            <item><title>Post 1</title><link>https://blog.com/1</link></item>
        </channel></rss>""")
        client = FeedsearchClient()
        items = client.fetch_feed("https://blog.com/feed.xml")
        assert len(items) == 1
        assert items[0]["title"] == "Post 1"


# ═══════════════════════════════════════════════════════════════════════════
# Yahoo Finance
# ═══════════════════════════════════════════════════════════════════════════


class TestYFinanceClient:
    @patch("agent.sources._finance._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "quotes": [
                {"symbol": "AAPL", "shortname": "Apple Inc.", "quoteType": "EQUITY",
                 "exchange": "NMS"},
            ]
        })
        client = YFinanceClient()
        results = client.search("apple")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"
        assert results[0]["name"] == "Apple Inc."

    @patch("agent.sources._finance._rate_limited_get")
    def test_quote(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "chart": {"result": [{"meta": {
                "symbol": "AAPL", "currency": "USD", "exchangeName": "NMS",
                "regularMarketPrice": 195.50, "previousClose": 194.00,
                "regularMarketTime": 1700000000,
                "regularMarketDayHigh": 196.00, "regularMarketDayLow": 193.50,
            }}]}
        })
        client = YFinanceClient()
        result = client.quote("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["price"] == 195.50

    @patch("agent.sources._finance._rate_limited_get")
    def test_quote_not_found(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {"chart": {"result": []}})
        client = YFinanceClient()
        result = client.quote("INVALID")
        assert "error" in result

    @patch("agent.sources._finance._rate_limited_get")
    def test_news(self, mock_get):
        mock_get.return_value = MagicMock(text="""<?xml version="1.0"?>
        <rss><channel>
            <item>
                <title>Apple earnings beat</title>
                <link>https://finance.yahoo.com/news/1</link>
            </item>
        </channel></rss>""")
        client = YFinanceClient()
        results = client.news("AAPL")
        assert len(results) == 1
        assert "Apple" in results[0]["title"]


# ═══════════════════════════════════════════════════════════════════════════
# Alpha Vantage
# ═══════════════════════════════════════════════════════════════════════════


class TestAlphaVantageClient:
    @patch("agent.sources._finance._rate_limited_get")
    def test_search(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "bestMatches": [
                {"1. symbol": "AAPL", "2. name": "Apple Inc.", "3. type": "Equity",
                 "4. region": "United States", "8. currency": "USD"},
            ]
        })
        client = AlphaVantageClient(api_key="test")
        results = client.search("apple")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"
        assert results[0]["name"] == "Apple Inc."

    @patch("agent.sources._finance._rate_limited_get")
    def test_quote(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "Global Quote": {
                "01. symbol": "AAPL", "02. open": "193.00", "03. high": "196.00",
                "04. low": "192.50", "05. price": "195.50", "06. volume": "50000000",
                "07. latest trading day": "2024-01-15", "08. previous close": "194.00",
                "09. change": "1.50", "10. change percent": "0.77%",
            }
        })
        client = AlphaVantageClient(api_key="test")
        result = client.quote("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["price"] == "195.50"
        assert result["change_percent"] == "0.77%"

    @patch("agent.sources._finance._rate_limited_get")
    def test_quote_not_found(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {"Global Quote": {}})
        client = AlphaVantageClient(api_key="test")
        result = client.quote("INVALID")
        assert "error" in result

    @patch("agent.sources._finance._rate_limited_get")
    def test_news_sentiment(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {
            "feed": [
                {"title": "Apple beats earnings", "url": "https://example.com/1",
                 "source": "Reuters", "time_published": "20240115T120000",
                 "summary": "Apple reported strong quarterly results.",
                 "overall_sentiment_score": 0.35,
                 "overall_sentiment_label": "Somewhat-Bullish",
                 "ticker_sentiment": [
                     {"ticker": "AAPL", "ticker_sentiment_score": "0.40",
                      "ticker_sentiment_label": "Bullish", "relevance_score": "0.95"}
                 ]},
            ]
        })
        client = AlphaVantageClient(api_key="test")
        results = client.news_sentiment(tickers="AAPL")
        assert len(results) == 1
        assert results[0]["overall_sentiment_score"] == 0.35
        assert results[0]["ticker_sentiment"][0]["ticker"] == "AAPL"

    @patch("agent.sources._finance._rate_limited_get")
    def test_news_sentiment_empty(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {"feed": []})
        client = AlphaVantageClient(api_key="test")
        results = client.news_sentiment(topics="technology")
        assert results == []
