"""Data source clients for knowledge gathering.

Sources:
    - Reddit (OAuth2, strict rate-limit tracking)
    - Wikipedia
    - Hacker News (Firebase API)
    - StackOverflow / StackExchange
    - GitHub (public repos, no auth required for reads)
    - Generic web page fetcher

All sources share the RateLimiter to ensure we never exceed limits.
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "never-sit-idle-agent/1.0 (knowledge-builder)"
DEFAULT_TIMEOUT = 30


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RateLimitState:
    """Tracks rate-limit state for a single source."""
    name: str
    max_requests_per_minute: float
    window_seconds: float = 60.0
    timestamps: list[float] = field(default_factory=list)
    remaining: int | None = None          # from response headers
    reset_at: float | None = None         # from response headers
    total_requests: int = 0
    total_throttled: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self) -> None:
        now = time.time()
        with self.lock:
            self.timestamps.append(now)
            self.total_requests += 1
            # Prune old timestamps
            cutoff = now - self.window_seconds
            self.timestamps = [t for t in self.timestamps if t > cutoff]

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update state from response headers (Reddit-style)."""
        with self.lock:
            remaining = headers.get("x-ratelimit-remaining")
            if remaining is not None:
                try:
                    self.remaining = int(float(remaining))
                except (ValueError, TypeError):
                    pass
            reset = headers.get("x-ratelimit-reset")
            if reset is not None:
                try:
                    self.reset_at = time.time() + int(float(reset))
                except (ValueError, TypeError):
                    pass

    def wait_if_needed(self) -> float:
        """Block until it's safe to make a request. Returns seconds waited."""
        # Check header-based remaining first (most accurate for Reddit)
        with self.lock:
            if self.remaining is not None and self.remaining <= 1 and self.reset_at:
                delay = max(0, self.reset_at - time.time() + 0.5)
                if delay > 0:
                    self.total_throttled += 1
                    logger.info(
                        "[%s] Header rate limit: waiting %.1fs (remaining=%s)",
                        self.name, delay, self.remaining,
                    )
                    # Release lock before sleeping
                    self.lock.release()
                    try:
                        time.sleep(delay)
                    finally:
                        self.lock.acquire()
                    return delay

        # Check window-based rate limit
        delay = 0.0
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            recent = [t for t in self.timestamps if t > cutoff]

            if len(recent) >= self.max_requests_per_minute:
                oldest = min(recent)
                delay = self.window_seconds - (now - oldest) + 0.1
                if delay > 0:
                    self.total_throttled += 1
                    logger.info(
                        "[%s] Window rate limit: waiting %.1fs (%d/%d in window)",
                        self.name, delay, len(recent), int(self.max_requests_per_minute),
                    )

        if delay > 0:
            time.sleep(delay)

        return delay

    def stats(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            recent = [t for t in self.timestamps if t > cutoff]
            return {
                "name": self.name,
                "total_requests": self.total_requests,
                "total_throttled": self.total_throttled,
                "requests_in_window": len(recent),
                "max_per_minute": self.max_requests_per_minute,
                "header_remaining": self.remaining,
                "header_reset_at": self.reset_at,
            }


class RateLimiter:
    """Central rate limiter managing multiple source limits."""

    def __init__(self) -> None:
        self._sources: dict[str, RateLimitState] = {}
        self._lock = threading.Lock()

    def register(self, name: str, max_requests_per_minute: float, window_seconds: float = 60.0) -> None:
        with self._lock:
            self._sources[name] = RateLimitState(
                name=name,
                max_requests_per_minute=max_requests_per_minute,
                window_seconds=window_seconds,
            )

    def get(self, name: str) -> RateLimitState:
        with self._lock:
            if name not in self._sources:
                raise KeyError(f"Source '{name}' not registered with rate limiter")
            return self._sources[name]

    def wait(self, name: str) -> float:
        return self.get(name).wait_if_needed()

    def record(self, name: str) -> None:
        self.get(name).record_request()

    def update_headers(self, name: str, headers: dict[str, str]) -> None:
        self.get(name).update_from_headers(headers)

    def all_stats(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.stats() for s in self._sources.values()]


# Global rate limiter instance
rate_limiter = RateLimiter()

# Register all sources with their limits
rate_limiter.register("reddit", max_requests_per_minute=30, window_seconds=60)      # Reddit: 30/min for OAuth
rate_limiter.register("wikipedia", max_requests_per_minute=200, window_seconds=60)   # Wikipedia: generous
rate_limiter.register("hackernews", max_requests_per_minute=60, window_seconds=60)   # HN Firebase: ~1/s
rate_limiter.register("stackexchange", max_requests_per_minute=30, window_seconds=60)  # SE: 30/min w/o key
rate_limiter.register("github", max_requests_per_minute=60, window_seconds=60)       # GitHub: 60/hr unauthenticated
rate_limiter.register("web", max_requests_per_minute=20, window_seconds=60)          # Generic: be polite


# ═══════════════════════════════════════════════════════════════════════════
# Base fetch with rate limiting
# ═══════════════════════════════════════════════════════════════════════════

def _rate_limited_get(
    source: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> requests.Response:
    """GET with rate limiting and retries."""
    all_headers = {"User-Agent": USER_AGENT}
    if headers:
        all_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait(source)
        try:
            resp = requests.get(url, headers=all_headers, params=params, timeout=timeout)
            rate_limiter.record(source)
            rate_limiter.update_headers(source, dict(resp.headers))

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "60")
                delay = min(int(float(retry_after)), 120)
                logger.warning("[%s] 429 Too Many Requests, sleeping %ds", source, delay)
                time.sleep(delay)
                continue

            if resp.status_code >= 500:
                last_exc = Exception(f"HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    raise last_exc or Exception(f"Failed to fetch {url}")


# ═══════════════════════════════════════════════════════════════════════════
# Reddit
# ═══════════════════════════════════════════════════════════════════════════

class RedditClient:
    """Reddit API client using OAuth2 (application-only auth).

    Uses the public JSON endpoint (no OAuth needed for read-only).
    Strictly respects rate limits: 30 requests/min with header tracking.
    """

    BASE = "https://www.reddit.com"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.BASE}{path}.json"
        resp = _rate_limited_get("reddit", url, params=params)
        return resp.json()

    def search_subreddit(self, subreddit: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search a subreddit. Returns list of post dicts."""
        data = self._get(f"/r/{subreddit}/search", params={
            "q": query, "restrict_sr": "on", "limit": min(limit, 25),
            "sort": "relevance", "t": "all",
        })
        return _extract_reddit_posts(data)

    def get_subreddit_top(self, subreddit: str, time_filter: str = "month", limit: int = 10) -> list[dict[str, Any]]:
        """Get top posts from a subreddit."""
        data = self._get(f"/r/{subreddit}/top", params={
            "t": time_filter, "limit": min(limit, 25),
        })
        return _extract_reddit_posts(data)

    def get_post_comments(self, subreddit: str, post_id: str, limit: int = 20) -> dict[str, Any]:
        """Get a post and its comments."""
        data = self._get(f"/r/{subreddit}/comments/{post_id}", params={"limit": limit})
        result: dict[str, Any] = {"post": {}, "comments": []}
        if isinstance(data, list) and len(data) >= 1:
            posts = data[0].get("data", {}).get("children", [])
            if posts:
                result["post"] = _clean_reddit_post(posts[0].get("data", {}))
            if len(data) >= 2:
                comments = data[1].get("data", {}).get("children", [])
                result["comments"] = [
                    _clean_reddit_comment(c.get("data", {}))
                    for c in comments
                    if c.get("kind") == "t1"
                ][:limit]
        return result

    def search_all(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search all of Reddit."""
        data = self._get("/search", params={
            "q": query, "limit": min(limit, 25), "sort": "relevance",
        })
        return _extract_reddit_posts(data)


def _extract_reddit_posts(data: dict) -> list[dict[str, Any]]:
    posts = []
    children = data.get("data", {}).get("children", [])
    for child in children:
        if child.get("kind") == "t3":
            posts.append(_clean_reddit_post(child.get("data", {})))
    return posts


def _clean_reddit_post(d: dict) -> dict[str, Any]:
    return {
        "id": d.get("id", ""),
        "title": d.get("title", ""),
        "selftext": (d.get("selftext", "") or "")[:2000],
        "subreddit": d.get("subreddit", ""),
        "author": d.get("author", ""),
        "score": d.get("score", 0),
        "num_comments": d.get("num_comments", 0),
        "url": d.get("url", ""),
        "permalink": d.get("permalink", ""),
        "created_utc": d.get("created_utc", 0),
    }


def _clean_reddit_comment(d: dict) -> dict[str, Any]:
    return {
        "id": d.get("id", ""),
        "author": d.get("author", ""),
        "body": (d.get("body", "") or "")[:1500],
        "score": d.get("score", 0),
        "created_utc": d.get("created_utc", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Wikipedia
# ═══════════════════════════════════════════════════════════════════════════

class WikipediaClient:
    """Wikipedia API client for fetching article summaries and content."""

    BASE = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_BASE = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Search Wikipedia articles."""
        resp = _rate_limited_get("wikipedia", self.SEARCH_BASE, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": min(limit, 20), "format": "json",
        })
        data = resp.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": _strip_html(item.get("snippet", "")),
                "pageid": str(item.get("pageid", "")),
            })
        return results

    def get_summary(self, title: str) -> dict[str, Any]:
        """Get article summary."""
        encoded = quote_plus(title.replace(" ", "_"))
        resp = _rate_limited_get("wikipedia", f"{self.BASE}/page/summary/{encoded}")
        data = resp.json()
        return {
            "title": data.get("title", ""),
            "extract": data.get("extract", ""),
            "description": data.get("description", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }

    def get_content(self, title: str) -> str:
        """Get full article text content."""
        resp = _rate_limited_get("wikipedia", self.SEARCH_BASE, params={
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": True, "format": "json",
        })
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Hacker News
# ═══════════════════════════════════════════════════════════════════════════

class HackerNewsClient:
    """Hacker News Firebase API client."""

    BASE = "https://hacker-news.firebaseio.com/v0"

    def get_top_stories(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get top story details."""
        resp = _rate_limited_get("hackernews", f"{self.BASE}/topstories.json")
        ids = resp.json()[:min(limit, 30)]
        stories = []
        for story_id in ids:
            try:
                story = self._get_item(story_id)
                if story:
                    stories.append(story)
            except Exception:
                continue
        return stories

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search HN via Algolia API."""
        resp = _rate_limited_get(
            "hackernews",
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "hitsPerPage": min(limit, 20)},
        )
        data = resp.json()
        results = []
        for hit in data.get("hits", []):
            results.append({
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "author": hit.get("author", ""),
                "points": hit.get("points", 0),
                "num_comments": hit.get("num_comments", 0),
                "objectID": hit.get("objectID", ""),
                "created_at": hit.get("created_at", ""),
            })
        return results

    def get_item_with_comments(self, item_id: int, comment_limit: int = 10) -> dict[str, Any]:
        """Get an item and its top comments."""
        item = self._get_item(item_id)
        if not item:
            return {}
        comments = []
        for kid_id in (item.get("kids") or [])[:comment_limit]:
            try:
                comment = self._get_item(kid_id)
                if comment and comment.get("text"):
                    comments.append({
                        "id": comment.get("id"),
                        "author": comment.get("by", ""),
                        "text": _strip_html(comment.get("text", ""))[:1500],
                    })
            except Exception:
                continue
        item["top_comments"] = comments
        return item

    def _get_item(self, item_id: int) -> dict[str, Any] | None:
        resp = _rate_limited_get("hackernews", f"{self.BASE}/item/{item_id}.json")
        data = resp.json()
        if not data:
            return None
        return {
            "id": data.get("id"),
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "text": _strip_html(data.get("text", "") or "")[:2000],
            "by": data.get("by", ""),
            "score": data.get("score", 0),
            "kids": data.get("kids", []),
            "type": data.get("type", ""),
        }


# ═══════════════════════════════════════════════════════════════════════════
# StackExchange
# ═══════════════════════════════════════════════════════════════════════════

class StackExchangeClient:
    """StackOverflow / StackExchange API client."""

    BASE = "https://api.stackexchange.com/2.3"

    def search(self, query: str, site: str = "stackoverflow", limit: int = 10) -> list[dict[str, Any]]:
        """Search questions."""
        resp = _rate_limited_get("stackexchange", f"{self.BASE}/search/advanced", params={
            "q": query, "site": site, "pagesize": min(limit, 25),
            "order": "desc", "sort": "relevance", "filter": "withbody",
        })
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "question_id": item.get("question_id"),
                "title": _strip_html(item.get("title", "")),
                "body": _strip_html(item.get("body", ""))[:2000],
                "tags": item.get("tags", []),
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "is_answered": item.get("is_answered", False),
                "link": item.get("link", ""),
            })
        return results

    def get_answers(self, question_id: int, site: str = "stackoverflow", limit: int = 5) -> list[dict[str, Any]]:
        """Get answers for a question."""
        resp = _rate_limited_get("stackexchange", f"{self.BASE}/questions/{question_id}/answers", params={
            "site": site, "pagesize": min(limit, 10),
            "order": "desc", "sort": "votes", "filter": "withbody",
        })
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "answer_id": item.get("answer_id"),
                "body": _strip_html(item.get("body", ""))[:3000],
                "score": item.get("score", 0),
                "is_accepted": item.get("is_accepted", False),
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════
# GitHub (public, no auth)
# ═══════════════════════════════════════════════════════════════════════════

class GitHubClient:
    """GitHub public API client for repo exploration."""

    BASE = "https://api.github.com"

    def search_repos(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        resp = _rate_limited_get("github", f"{self.BASE}/search/repositories", params={
            "q": query, "per_page": min(limit, 10), "sort": "stars",
        })
        results = []
        for item in resp.json().get("items", []):
            results.append({
                "full_name": item.get("full_name", ""),
                "description": item.get("description", ""),
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language", ""),
                "url": item.get("html_url", ""),
                "topics": item.get("topics", []),
            })
        return results

    def get_readme(self, owner: str, repo: str) -> str:
        """Get repo README content."""
        resp = _rate_limited_get("github", f"{self.BASE}/repos/{owner}/{repo}/readme", headers={
            "Accept": "application/vnd.github.raw",
        })
        return resp.text[:5000]

    def search_code(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        resp = _rate_limited_get("github", f"{self.BASE}/search/code", params={
            "q": query, "per_page": min(limit, 10),
        })
        results = []
        for item in resp.json().get("items", []):
            results.append({
                "name": item.get("name", ""),
                "path": item.get("path", ""),
                "repo": item.get("repository", {}).get("full_name", ""),
                "url": item.get("html_url", ""),
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Generic Web Fetcher
# ═══════════════════════════════════════════════════════════════════════════

class WebFetcher:
    """Fetches and extracts text content from web pages."""

    def fetch(self, url: str) -> dict[str, str]:
        """Fetch a URL and extract text content."""
        resp = _rate_limited_get("web", url)
        content_type = resp.headers.get("Content-Type", "")

        if "json" in content_type:
            return {"url": url, "content": json.dumps(resp.json(), indent=2)[:5000], "type": "json"}

        # HTML → text extraction
        text = _strip_html(resp.text)
        return {"url": url, "content": text[:5000], "type": "html"}


# ═══════════════════════════════════════════════════════════════════════════
# HTML stripping helper
# ═══════════════════════════════════════════════════════════════════════════

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text)


def _strip_html(html: str) -> str:
    if not html:
        return ""
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
    except Exception:
        # Fallback: regex strip
        return re.sub(r"<[^>]+>", " ", html).strip()
    return " ".join(stripper.parts)
