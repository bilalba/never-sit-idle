"""Reddit API client."""

from __future__ import annotations

from typing import Any

import requests

from ._base import USER_AGENT, _rate_limited_get


class RedditClient:
    """Reddit API client using public JSON endpoints (read-only).

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
