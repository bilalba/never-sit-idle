"""Hacker News Firebase API client."""

from __future__ import annotations

from typing import Any

from ._base import _rate_limited_get, _strip_html


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
