"""StackOverflow / StackExchange API client."""

from __future__ import annotations

from typing import Any

from ._base import _rate_limited_get, _strip_html


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
