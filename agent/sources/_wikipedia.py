"""Wikipedia API client."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from ._base import _rate_limited_get, _strip_html


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
