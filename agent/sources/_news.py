"""News source clients: Google News RSS, GDELT, and feed discovery."""

from __future__ import annotations

import json
import re
from typing import Any

from ._base import _rate_limited_get, _parse_rss_items, _strip_html


# ═══════════════════════════════════════════════════════════════════════════
# Google News (searchable RSS)
# ═══════════════════════════════════════════════════════════════════════════


class GoogleNewsClient:
    """Google News RSS — searchable news via RSS feed.

    Uses news.google.com/rss/search?q=QUERY which returns RSS XML.
    No auth needed. Results include title, link, source, and publish date.
    """

    BASE = "https://news.google.com/rss/search"

    def search(self, query: str, limit: int = 10, lang: str = "en", country: str = "US") -> list[dict[str, str]]:
        """Search Google News. Returns list of article dicts."""
        params = {
            "q": query,
            "hl": f"{lang}-{country}",
            "gl": country,
            "ceid": f"{country}:{lang}",
        }
        resp = _rate_limited_get("googlenews", self.BASE, params=params)
        return _parse_rss_items(resp.text, limit)

    def topic(self, topic: str, limit: int = 10) -> list[dict[str, str]]:
        """Get news for a broad topic (WORLD, NATION, BUSINESS, TECHNOLOGY, etc.)."""
        url = f"https://news.google.com/rss/headlines/section/topic/{topic.upper()}"
        resp = _rate_limited_get("googlenews", url, params={"hl": "en-US", "gl": "US", "ceid": "US:en"})
        return _parse_rss_items(resp.text, limit)


# ═══════════════════════════════════════════════════════════════════════════
# GDELT (global news/events API)
# ═══════════════════════════════════════════════════════════════════════════


class GDELTClient:
    """GDELT Project API — global news monitoring with tone/sentiment scores.

    Free, no auth needed. Returns articles with tone scores built in.
    Tone ranges from -100 (extremely negative) to +100 (extremely positive).
    """

    BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

    def search(
        self,
        query: str,
        mode: str = "artlist",
        limit: int = 10,
        timespan: str = "7d",
    ) -> list[dict[str, Any]]:
        """Search GDELT articles.

        Args:
            query: Search keywords.
            mode: 'artlist' for articles, 'tonechart' for tone over time.
            limit: Max results (max 250).
            timespan: Time span like '7d', '30d', '1y'.
        """
        params = {
            "query": query,
            "mode": mode,
            "maxrecords": min(limit, 250),
            "format": "json",
            "timespan": timespan,
        }
        resp = _rate_limited_get("gdelt", self.BASE, params=params)
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return []
        return self._parse_artlist(data, limit)

    def tone_chart(self, query: str, timespan: str = "30d") -> dict[str, Any]:
        """Get tone/sentiment timeline for a query."""
        params = {
            "query": query,
            "mode": "timelinetone",
            "format": "json",
            "timespan": timespan,
        }
        resp = _rate_limited_get("gdelt", self.BASE, params=params)
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return {}

    def _parse_artlist(self, data: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        articles = []
        for art in data.get("articles", [])[:limit]:
            articles.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("domain", ""),
                "language": art.get("language", ""),
                "seendate": art.get("seendate", ""),
                "tone": art.get("tone", 0),
                "socialimage": art.get("socialimage", ""),
            })
        return articles


# ═══════════════════════════════════════════════════════════════════════════
# Feedsearch (RSS feed discovery)
# ═══════════════════════════════════════════════════════════════════════════


class FeedsearchClient:
    """Discovers RSS/Atom feed URLs for any website.

    Checks HTML <link> tags and tries common feed paths.
    No auth needed.
    """

    COMMON_PATHS = [
        "/feed", "/rss", "/rss.xml", "/atom.xml", "/feed.xml",
        "/feeds/posts/default", "/blog/feed", "/index.xml",
    ]

    def discover(self, url: str) -> list[dict[str, str]]:
        """Find RSS/Atom feeds for a URL. Returns list of {url, title, type}."""
        feeds: list[dict[str, str]] = []

        # 1. Fetch the page and look for <link> feed tags
        try:
            resp = _rate_limited_get("feedsearch", url)
            feeds.extend(self._extract_link_feeds(resp.text, url))
        except Exception:
            pass

        # 2. Try common feed paths
        base = self._base_url(url)
        for path in self.COMMON_PATHS:
            feed_url = base + path
            if any(f["url"] == feed_url for f in feeds):
                continue
            try:
                resp = _rate_limited_get("feedsearch", feed_url, max_retries=1)
                content_type = resp.headers.get("Content-Type", "")
                text_start = resp.text[:200]
                if ("xml" in content_type or "rss" in content_type or
                        "<rss" in text_start or "<feed" in text_start or "<?xml" in text_start):
                    feeds.append({"url": feed_url, "title": "", "type": "discovered"})
            except Exception:
                continue

        return feeds

    def fetch_feed(self, feed_url: str, limit: int = 10) -> list[dict[str, str]]:
        """Fetch and parse an RSS/Atom feed URL. Returns list of items."""
        resp = _rate_limited_get("feedsearch", feed_url)
        return _parse_rss_items(resp.text, limit)

    def _extract_link_feeds(self, html: str, page_url: str) -> list[dict[str, str]]:
        """Extract feed URLs from HTML <link> tags."""
        feeds = []
        pattern = r'<link[^>]+type=["\']application/(rss|atom)\+xml["\'][^>]*>'
        for match in re.finditer(pattern, html, re.IGNORECASE):
            tag = match.group(0)
            href_match = re.search(r'href=["\']([^"\']+)["\']', tag)
            title_match = re.search(r'title=["\']([^"\']+)["\']', tag)
            if href_match:
                href = href_match.group(1)
                if href.startswith("/"):
                    href = self._base_url(page_url) + href
                feeds.append({
                    "url": href,
                    "title": title_match.group(1) if title_match else "",
                    "type": match.group(1),
                })
        return feeds

    @staticmethod
    def _base_url(url: str) -> str:
        """Extract scheme + host from a URL."""
        if "://" in url:
            scheme, rest = url.split("://", 1)
            host = rest.split("/", 1)[0]
            return f"{scheme}://{host}"
        return url
