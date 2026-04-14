"""Generic web page fetcher."""

from __future__ import annotations

import json

from ._base import _rate_limited_get, _strip_html


class WebFetcher:
    """Fetches and extracts text content from web pages."""

    def fetch(self, url: str) -> dict[str, str]:
        """Fetch a URL and extract text content."""
        resp = _rate_limited_get("web", url)
        content_type = resp.headers.get("Content-Type", "")

        if "json" in content_type:
            return {"url": url, "content": json.dumps(resp.json(), indent=2)[:5000], "type": "json"}

        # HTML -> text extraction
        text = _strip_html(resp.text)
        return {"url": url, "content": text[:5000], "type": "html"}
