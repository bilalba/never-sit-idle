"""Rate limiting, base HTTP fetch, and shared helpers for data sources."""

from __future__ import annotations

import json
import logging
import re
import time
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

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
rate_limiter.register("reddit", max_requests_per_minute=30, window_seconds=60)
rate_limiter.register("wikipedia", max_requests_per_minute=200, window_seconds=60)
rate_limiter.register("hackernews", max_requests_per_minute=60, window_seconds=60)
rate_limiter.register("stackexchange", max_requests_per_minute=30, window_seconds=60)
rate_limiter.register("github", max_requests_per_minute=60, window_seconds=60)
rate_limiter.register("web", max_requests_per_minute=20, window_seconds=60)
rate_limiter.register("googlenews", max_requests_per_minute=30, window_seconds=60)
rate_limiter.register("gdelt", max_requests_per_minute=30, window_seconds=60)
rate_limiter.register("feedsearch", max_requests_per_minute=10, window_seconds=60)
rate_limiter.register("yfinance", max_requests_per_minute=30, window_seconds=60)
rate_limiter.register("alphavantage", max_requests_per_minute=5, window_seconds=60)


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
# RSS parsing helper
# ═══════════════════════════════════════════════════════════════════════════

def _parse_rss_items(xml_text: str, limit: int) -> list[dict[str, str]]:
    """Parse RSS XML into a list of article dicts."""
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    for item in root.iter("item"):
        entry: dict[str, str] = {}
        title_el = item.find("title")
        if title_el is not None and title_el.text:
            entry["title"] = title_el.text.strip()
        link_el = item.find("link")
        if link_el is not None and link_el.text:
            entry["link"] = link_el.text.strip()
        pub_el = item.find("pubDate")
        if pub_el is not None and pub_el.text:
            entry["published"] = pub_el.text.strip()
        source_el = item.find("source")
        if source_el is not None and source_el.text:
            entry["source"] = source_el.text.strip()
        desc_el = item.find("description")
        if desc_el is not None and desc_el.text:
            entry["description"] = _strip_html(desc_el.text)[:500]
        if entry:
            items.append(entry)
        if len(items) >= limit:
            break
    return items


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
        return re.sub(r"<[^>]+>", " ", html).strip()
    return " ".join(stripper.parts)
