"""Data source clients for knowledge gathering.

Sources:
    - Reddit (OAuth2, strict rate-limit tracking)
    - Wikipedia
    - Hacker News (Firebase API)
    - StackOverflow / StackExchange
    - GitHub (public repos, no auth required for reads)
    - Generic web page fetcher
    - Google News (searchable RSS)
    - GDELT (global news/events API with tone scores)
    - Feedsearch (RSS feed discovery for any domain)
    - Yahoo Finance (quotes, search, news via public endpoints)
    - Alpha Vantage (stocks, news sentiment)

All sources share the RateLimiter to ensure we never exceed limits.
"""

from agent.sources._base import (
    RateLimitState,
    RateLimiter,
    rate_limiter,
    _rate_limited_get,
    _parse_rss_items,
    _strip_html,
)
from agent.sources._reddit import RedditClient
from agent.sources._wikipedia import WikipediaClient
from agent.sources._hackernews import HackerNewsClient
from agent.sources._stackexchange import StackExchangeClient
from agent.sources._github import GitHubClient
from agent.sources._web import WebFetcher
from agent.sources._news import GoogleNewsClient, GDELTClient, FeedsearchClient
from agent.sources._finance import YFinanceClient, AlphaVantageClient

__all__ = [
    "RateLimitState",
    "RateLimiter",
    "rate_limiter",
    "_rate_limited_get",
    "_parse_rss_items",
    "_strip_html",
    "RedditClient",
    "WikipediaClient",
    "HackerNewsClient",
    "StackExchangeClient",
    "GitHubClient",
    "WebFetcher",
    "GoogleNewsClient",
    "GDELTClient",
    "FeedsearchClient",
    "YFinanceClient",
    "AlphaVantageClient",
]
