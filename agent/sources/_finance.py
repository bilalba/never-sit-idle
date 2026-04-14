"""Finance source clients: Yahoo Finance and Alpha Vantage."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from ._base import _rate_limited_get, _parse_rss_items


# ═══════════════════════════════════════════════════════════════════════════
# Yahoo Finance (quotes, search, news)
# ═══════════════════════════════════════════════════════════════════════════


class YFinanceClient:
    """Yahoo Finance public API client.

    Uses public query endpoints — no API key needed.
    Provides stock quotes, ticker search, and news via RSS.
    """

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for tickers/companies."""
        resp = _rate_limited_get("yfinance", "https://query2.finance.yahoo.com/v1/finance/search", params={
            "q": query, "quotesCount": min(limit, 20), "newsCount": 0,
        })
        data = resp.json()
        results = []
        for q in data.get("quotes", [])[:limit]:
            results.append({
                "symbol": q.get("symbol", ""),
                "name": q.get("shortname", q.get("longname", "")),
                "type": q.get("quoteType", ""),
                "exchange": q.get("exchange", ""),
            })
        return results

    def quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote data for a ticker symbol."""
        resp = _rate_limited_get(
            "yfinance",
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}",
            params={"interval": "1d", "range": "5d"},
        )
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return {"error": f"No data for {symbol}"}
        meta = result[0].get("meta", {})
        return {
            "symbol": meta.get("symbol", symbol),
            "currency": meta.get("currency", ""),
            "exchange": meta.get("exchangeName", ""),
            "price": meta.get("regularMarketPrice", 0),
            "previous_close": meta.get("previousClose", 0),
            "market_time": meta.get("regularMarketTime", 0),
            "day_high": meta.get("regularMarketDayHigh", 0),
            "day_low": meta.get("regularMarketDayLow", 0),
        }

    def news(self, symbol: str, limit: int = 10) -> list[dict[str, str]]:
        """Get news for a ticker via Yahoo Finance RSS."""
        url = f"https://finance.yahoo.com/rss/headline?s={quote_plus(symbol)}"
        resp = _rate_limited_get("yfinance", url)
        return _parse_rss_items(resp.text, limit)


# ═══════════════════════════════════════════════════════════════════════════
# Alpha Vantage (stocks, news sentiment, fundamentals)
# ═══════════════════════════════════════════════════════════════════════════


class AlphaVantageClient:
    """Alpha Vantage API client.

    Requires an API key (free tier: 25 requests/day).
    Provides stock quotes, ticker search, and news with sentiment scores.
    """

    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params["apikey"] = self.api_key
        resp = _rate_limited_get("alphavantage", self.BASE, params=params)
        return resp.json()

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for tickers/companies."""
        data = self._get({"function": "SYMBOL_SEARCH", "keywords": query})
        results = []
        for match in data.get("bestMatches", [])[:limit]:
            results.append({
                "symbol": match.get("1. symbol", ""),
                "name": match.get("2. name", ""),
                "type": match.get("3. type", ""),
                "region": match.get("4. region", ""),
                "currency": match.get("8. currency", ""),
            })
        return results

    def quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote for a ticker."""
        data = self._get({"function": "GLOBAL_QUOTE", "symbol": symbol})
        q = data.get("Global Quote", {})
        if not q:
            return {"error": f"No data for {symbol}"}
        return {
            "symbol": q.get("01. symbol", symbol),
            "price": q.get("05. price", ""),
            "change": q.get("09. change", ""),
            "change_percent": q.get("10. change percent", ""),
            "volume": q.get("06. volume", ""),
            "latest_trading_day": q.get("07. latest trading day", ""),
            "previous_close": q.get("08. previous close", ""),
            "open": q.get("02. open", ""),
            "high": q.get("03. high", ""),
            "low": q.get("04. low", ""),
        }

    def news_sentiment(self, tickers: str = "", topics: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Get news with sentiment scores.

        Args:
            tickers: Comma-separated ticker symbols (e.g. 'AAPL,MSFT').
            topics: Comma-separated topics (e.g. 'technology,earnings').
            limit: Max articles (max 200).
        """
        params: dict[str, Any] = {"function": "NEWS_SENTIMENT", "limit": min(limit, 200)}
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        data = self._get(params)
        articles = []
        for item in data.get("feed", [])[:limit]:
            article: dict[str, Any] = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published": item.get("time_published", ""),
                "summary": item.get("summary", "")[:500],
                "overall_sentiment_score": item.get("overall_sentiment_score", 0),
                "overall_sentiment_label": item.get("overall_sentiment_label", ""),
            }
            ticker_sentiments = []
            for ts in item.get("ticker_sentiment", []):
                ticker_sentiments.append({
                    "ticker": ts.get("ticker", ""),
                    "sentiment_score": ts.get("ticker_sentiment_score", ""),
                    "sentiment_label": ts.get("ticker_sentiment_label", ""),
                    "relevance_score": ts.get("relevance_score", ""),
                })
            if ticker_sentiments:
                article["ticker_sentiment"] = ticker_sentiments
            articles.append(article)
        return articles
