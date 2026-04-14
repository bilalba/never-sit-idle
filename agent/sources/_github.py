"""GitHub public API client."""

from __future__ import annotations

from typing import Any

from ._base import _rate_limited_get


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
