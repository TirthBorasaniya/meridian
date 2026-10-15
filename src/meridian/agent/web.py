"""Web search fallback through Tavily. Results are cited as `[w:<n>]` and marked external."""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class WebSearch(Protocol):
    """Anything that returns web results as dicts with title, url, and content."""

    def search(self, query: str, max_results: int) -> list[dict]: ...


class TavilyWebSearch:
    """Tavily client wrapper."""

    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int) -> list[dict]:
        """Return up to `max_results` results; an API failure returns no results."""
        from tavily import errors as tavily_errors

        try:
            response = self._client.search(query=query, max_results=max_results)
        except (
            tavily_errors.BadRequestError,
            tavily_errors.ForbiddenError,
            tavily_errors.InvalidAPIKeyError,
            tavily_errors.MissingAPIKeyError,
            tavily_errors.UsageLimitExceededError,
            tavily_errors.TimeoutError,
        ) as exc:
            logger.warning("web search failed", extra={"error": str(exc)})
            return []
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in response.get("results", [])[:max_results]
        ]
