"""Shared AsyncWebCrawler helpers extracted from project-specific pipelines.

The helpers in this module are intentionally generic so that individual
projects provide their own crawling strategy (URL lists, throttling rules,
robots policies, etc.) while reusing the same Crawl4AI plumbing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from crawl4ai import AsyncWebCrawler


@asynccontextmanager
async def webcrawler(**kwargs) -> AsyncIterator[AsyncWebCrawler]:
    """Async context manager wrapping ``AsyncWebCrawler``.

    Provides a common entry point for projects that need to perform grouped
    crawl operations while sharing a single crawler instance.
    """
    async with AsyncWebCrawler(**kwargs) as crawler:
        yield crawler


async def fetch_markdown(
    crawler: AsyncWebCrawler,
    url: str,
    *,
    extractor: Optional[Callable[[Any], str]] = None,
) -> str:
    """Fetch ``url`` and return markdown extracted from the Crawl4AI result.

    ``extractor`` allows callers to specify a custom transformation (e.g. to
    pull HTML or plain text) without duplicating the success handling.
    """
    result = await crawler.arun(url=url)
    if not result.success:
        return ""

    if extractor:
        return extractor(result)

    return result.markdown.raw_markdown


# Backwards-compatible aliases (existing projects can keep using them until
# their imports are updated).
async def crawl_markdown_ai_infra(crawler: AsyncWebCrawler, url: str) -> str:  # pragma: no cover
    return await fetch_markdown(crawler, url)


async def crawl_markdown_fitness(crawler: AsyncWebCrawler, url: str) -> str:  # pragma: no cover
    return await fetch_markdown(crawler, url)
