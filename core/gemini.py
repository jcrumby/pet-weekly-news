"""Generic Gemini utility helpers for weekly news automations."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import date, datetime
from typing import Any, Callable, Iterable, Sequence

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
from google import generativeai as genai

load_dotenv()

default_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_URL_PATTERN = r"https?://[^\s)\]]+"
_DEFAULT_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d %B %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%b %d, %Y",
)


def _today_str() -> str:
    return date.today().strftime("%B %d, %Y")


def get_model(model_name: str | None = None):
    """Return a configured Gemini model instance."""

    return genai.GenerativeModel(model_name or default_model_name)


def _normalise_response_text(response: Any) -> str:
    text = getattr(response, "text", "")
    if text:
        return text.strip()

    try:
        candidate = response.candidates[0]
        part = candidate.content.parts[0]
        text = getattr(part, "text", "")
    except Exception:
        text = ""
    return text.strip()


def generate_text_with_retry(
    content: str | Sequence[str],
    *,
    model_name: str | None = None,
    max_retries: int = 6,
    base_delay_seconds: float = 2.0,
    response_mime_type: str | None = None,
) -> str:
    """Call Gemini with exponential backoff on transient failures."""

    if isinstance(content, (list, tuple)):
        payload = list(content)
    else:
        payload = [content]

    model = get_model(model_name)

    attempt = 0
    while True:
        try:
            kwargs = {}
            if response_mime_type:
                kwargs["generation_config"] = {"response_mime_type": response_mime_type}
            response = model.generate_content(payload, **kwargs)
            return _normalise_response_text(response)
        except Exception as exc:  # pragma: no cover
            attempt += 1
            if attempt > max_retries:
                raise

            message = str(exc)
            should_backoff = (
                "429" in message
                or "rate" in message.lower()
                or "quota" in message.lower()
                or "exceeded" in message.lower()
                or "temporary" in message.lower()
                or "unavailable" in message.lower()
            )
            if not should_backoff:
                time.sleep(0.5)
                continue

            delay = min(base_delay_seconds * (2 ** (attempt - 1)), 30)
            delay += 0.25 * (1 + (attempt % 3))
            time.sleep(delay)


def _clean_json_fence(text: str) -> str:
    return re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.DOTALL)


def generate_json_with_retry(
    content: str | Sequence[str],
    *,
    model_name: str | None = None,
    max_retries: int = 6,
    base_delay_seconds: float = 2.0,
) -> Any:
    """Return parsed JSON from Gemini output."""

    text = generate_text_with_retry(
        content,
        model_name=model_name,
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
        response_mime_type="application/json",
    )
    cleaned = _clean_json_fence(text)
    return json.loads(cleaned or "null")


def gemini_extract_links(
    markdown: str,
    prompt_template: str,
    *,
    today_str: str | None = None,
    non_article_substrings: Sequence[str] | None = None,
    model_name: str | None = None,
    url_pattern: str = _URL_PATTERN,
) -> list[str]:
    """Extract article URLs using a caller-supplied prompt template.

    ``prompt_template`` should contain placeholders ``{today}`` and
    ``{markdown}`` which will be interpolated prior to calling Gemini.
    """

    prompt = prompt_template.format(today=today_str or _today_str(), markdown=markdown)
    text = generate_text_with_retry(prompt, model_name=model_name)
    urls = re.findall(url_pattern, text)

    if non_article_substrings:
        urls = [
            url for url in urls if not any(substr.lower() in url.lower() for substr in non_article_substrings)
        ]

    # Preserve order while deduplicating
    unique: dict[str, None] = {}
    for url in urls:
        unique.setdefault(url, None)
    return list(unique.keys())


def gemini_summarize(
    article_markdowns: Sequence[str],
    prompt_template: str,
    *,
    today_str: str | None = None,
    model_name: str | None = None,
    separator: str = "\n\n---\n\n",
) -> str:
    """Summarise articles using a single Gemini prompt."""

    prompt = prompt_template.format(
        today=today_str or _today_str(),
        articles=separator.join(article_markdowns),
    )
    return generate_text_with_retry(prompt, model_name=model_name)


def merge_source_payloads(
    partials: Sequence[dict[str, Any]],
    *,
    today_str: str | None = None,
    date_key: str = "date",
    sources_key: str = "sources",
    site_key: str = "site",
    articles_key: str = "articles",
    publication_key: str = "publication_date",
    date_formats: Sequence[str] = _DEFAULT_DATE_FORMATS,
) -> dict[str, Any]:
    """Merge multi-batch summary payloads by site and deduplicate articles."""

    site_to_articles: dict[str, list[dict[str, Any]]] = {}
    for partial in partials:
        for src in partial.get(sources_key, []):
            site = src.get(site_key) or "Unknown"
            site_to_articles.setdefault(site, []).extend(src.get(articles_key, []))

    def _parse_pub_date(value: str | None) -> datetime | None:
        if not value:
            return None
        v = value.strip()
        for fmt in date_formats:
            try:
                return datetime.strptime(v, fmt)
            except Exception:
                continue
        try:
            return datetime.strptime(v[:10], "%Y-%m-%d")
        except Exception:
            return None

    merged_sources = []
    for site, articles in site_to_articles.items():
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for art in articles:
            url = (art.get("url") or "").strip().lower()
            title_norm = (art.get("title") or "").strip().lower()
            if url and url in seen_urls:
                continue
            if title_norm and title_norm in seen_titles:
                continue
            if url:
                seen_urls.add(url)
            if title_norm:
                seen_titles.add(title_norm)
            deduped.append(art)

        for art in deduped:
            art["_parsed_dt"] = _parse_pub_date(art.get(publication_key))
        deduped.sort(key=lambda art: (art["_parsed_dt"] is None, art["_parsed_dt"]))
        deduped.reverse()
        for art in deduped:
            art.pop("_parsed_dt", None)

        merged_sources.append({site_key: site, articles_key: deduped})

    return {
        date_key: today_str or _today_str(),
        sources_key: merged_sources,
    }


def gemini_summarize_batched(
    article_markdowns: Sequence[str],
    prompt_template: str,
    *,
    today_str: str | None = None,
    model_name: str | None = None,
    batch_size: int = 8,
    separator: str = "\n\n---\n\n",
    merge_strategy: Callable[[Sequence[dict[str, Any]]], dict[str, Any]] | None = None,
    date_formats: Sequence[str] = _DEFAULT_DATE_FORMATS,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Summarise articles in batches and optionally merge the results."""

    if not article_markdowns:
        payload = {"date": today_str or _today_str(), "sources": []}
        return payload if merge_strategy else [payload]

    partials: list[dict[str, Any]] = []
    for start in range(0, len(article_markdowns), batch_size):
        batch = article_markdowns[start:start + batch_size]
        prompt = prompt_template.format(
            today=today_str or _today_str(),
            articles=separator.join(batch),
        )
        try:
            partial = generate_json_with_retry([prompt], model_name=model_name)
        except Exception:
            continue
        if isinstance(partial, dict):
            partials.append(partial)

    if merge_strategy:
        return merge_strategy(partials)

    return partials


def gemini_extract_companies(
    summary_text: str,
    prompt_template: str,
    *,
    model_name: str | None = None,
) -> str:
    """Extract company names from a summary text via Gemini."""

    prompt = prompt_template.format(summary=summary_text)
    return generate_text_with_retry(prompt, model_name=model_name)


def evaluate_articles_with_gemini(
    articles: Sequence[dict[str, Any]],
    prompt_template: str,
    *,
    model_name: str | None = None,
    batch_size: int = 30,
    max_retries: int = 6,
    base_delay_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Score articles using a caller-supplied evaluation prompt."""

    evaluated: list[dict[str, Any]] = []
    for idx in range(0, len(articles), batch_size):
        batch = articles[idx:idx + batch_size]
        try:
            payload = generate_json_with_retry(
                [prompt_template, json.dumps(batch, ensure_ascii=False)],
                model_name=model_name,
                max_retries=max_retries,
                base_delay_seconds=base_delay_seconds,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Failed to evaluate batch {idx // batch_size + 1}: {exc}")
            continue

        if isinstance(payload, list):
            evaluated.extend(payload)
        else:
            print(f"Unexpected Gemini response for batch {idx // batch_size + 1}: {payload!r}")
    return evaluated


def _default_summary_builder(
    template: str,
    batch: Sequence[dict[str, Any]],
    *,
    text_key: str = "full_text",
    max_chars: int = 6000,
) -> str:
    articles_json = json.dumps(
        [
            {
                "title": art.get("title"),
                "url": art.get("url"),
                "text": str(
                    art.get(text_key)
                    or art.get("_feed_full_text")
                    or art.get("description")
                    or art.get("summary")
                    or ""
                )[:max_chars],
            }
            for art in batch
        ],
        ensure_ascii=False,
    )
    return template.format(articles_json=articles_json)


async def crawl_and_summarize_articles(
    articles: list[dict[str, Any]],
    *,
    min_relevance: int = 0,
    sponsor_names: Sequence[str] | None = None,
    summary_prompt_template: str | None = None,
    summary_prompt_builder: Callable[[Sequence[dict[str, Any]]], str] | None = None,
    crawler_kwargs: dict[str, Any] | None = None,
    batch_size: int = 5,
    throttle_seconds: float = 0.1,
    pause_between_batches: float = 4.0,
    model_name: str | None = None,
    text_max_chars: int = 6000,
) -> list[dict[str, Any]]:
    """Crawl article bodies and summarise them via Gemini.

    ``summary_prompt_template`` should include ``{articles_json}`` if
    ``summary_prompt_builder`` is not provided.
    """

    if summary_prompt_builder is None:
        if summary_prompt_template is None:
            raise ValueError("Provide either summary_prompt_template or summary_prompt_builder")
        summary_prompt_builder = lambda batch: _default_summary_builder(
            summary_prompt_template,
            batch,
            max_chars=text_max_chars,
        )

    selected = [a for a in articles if a.get("relevance_score", 0) >= min_relevance]
    if not selected:
        return articles

    async def _fetch_markdown(crawler: AsyncWebCrawler, url: str) -> str:
        result = await crawler.arun(url=url)
        return result.markdown.raw_markdown if result.success else ""

    need_crawl = [
        art for art in selected
        if not art.get("_skip_crawl") and not art.get("full_text")
    ]

    if need_crawl:
        async with AsyncWebCrawler(**(crawler_kwargs or {})) as crawler:
            for art in need_crawl:
                art["full_text"] = await _fetch_markdown(crawler, art.get("url", ""))
                if throttle_seconds:
                    await asyncio.sleep(throttle_seconds)

    if sponsor_names:
        pattern_cache: dict[str, re.Pattern[str]] = {}
        for art in selected:
            text_to_check = " ".join([
                art.get("full_text") or art.get("_feed_full_text") or "",
                art.get("title") or "",
            ])
            matches: list[str] = []
            for sponsor in sponsor_names:
                pattern = pattern_cache.setdefault(sponsor, re.compile(r"\b" + re.escape(sponsor) + r"\b", re.IGNORECASE))
                if pattern.search(text_to_check):
                    matches.append(sponsor)
            art["sponsors"] = sorted(set(matches))

    model = get_model(model_name)
    for idx in range(0, len(selected), batch_size):
        batch = selected[idx:idx + batch_size]
        prompt = summary_prompt_builder(batch)
        try:
            response = model.generate_content(
                [prompt],
                generation_config={"response_mime_type": "application/json"},
            )
            payload = json.loads(_clean_json_fence(_normalise_response_text(response)))
            lookup = {item.get("url"): item.get("summary", "").strip() for item in payload if isinstance(item, dict)}
            for art in batch:
                summary = lookup.get(art.get("url"))
                if summary:
                    art["summary"] = summary
        except Exception:
            pass
        if pause_between_batches:
            await asyncio.sleep(pause_between_batches)

    for art in selected:
        art.pop("full_text", None)
        art.pop("_skip_crawl", None)

    return articles


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
