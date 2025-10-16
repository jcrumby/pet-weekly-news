"""Pet Connect weekly automation pipeline."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List
from urllib.parse import urljoin

import requests
from core.crawler import fetch_markdown, webcrawler
from core.email import send_articles_email
from core.gemini import (
    crawl_and_summarize_articles,
    evaluate_articles_with_gemini,
    generate_json_with_retry,
    save_json,
)
from email.utils import parsedate_to_datetime
from html import unescape
from xml.etree import ElementTree as ET
from pet_connect.config import (
    ARTICLE_TEXT_MAX_CHARS,
    ARTICLE_THROTTLE_SECONDS,
    CRAWLER_KWARGS,
    DEFAULT_MAX_ARTICLES_PER_SITE,
    EMAIL_SETTINGS,
    LISTING_THROTTLE_SECONDS,
    NON_ARTICLE_SUBSTRINGS,
    OUTPUT_PATHS,
    RELEVANCE_BATCH_SIZE,
    RELEVANCE_THRESHOLD,
    SOURCE_SITES,
    SPONSOR_KEYWORDS,
    SUMMARY_BATCH_SIZE,
    SUMMARY_PAUSE_SECONDS,
)
from pet_connect.prompts import (
    LISTING_METADATA_PROMPT,
    RELEVANCE_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
)

# --------------------------------------------------------------------------- #
# Markdown parsing helpers
# --------------------------------------------------------------------------- #

ISO_DATE_REGEX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH_NAME_REGEX = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
MONTH_ABBR_REGEX = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?[\s]+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
EU_DATE_REGEX = re.compile(
    r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
HTML_BREAK_TAGS = re.compile(r"</?(?:p|br|div|li|h[1-6])[^>]*>", re.IGNORECASE)
HTML_TAGS = re.compile(r"<[^>]+>")


def _normalise_url(url: str) -> str:
    return url.rstrip("/")


def _clean_html(value: str | None, *, preserve_paragraphs: bool = False) -> str:
    if not value:
        return ""

    text = unescape(value)
    if preserve_paragraphs:
        text = HTML_BREAK_TAGS.sub("\n", text)
    else:
        text = HTML_BREAK_TAGS.sub(" ", text)

    text = HTML_TAGS.sub(" ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    if preserve_paragraphs:
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _parse_rss_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).isoformat()
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _collect_from_rss(site: dict[str, Any], seen_urls: set[str]) -> list[dict[str, Any]]:
    feed_url = site.get("feed_url") or site.get("listing_url")
    if not feed_url:
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }

    try:
        response = requests.get(feed_url, timeout=30, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        print(f"Failed to fetch RSS feed {feed_url}: {exc}")
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        print(f"Failed to parse RSS feed {feed_url}: {exc}")
        return []

    max_articles = site.get("max_articles", DEFAULT_MAX_ARTICLES_PER_SITE)
    use_feed_content = site.get("use_feed_content", False)
    articles: list[dict[str, Any]] = []

    items = root.findall(".//item")
    for item in items:
        link = (item.findtext("link") or "").strip()
        if not link:
            continue
        norm_url = _normalise_url(link)
        if norm_url in seen_urls:
            continue

        title = _clean_html(item.findtext("title")) or "Untitled"
        description_raw = item.findtext("description") or ""
        content_node = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        content_raw = content_node.text if content_node is not None and content_node.text else ""
        pub_date = _parse_rss_date(item.findtext("pubDate"))

        description_text = _clean_html(description_raw)
        description_text_single = description_text.replace("\n", " ").strip()
        content_text = _clean_html(content_raw, preserve_paragraphs=True)
        article = {
            "title": title,
            "description": description_text_single or title,
            "url": link,
            "date": pub_date,
            "source_section": site["name"],
            "sponsors": [],
        }

        text_content = content_text or description_text
        if use_feed_content and content_text:
            article["_feed_full_text"] = content_text
            article["_skip_crawl"] = True

        articles.append(article)
        seen_urls.add(norm_url)

        if len(articles) >= max_articles:
            break

    return articles


def _normalise_date_string(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None

    try:
        # Attempt ISO parsing first
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    parsed = _parse_rss_date(value)
    if parsed:
        return parsed

    for pattern, fmt in (
        (ISO_DATE_REGEX, "%Y-%m-%d"),
        (MONTH_NAME_REGEX, "%B %d, %Y"),
        (MONTH_ABBR_REGEX, "%b %d, %Y"),
        (EU_DATE_REGEX, "%d %B %Y"),
    ):
        match = pattern.search(value)
        if not match:
            continue
        try:
            token = match.group(0).replace(".", "")
            dt = datetime.strptime(token, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            continue

    return value  # Fallback: leave original string so downstream can handle it


async def _collect_from_listing(
    crawler,
    site: dict[str, Any],
    seen_urls: set[str],
) -> list[dict[str, Any]]:
    listing_url = site.get("listing_url")
    if not listing_url:
        return []

    listing_markdown = await fetch_markdown(crawler, listing_url)
    if not listing_markdown:
        return []

    prompt_template = site.get("metadata_prompt", LISTING_METADATA_PROMPT)
    today_str = datetime.utcnow().strftime("%B %d, %Y")
    prompt = prompt_template.format(
        site_name=site["name"],
        today=today_str,
        markdown=listing_markdown,
    )

    try:
        items = generate_json_with_retry(prompt)
    except Exception as exc:
        print(f"Failed to extract metadata for {site['name']}: {exc}")
        return []

    if not isinstance(items, list):
        print(f"Metadata prompt for {site['name']} did not return a list.")
        return []

    articles: list[dict[str, Any]] = []
    max_articles = site.get("max_articles", DEFAULT_MAX_ARTICLES_PER_SITE)

    for raw in items:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url", "")).strip()
        if not url:
            continue
        if not url.lower().startswith("http"):
            url = urljoin(listing_url, url)

        norm_url = _normalise_url(url)
        if norm_url in seen_urls:
            continue

        article = {
            "title": (raw.get("title") or "").strip() or "Untitled",
            "description": (raw.get("description") or "").strip(),
            "url": url,
            "date": _normalise_date_string(raw.get("date")),
            "source_section": site["name"],
            "sponsors": [],
        }

        articles.append(article)
        seen_urls.add(norm_url)

        if len(articles) >= max_articles:
            break

    return articles


# --------------------------------------------------------------------------- #
# Crawling and pipeline orchestration
# --------------------------------------------------------------------------- #

async def _collect_article_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    # Handle RSS feeds first (no Crawl4AI required)
    for site in SOURCE_SITES:
        if site.get("method") == "rss":
            rss_articles = _collect_from_rss(site, seen_urls)
            if rss_articles:
                candidates.extend(rss_articles)

    # Handle listing-based sources with Crawl4AI
    listing_sites = [s for s in SOURCE_SITES if s.get("method", "listing") == "listing"]

    if not listing_sites:
        return candidates

    async with webcrawler(**CRAWLER_KWARGS) as crawler:
        for site in listing_sites:
            articles = await _collect_from_listing(crawler, site, seen_urls)
            if articles:
                candidates.extend(articles)

            if LISTING_THROTTLE_SECONDS:
                await asyncio.sleep(LISTING_THROTTLE_SECONDS)

    return candidates


def _send_email(articles: Iterable[dict[str, Any]]) -> None:
    send_articles_email(
        list(articles),
        min_score=RELEVANCE_THRESHOLD,
        template_name=EMAIL_SETTINGS["template_name"],
        template_dir=EMAIL_SETTINGS["template_dir"],
        subject_template=EMAIL_SETTINGS["subject_template"],
    )


def main() -> None:
    # Step 1: collect article candidates from listings.
    candidates = asyncio.run(_collect_article_candidates())

    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=7)

    recent_candidates = []
    for art in candidates:
        date_str = art.get("date")
        if not date_str:
            recent_candidates.append(art)
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            recent_candidates.append(art)
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt >= cutoff:
            recent_candidates.append(art)

    candidates = recent_candidates

    save_json(candidates, OUTPUT_PATHS["listing_articles"])

    if not candidates:
        print("No articles discovered from listings; skipping remaining steps.")
        return

    # Step 2: evaluate relevance.
    eval_input = [
        {k: v for k, v in art.items() if k not in ("full_text", "_feed_full_text")}
        for art in candidates
    ]

    evaluated_payload = evaluate_articles_with_gemini(
        eval_input,
        RELEVANCE_PROMPT,
        batch_size=RELEVANCE_BATCH_SIZE,
    )

    if len(evaluated_payload) != len(candidates):
        print(f"Warning: Gemini returned {len(evaluated_payload)} items for {len(candidates)} inputs.")

    for original, updated in zip(candidates, evaluated_payload):
        original.update(updated)

    save_json(candidates, OUTPUT_PATHS["evaluated_articles"])

    if not evaluated_payload:
        print("Gemini evaluation returned no data; stopping.")
        return

    for art in candidates:
        if (
            art.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
            and art.get("_feed_full_text")
        ):
            art["full_text"] = art["_feed_full_text"]
            art["_skip_crawl"] = True

    # Step 3: crawl full articles and summarise relevant ones.
    summarized = asyncio.run(
        crawl_and_summarize_articles(
            candidates,
            min_relevance=RELEVANCE_THRESHOLD,
            sponsor_names=SPONSOR_KEYWORDS,
            summary_prompt_template=SUMMARY_PROMPT_TEMPLATE,
            crawler_kwargs=CRAWLER_KWARGS,
            batch_size=SUMMARY_BATCH_SIZE,
            throttle_seconds=ARTICLE_THROTTLE_SECONDS,
            pause_between_batches=SUMMARY_PAUSE_SECONDS,
            text_max_chars=ARTICLE_TEXT_MAX_CHARS,
        )
    )
    save_json(summarized, OUTPUT_PATHS["summaries"])

    # Step 4: send weekly email.
    _send_email(summarized)


if __name__ == "__main__":
    main()
