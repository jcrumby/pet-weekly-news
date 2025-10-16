"""Templated email sender utilities shared across projects."""

from __future__ import annotations

import json
import os
import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import jinja2
from dotenv import load_dotenv

load_dotenv()

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = [email.strip() for email in os.getenv("EMAIL_TO", "").split(",") if email.strip()]
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "").strip()


def render_template(
    template_name: str,
    context: Mapping[str, object],
    *,
    template_dir: str | os.PathLike[str] = ".",
) -> str:
    """Render a Jinja2 template with the supplied context."""

    loader = jinja2.FileSystemLoader(searchpath=Path(template_dir))
    env = jinja2.Environment(loader=loader)
    template = env.get_template(template_name)
    return template.render(**context)


def _resolve_recipients(email_to: Sequence[str] | None) -> list[str]:
    recipients = list(email_to) if email_to is not None else EMAIL_TO
    if not recipients:
        raise ValueError("No recipients provided and EMAIL_TO is empty")
    return recipients


def _send_multipart_email(
    *,
    subject: str,
    html_content: str,
    email_from: str | None = None,
    email_to: Sequence[str] | None = None,
    smtp_server: str = SMTP_SERVER,
    smtp_port: int = SMTP_PORT,
    username: str | None = EMAIL_USER,
    password: str | None = EMAIL_PASSWORD,
    sender_name: str | None = EMAIL_FROM_NAME,
) -> None:
    """Send an HTML email using the configured SMTP server."""

    recipients = _resolve_recipients(email_to)
    sender = email_from or username
    if not sender:
        raise ValueError("No sender email configured; set EMAIL_USER or pass email_from")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((sender_name, sender)) if sender_name else sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(msg)


def send_articles_email(
    articles: Iterable[Mapping[str, Any]] | str,
    *,
    min_score: int | None = None,
    max_score: int | None = None,
    score_key: str = "relevance_score",
    sort_key: str = "date",
    date_key: str = "date",
    formatted_date_key: str = "formatted_date",
    date_formatter: Callable[[str], str] | None = None,
    template_name: str = "email_template.html",
    template_dir: str | os.PathLike[str] = ".",
    subject_template: str = "Weekly News Summary - {today}",
    today_format: str = "%B %d, %Y",
    email_to: Sequence[str] | None = None,
    email_from: str | None = None,
    sender_name: str | None = EMAIL_FROM_NAME,
    extra_context: Mapping[str, object] | None = None,
    context_key: str = "articles",
) -> None:
    """Render the supplied article JSON into an email and send it."""

    if isinstance(articles, str):
        with open(articles, "r", encoding="utf-8") as handle:
            articles = json.load(handle)

    article_list = list(articles)

    if min_score is not None:
        article_list = [art for art in article_list if art.get(score_key, 0) >= min_score]
    if max_score is not None:
        article_list = [art for art in article_list if art.get(score_key, 0) <= max_score]

    if date_formatter is None:
        def date_formatter(value: str) -> str:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(today_format)
            except Exception:
                return value

    for art in article_list:
        if date_key in art:
            art[formatted_date_key] = date_formatter(str(art.get(date_key, "")))

    article_list.sort(key=lambda item: item.get(sort_key) or "", reverse=True)

    if not article_list:
        return

    today_str = date.today().strftime(today_format)
    context: dict[str, Any] = {context_key: article_list, "today": today_str}
    if extra_context:
        context.update(extra_context)

    html_content = render_template(template_name, context, template_dir=template_dir)
    subject = subject_template.format(today=today_str)
    _send_multipart_email(
        subject=subject,
        html_content=html_content,
        email_from=email_from,
        email_to=email_to,
        sender_name=sender_name,
    )
