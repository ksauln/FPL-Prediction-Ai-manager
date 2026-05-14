"""Lightweight web search helpers for current FPL context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests


CURRENT_INFO_KEYWORDS = {
    "available",
    "availability",
    "doubt",
    "doubtful",
    "fit",
    "fitness",
    "injured",
    "injury",
    "late",
    "latest",
    "news",
    "press conference",
    "return",
    "rotation",
    "rumour",
    "rumor",
    "start",
    "status",
    "suspended",
    "suspension",
}

MAX_RESULT_TEXT_CHARS = 500


@dataclass
class SearchSettings:
    """Runtime settings for online search."""

    enabled: bool = False
    max_results: int = 5
    timeout_seconds: int = 10
    manual_query: str = ""


class SearchResultParser(HTMLParser):
    """Extract DuckDuckGo lite result rows without adding parser dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._active_link: Dict[str, str] | None = None
        self._capture_snippet = False
        self._snippet_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        class_name = attr_map.get("class", "")

        if tag == "a" and "result-link" in class_name:
            self._active_link = {"title": "", "url": self._clean_url(attr_map.get("href", ""))}
            return

        if tag in {"td", "div", "span"} and "result-snippet" in class_name:
            self._capture_snippet = True
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._active_link is not None:
            self._active_link["title"] += f" {text}" if self._active_link["title"] else text
        elif self._capture_snippet:
            self._snippet_parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_link is not None:
            if self._active_link["title"] and self._active_link["url"]:
                self.results.append(self._active_link)
            self._active_link = None
            return

        if self._capture_snippet and tag in {"td", "div", "span"}:
            if self.results and self._snippet_parts and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = " ".join(self._snippet_parts)
            self._capture_snippet = False
            self._snippet_parts = []

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return unquote(target)
        return url


def question_needs_current_info(question: str) -> bool:
    """Return whether a user question likely needs current online information."""

    normalized = question.lower()
    for keyword in CURRENT_INFO_KEYWORDS:
        if " " in keyword and keyword in normalized:
            return True
        if " " not in keyword and re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return True
    return False


def build_search_query(question: str, context: Dict[str, Any], manual_query: str = "") -> str:
    """Build a targeted search query from the user question and squad context."""

    if manual_query.strip():
        return manual_query.strip()

    player_names = _matching_player_names(question, context)
    if not player_names:
        player_names = [item.get("player") for item in context.get("risk_flags", [])]

    player_fragment = " ".join(str(name) for name in player_names[:3] if name)
    question_terms = " ".join(question.strip().split()[:12])
    return f"Fantasy Premier League injury news {player_fragment} {question_terms}".strip()


def search_web(query: str, *, max_results: int = 5, timeout_seconds: int = 10) -> List[Dict[str, str]]:
    """Search DuckDuckGo lite and return result snippets with URLs."""

    if not query.strip():
        return []

    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()

    parser = SearchResultParser()
    parser.feed(response.text)

    seen_urls: set[str] = set()
    clean_results: List[Dict[str, str]] = []
    for result in parser.results:
        url_value = result.get("url", "").strip()
        clean = {
            "source_domain": urlparse(url_value).netloc,
            "url": url_value,
            "untrusted_title": _sanitize_result_text(result.get("title", "")),
            "untrusted_snippet": _sanitize_result_text(result.get("snippet", "")),
        }
        if not clean["untrusted_title"] or not clean["url"] or clean["url"] in seen_urls:
            continue
        seen_urls.add(clean["url"])
        clean_results.append(clean)
        if len(clean_results) >= max_results:
            break
    return clean_results


def _sanitize_result_text(value: str) -> str:
    """Normalize result text and label it as untrusted before prompt use."""

    text = unescape(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_RESULT_TEXT_CHARS]


def build_online_research(
    *,
    question: str,
    context: Dict[str, Any],
    settings: SearchSettings,
) -> Dict[str, Any]:
    """Fetch optional current web context for a question."""

    should_search = settings.enabled and (
        question_needs_current_info(question) or bool(settings.manual_query.strip())
    )
    if not should_search:
        return {
            "enabled": settings.enabled,
            "searched": False,
            "reason": "Question did not appear to require current online information.",
            "query": "",
            "results": [],
        }

    query = build_search_query(question, context, settings.manual_query)
    try:
        results = search_web(
            query,
            max_results=settings.max_results,
            timeout_seconds=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        return {
            "enabled": settings.enabled,
            "searched": True,
            "query": query,
            "error": str(exc),
            "results": [],
        }

    return {
        "enabled": settings.enabled,
        "searched": True,
        "query": query,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }


def _matching_player_names(question: str, context: Dict[str, Any]) -> List[str]:
    """Find squad player names explicitly mentioned in the question."""

    normalized = question.lower()
    matches: List[str] = []
    players: Iterable[Dict[str, Any]] = context.get("user_squad", {}).get("players", [])
    for player in players:
        name = str(player.get("full_name") or "")
        if name and name.lower() in normalized:
            matches.append(name)
    return matches
