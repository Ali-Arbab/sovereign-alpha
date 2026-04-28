"""GDELT Project ingestion adapter -- open news-corpus substitute.

Per master directive section 3.1 (Bloomberg / Reuters / AP newswire) and
0.5.1.A (GDELT named as the open substitute when licensed corpora are
unavailable). Provides article-list search across global news with full
domain / language / source-country metadata.

API: https://api.gdeltproject.org/api/v2/doc/doc -- free, no key, no rate
limit on small queries. Returns up to 250 articles per call. Time range
is bounded by `startdatetime` / `enddatetime` in YYYYMMDDHHMMSS form.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
GDELT_HOST = "api.gdeltproject.org"
GDELT_DOC_URL = f"https://{GDELT_HOST}/api/v2/doc/doc"

MAX_RECORDS_PER_CALL = 250  # GDELT v2 hard limit

Fetcher = Callable[[str, dict[str, str]], bytes]


class GDELTArticle(BaseModel):
    """One article from a GDELT DOC API search response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    title: str
    seendate: Annotated[str, Field(pattern=r"^\d{8}T\d{6}Z$")]
    domain: str
    language: str
    sourcecountry: str = ""
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION


def _format_dt(dt: datetime) -> str:
    """GDELT expects naive YYYYMMDDHHMMSS (UTC implied)."""
    return dt.strftime("%Y%m%d%H%M%S")


def doc_search_url(
    query: str,
    *,
    start_dt: datetime,
    end_dt: datetime,
    mode: str = "ArtList",
    format: str = "json",
    max_records: int = MAX_RECORDS_PER_CALL,
    sort: str = "DateDesc",
) -> str:
    """Build a GDELT DOC API URL.

    `query` supports GDELT's mini-DSL (e.g. `Tesla AND domain:reuters.com`).
    `mode='ArtList'` returns article metadata. `start_dt` / `end_dt` are
    Python datetimes; they are formatted to the API's YYYYMMDDHHMMSS form
    in UTC.
    """
    if not query.strip():
        raise ValueError("query must be non-empty")
    if start_dt > end_dt:
        raise ValueError("start_dt must be <= end_dt")
    if max_records <= 0 or max_records > MAX_RECORDS_PER_CALL:
        raise ValueError(f"max_records must be in (0, {MAX_RECORDS_PER_CALL}]")

    params = {
        "query": query,
        "mode": mode,
        "format": format,
        "maxrecords": str(max_records),
        "startdatetime": _format_dt(start_dt),
        "enddatetime": _format_dt(end_dt),
        "sort": sort,
    }
    return f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"


def parse_articles(payload: dict) -> list[GDELTArticle]:
    """Parse the GDELT ArtList JSON envelope into a list of GDELTArticle."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    raw_articles = payload.get("articles", [])
    if not isinstance(raw_articles, list):
        raise ValueError("payload['articles'] must be a list")
    return [
        GDELTArticle(
            url=a.get("url", ""),
            title=a.get("title", ""),
            seendate=a["seendate"],
            domain=a.get("domain", ""),
            language=a.get("language", ""),
            sourcecountry=a.get("sourcecountry", ""),
        )
        for a in raw_articles
    ]


def _default_fetch(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


class GDELTClient:
    """Thin GDELT v2 DOC API client. Pure URL helpers + injectable fetcher."""

    def __init__(
        self,
        user_agent: str = "Sovereign Alpha Research",
        *,
        fetcher: Fetcher | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("user_agent must be non-empty")
        self.user_agent = user_agent
        self._fetcher = fetcher or _default_fetch

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Host": GDELT_HOST,
        }

    def search_raw(
        self,
        query: str,
        *,
        start_dt: datetime,
        end_dt: datetime,
        max_records: int = MAX_RECORDS_PER_CALL,
    ) -> dict:
        """Fetch and JSON-decode the raw search response."""
        url = doc_search_url(
            query,
            start_dt=start_dt,
            end_dt=end_dt,
            max_records=max_records,
        )
        body = self._fetcher(url, self._headers())
        return json.loads(body)

    def search_articles(
        self,
        query: str,
        *,
        start_dt: datetime,
        end_dt: datetime,
        max_records: int = MAX_RECORDS_PER_CALL,
    ) -> list[GDELTArticle]:
        """Fetch + parse in one call."""
        return parse_articles(
            self.search_raw(
                query, start_dt=start_dt, end_dt=end_dt, max_records=max_records
            )
        )
