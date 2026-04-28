"""Tests for the GDELT ingestion adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from modules.module_1_extraction.ingestion.gdelt import (
    GDELT_DOC_URL,
    GDELT_HOST,
    MAX_RECORDS_PER_CALL,
    GDELTArticle,
    GDELTClient,
    doc_search_url,
    parse_articles,
)


def _sample_payload() -> dict:
    return {
        "articles": [
            {
                "url": "https://reuters.com/article/123",
                "url_mobile": "",
                "title": "Fed signals patience on rate cuts",
                "seendate": "20240501T143000Z",
                "socialimage": "",
                "domain": "reuters.com",
                "language": "English",
                "sourcecountry": "US",
            },
            {
                "url": "https://bloomberg.com/article/456",
                "title": "TSMC beats earnings estimates",
                "seendate": "20240501T091500Z",
                "domain": "bloomberg.com",
                "language": "English",
                "sourcecountry": "US",
            },
        ]
    }


def test_doc_search_url_includes_required_params() -> None:
    url = doc_search_url(
        "TSMC",
        start_dt=datetime(2024, 5, 1, 0, 0, 0, tzinfo=UTC),
        end_dt=datetime(2024, 5, 2, 0, 0, 0, tzinfo=UTC),
        max_records=50,
    )
    assert url.startswith(GDELT_DOC_URL + "?")
    assert "query=TSMC" in url
    assert "startdatetime=20240501000000" in url
    assert "enddatetime=20240502000000" in url
    assert "maxrecords=50" in url
    assert "mode=ArtList" in url
    assert "format=json" in url


def test_doc_search_url_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query"):
        doc_search_url(
            "  ",
            start_dt=datetime(2024, 1, 1, tzinfo=UTC),
            end_dt=datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_doc_search_url_rejects_inverted_dates() -> None:
    with pytest.raises(ValueError, match="start_dt"):
        doc_search_url(
            "x",
            start_dt=datetime(2024, 1, 5, tzinfo=UTC),
            end_dt=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_doc_search_url_rejects_max_records_above_limit() -> None:
    with pytest.raises(ValueError, match="max_records"):
        doc_search_url(
            "x",
            start_dt=datetime(2024, 1, 1, tzinfo=UTC),
            end_dt=datetime(2024, 1, 2, tzinfo=UTC),
            max_records=MAX_RECORDS_PER_CALL + 1,
        )


def test_parse_articles_returns_typed_records() -> None:
    arts = parse_articles(_sample_payload())
    assert len(arts) == 2
    assert all(isinstance(a, GDELTArticle) for a in arts)
    assert arts[0].domain == "reuters.com"
    assert arts[0].seendate == "20240501T143000Z"


def test_parse_articles_ignores_unknown_fields_in_envelope() -> None:
    """GDELT envelope only requires `articles`; extra fields like
    `url_mobile` / `socialimage` should be silently ignored."""
    arts = parse_articles(_sample_payload())
    assert arts[0].url == "https://reuters.com/article/123"


def test_parse_articles_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="dict"):
        parse_articles([])  # type: ignore[arg-type]


def test_parse_articles_rejects_non_list_articles() -> None:
    with pytest.raises(ValueError, match="list"):
        parse_articles({"articles": "oops"})


def test_article_validates_seendate_format() -> None:
    with pytest.raises(ValidationError):
        GDELTArticle(
            url="x",
            title="x",
            seendate="2024-05-01T14:30:00Z",  # ISO with separators -- not GDELT format
            domain="x",
            language="English",
        )


def test_client_rejects_empty_user_agent() -> None:
    with pytest.raises(ValueError, match="user_agent"):
        GDELTClient(user_agent="   ")


def test_client_passes_user_agent_and_correct_host() -> None:
    captured: list[tuple[str, dict[str, str]]] = []

    def mock_fetch(url: str, headers: dict[str, str]) -> bytes:
        captured.append((url, headers))
        return json.dumps(_sample_payload()).encode()

    client = GDELTClient(user_agent="Sovereign <a@b.com>", fetcher=mock_fetch)
    arts = client.search_articles(
        "TSMC",
        start_dt=datetime(2024, 5, 1, tzinfo=UTC),
        end_dt=datetime(2024, 5, 2, tzinfo=UTC),
    )
    assert len(arts) == 2
    url, headers = captured[0]
    assert "TSMC" in url
    assert headers["User-Agent"] == "Sovereign <a@b.com>"
    assert headers["Host"] == GDELT_HOST


def test_client_search_raw_returns_dict() -> None:
    def mock_fetch(url: str, headers: dict[str, str]) -> bytes:
        return json.dumps({"articles": []}).encode()

    client = GDELTClient(fetcher=mock_fetch)
    raw = client.search_raw(
        "x",
        start_dt=datetime(2024, 1, 1, tzinfo=UTC),
        end_dt=datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert raw == {"articles": []}
