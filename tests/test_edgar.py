"""Tests for the EDGAR ingestion adapter -- URLs, parsing, mocked fetch."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from modules.module_1_extraction.ingestion.edgar import (
    ARCHIVE_HOST,
    SUBMISSIONS_HOST,
    CompanyFilings,
    EDGARClient,
    FilingMetadata,
    content_hash,
    filing_url,
    filter_by_date_range,
    filter_by_form,
    parse_company_submissions,
    submissions_url,
)

APPLE_CIK = 320_193


def _sample_submissions_json() -> dict:
    return {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000098",
                    "0000320193-23-000150",
                ],
                "form": ["10-K", "10-Q", "8-K"],
                "filingDate": ["2024-11-01", "2024-08-02", "2023-12-15"],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240629.htm",
                    "aapl-20231215.htm",
                ],
            }
        },
    }


def test_submissions_url_zero_pads_cik() -> None:
    assert submissions_url(APPLE_CIK) == f"https://{SUBMISSIONS_HOST}/submissions/CIK0000320193.json"


def test_submissions_url_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        submissions_url(-1)


def test_filing_url_strips_dashes_from_accession() -> None:
    url = filing_url(APPLE_CIK, "0000320193-24-000123", "aapl-20240928.htm")
    assert url == (
        f"https://{ARCHIVE_HOST}/Archives/edgar/data/{APPLE_CIK}/"
        f"000032019324000123/aapl-20240928.htm"
    )


def test_filing_url_validates_args() -> None:
    with pytest.raises(ValueError, match="cik"):
        filing_url(-1, "0000320193-24-000123", "x.htm")
    with pytest.raises(ValueError, match="accession_number"):
        filing_url(APPLE_CIK, "", "x.htm")
    with pytest.raises(ValueError, match="primary_document"):
        filing_url(APPLE_CIK, "0000320193-24-000123", "")


def test_parse_company_submissions_shape() -> None:
    parsed = parse_company_submissions(_sample_submissions_json(), APPLE_CIK)
    assert isinstance(parsed, CompanyFilings)
    assert parsed.cik == APPLE_CIK
    assert parsed.name == "Apple Inc."
    assert parsed.tickers == ["AAPL"]
    assert len(parsed.filings) == 3
    assert {f.form for f in parsed.filings} == {"10-K", "10-Q", "8-K"}


def test_parse_handles_uneven_arrays_by_truncating_to_min() -> None:
    """If EDGAR returns mismatched arrays, we take min length to be safe."""
    data = {
        "name": "X",
        "tickers": [],
        "filings": {
            "recent": {
                "accessionNumber": ["0000123456-24-000001", "0000123456-24-000002"],
                "form": ["10-K"],
                "filingDate": ["2024-01-01"],
                "primaryDocument": ["x.htm"],
            }
        },
    }
    parsed = parse_company_submissions(data, 123_456)
    assert len(parsed.filings) == 1


def test_filter_by_form_keeps_only_listed_forms() -> None:
    parsed = parse_company_submissions(_sample_submissions_json(), APPLE_CIK)
    annual_quarterly = filter_by_form(parsed.filings, {"10-K", "10-Q"})
    assert len(annual_quarterly) == 2
    assert all(f.form in {"10-K", "10-Q"} for f in annual_quarterly)


def test_filter_by_date_range() -> None:
    parsed = parse_company_submissions(_sample_submissions_json(), APPLE_CIK)
    in_2024 = filter_by_date_range(parsed.filings, "2024-01-01", "2024-12-31")
    assert {f.filing_date for f in in_2024} == {"2024-11-01", "2024-08-02"}


def test_content_hash_deterministic() -> None:
    a = content_hash(b"hello world")
    b = content_hash(b"hello world")
    assert a == b
    assert len(a) == 64
    assert content_hash(b"hello") != content_hash(b"world")


def test_filing_metadata_validates_accession_format() -> None:
    with pytest.raises(ValidationError):
        FilingMetadata(
            cik=APPLE_CIK,
            accession_number="not-an-accession",
            form="10-K",
            filing_date="2024-01-01",
            primary_document="x.htm",
        )


def test_filing_metadata_validates_date_format() -> None:
    with pytest.raises(ValidationError):
        FilingMetadata(
            cik=APPLE_CIK,
            accession_number="0000320193-24-000123",
            form="10-K",
            filing_date="not-a-date",
            primary_document="x.htm",
        )


def test_client_rejects_empty_user_agent() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EDGARClient(user_agent="   ")


def test_client_passes_user_agent_in_headers() -> None:
    captured: list[tuple[str, dict[str, str]]] = []

    def mock_fetch(url: str, headers: dict[str, str]) -> bytes:
        captured.append((url, headers))
        return json.dumps(_sample_submissions_json()).encode()

    client = EDGARClient(user_agent="Test Agent <test@example.com>", fetcher=mock_fetch)
    parsed = client.get_company_filings(APPLE_CIK)
    assert parsed.cik == APPLE_CIK
    assert len(captured) == 1
    url, headers = captured[0]
    assert "submissions/CIK0000320193.json" in url
    assert headers["User-Agent"] == "Test Agent <test@example.com>"
    assert headers["Host"] == SUBMISSIONS_HOST


def test_client_fetch_filing_uses_archive_host() -> None:
    captured_urls: list[str] = []

    def mock_fetch(url: str, headers: dict[str, str]) -> bytes:
        captured_urls.append(url)
        assert headers["Host"] == ARCHIVE_HOST
        return b"<html>filing body</html>"

    client = EDGARClient(user_agent="Test <t@e.com>", fetcher=mock_fetch)
    filing = FilingMetadata(
        cik=APPLE_CIK,
        accession_number="0000320193-24-000123",
        form="10-K",
        filing_date="2024-11-01",
        primary_document="aapl-20240928.htm",
    )
    body = client.fetch_filing(filing)
    assert body == b"<html>filing body</html>"
    assert "Archives/edgar/data" in captured_urls[0]
