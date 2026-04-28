"""Tests for the FOMC ingestion adapter."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from modules.module_1_extraction.ingestion.fomc import (
    FED_HOST,
    FOMCArtifact,
    FOMCClient,
    all_artifacts_for_meeting,
    minutes_url,
    press_conference_url,
    projections_url,
    statement_url,
)


def test_statement_url_matches_fed_pattern() -> None:
    url = statement_url(date(2024, 5, 1))
    assert url == f"https://{FED_HOST}/newsevents/pressreleases/monetary20240501a.htm"


def test_minutes_url_matches_fed_pattern() -> None:
    url = minutes_url(date(2024, 5, 1))
    assert url == f"https://{FED_HOST}/monetarypolicy/fomcminutes20240501.htm"


def test_projections_url_returns_pdf_path() -> None:
    url = projections_url(date(2024, 3, 20))
    assert url.endswith(".pdf")
    assert "fomcprojtabl20240320.pdf" in url


def test_press_conference_url_matches_fed_pattern() -> None:
    url = press_conference_url(date(2024, 5, 1))
    assert url == f"https://{FED_HOST}/monetarypolicy/fomcpresconf20240501.htm"


def test_url_helpers_accept_iso_string() -> None:
    a = statement_url("2024-05-01")
    b = statement_url(date(2024, 5, 1))
    assert a == b


def test_url_helpers_reject_invalid_string() -> None:
    with pytest.raises(ValueError):
        statement_url("not-a-date")


def test_all_artifacts_for_meeting_returns_four_artifacts() -> None:
    arts = all_artifacts_for_meeting(date(2024, 5, 1))
    assert len(arts) == 4
    assert {a.artifact_type for a in arts} == {
        "statement",
        "minutes",
        "projections",
        "press_conference",
    }
    for a in arts:
        assert a.meeting_end_date == "2024-05-01"
        assert a.url.startswith(f"https://{FED_HOST}/")


def test_artifact_validates_meeting_end_date_format() -> None:
    with pytest.raises(ValidationError):
        FOMCArtifact(
            meeting_end_date="2024/05/01",  # wrong separator
            artifact_type="statement",
            url="https://example.com/foo",
        )


def test_artifact_validates_artifact_type_literal() -> None:
    with pytest.raises(ValidationError):
        FOMCArtifact(
            meeting_end_date="2024-05-01",
            artifact_type="not_a_real_type",  # type: ignore[arg-type]
            url="https://example.com/foo",
        )


def test_client_rejects_empty_user_agent() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FOMCClient(user_agent="")


def test_client_passes_user_agent_and_correct_host() -> None:
    captured: list[tuple[str, dict[str, str]]] = []

    def mock_fetch(url: str, headers: dict[str, str]) -> bytes:
        captured.append((url, headers))
        return b"<html>statement body</html>"

    client = FOMCClient(user_agent="Sovereign Alpha <a@b.com>", fetcher=mock_fetch)
    arts = all_artifacts_for_meeting(date(2024, 5, 1))
    body = client.fetch_artifact(arts[0])
    assert body == b"<html>statement body</html>"
    url, headers = captured[0]
    assert headers["User-Agent"] == "Sovereign Alpha <a@b.com>"
    assert headers["Host"] == FED_HOST
    assert "20240501" in url
