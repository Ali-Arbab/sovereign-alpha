"""Tests for the BLS ingestion adapter."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from modules.module_1_extraction.ingestion.bls import (
    BLS_API_URL,
    BLS_HOST,
    BLSClient,
    BLSDataPoint,
    CommonSeries,
    parse_bls_response,
)


def _success_payload(series_id: str = "CUUR0000SA0") -> dict:
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 123,
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": series_id,
                    "data": [
                        {
                            "year": "2024",
                            "period": "M03",
                            "periodName": "March",
                            "value": "313.548",
                            "footnotes": [{}],
                        },
                        {
                            "year": "2024",
                            "period": "M02",
                            "periodName": "February",
                            "value": "312.230",
                            "footnotes": [{}],
                        },
                    ],
                }
            ]
        },
    }


def test_common_series_ids_are_strings_with_expected_format() -> None:
    assert CommonSeries.CPI_ALL_ITEMS == "CUUR0000SA0"
    assert CommonSeries.UNEMPLOYMENT_RATE == "LNS14000000"
    assert CommonSeries.NONFARM_PAYROLLS == "CES0000000001"


def test_parse_success_payload() -> None:
    series = parse_bls_response(_success_payload())
    assert len(series) == 1
    s = series[0]
    assert s.series_id == "CUUR0000SA0"
    assert len(s.data) == 2
    assert s.data[0].year == 2024
    assert s.data[0].period == "M03"
    assert s.data[0].value == pytest.approx(313.548)


def test_parse_raises_on_request_failure() -> None:
    bad = {
        "status": "REQUEST_NOT_PROCESSED",
        "message": ["Series does not exist"],
        "Results": {"series": []},
    }
    with pytest.raises(RuntimeError, match="BLS API error"):
        parse_bls_response(bad)


def test_parse_raises_on_missing_results_series() -> None:
    bad = {"status": "REQUEST_SUCCEEDED", "Results": {}}
    with pytest.raises(ValueError, match=r"Results\.series"):
        parse_bls_response(bad)


def test_parse_raises_on_series_without_id() -> None:
    bad = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {"series": [{"data": [{"year": "2024", "period": "M01", "value": "1.0"}]}]},
    }
    with pytest.raises(ValueError, match="seriesID"):
        parse_bls_response(bad)


def test_data_point_validates_period_format() -> None:
    with pytest.raises(ValidationError):
        BLSDataPoint(
            series_id="X", year=2024, period="not-a-period", period_name="x", value=1.0
        )


def test_data_point_validates_year_range() -> None:
    with pytest.raises(ValidationError):
        BLSDataPoint(
            series_id="X", year=1800, period="M01", period_name="x", value=1.0
        )


def test_client_without_key_omits_key_from_body() -> None:
    captured: list[tuple[str, dict[str, str], bytes]] = []

    def mock_fetch(url: str, headers: dict[str, str], body: bytes) -> bytes:
        captured.append((url, headers, body))
        return json.dumps(_success_payload()).encode()

    client = BLSClient(fetcher=mock_fetch)
    series = client.get_series([CommonSeries.CPI_ALL_ITEMS], 2023, 2024)
    assert len(series) == 1

    url, headers, body = captured[0]
    assert url == BLS_API_URL
    assert headers["Host"] == BLS_HOST
    assert headers["Content-Type"] == "application/json"
    body_dict = json.loads(body)
    assert body_dict["seriesid"] == [CommonSeries.CPI_ALL_ITEMS]
    assert body_dict["startyear"] == "2023"
    assert body_dict["endyear"] == "2024"
    assert "registrationkey" not in body_dict


def test_client_with_key_includes_key_in_body() -> None:
    captured: list[bytes] = []

    def mock_fetch(url: str, headers: dict[str, str], body: bytes) -> bytes:
        captured.append(body)
        return json.dumps(_success_payload()).encode()

    client = BLSClient(registration_key="abc123", fetcher=mock_fetch)
    client.get_series([CommonSeries.CPI_ALL_ITEMS], 2023, 2024)
    body_dict = json.loads(captured[0])
    assert body_dict["registrationkey"] == "abc123"


def test_client_rejects_invalid_args() -> None:
    client = BLSClient(fetcher=lambda u, h, b: b"")
    with pytest.raises(ValueError, match="series_ids"):
        client.fetch_series_raw([], 2020, 2024)
    with pytest.raises(ValueError, match="start_year"):
        client.fetch_series_raw(["X"], 2024, 2020)
    with pytest.raises(ValueError, match=r"\[1900, 2100\]"):
        client.fetch_series_raw(["X"], 1850, 2024)


def test_client_round_trip_returns_typed_series() -> None:
    def mock_fetch(url: str, headers: dict[str, str], body: bytes) -> bytes:
        return json.dumps(_success_payload("LNS14000000")).encode()

    client = BLSClient(fetcher=mock_fetch)
    series = client.get_series([CommonSeries.UNEMPLOYMENT_RATE], 2023, 2024)
    assert series[0].series_id == "LNS14000000"
    assert all(isinstance(p, BLSDataPoint) for p in series[0].data)
