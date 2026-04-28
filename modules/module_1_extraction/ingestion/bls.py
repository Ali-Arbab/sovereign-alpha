"""BLS (Bureau of Labor Statistics) ingestion adapter.

Per master directive section 3.1: macroeconomic releases (CPI, employment
situation, JOLTS, AHE) are primary inputs to the hawkish_fed_strategist
persona. BLS exposes a JSON API at api.bls.gov; the protocol differs from
EDGAR / FOMC -- it is POST-with-JSON-body rather than plain GET, so the
fetcher signature is a `(url, headers, body)` triple and the URL is fixed.

Endpoint and limits (as of 2024):
- Base URL: https://api.bls.gov/publicAPI/v2/timeseries/data/
- Free tier: 25 queries/day, no key required.
- Registered tier: 500 queries/day with `registrationkey` in body.

Series IDs are documented at bls.gov; the most commonly tracked ones are
constants on `CommonSeries` below.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
BLS_HOST = "api.bls.gov"
BLS_API_URL = f"https://{BLS_HOST}/publicAPI/v2/timeseries/data/"

PostFetcher = Callable[[str, dict[str, str], bytes], bytes]


class CommonSeries:
    """Most-watched BLS series IDs for macro work."""

    CPI_ALL_ITEMS: Final[str] = "CUUR0000SA0"  # CPI-U, NSA
    CPI_CORE: Final[str] = "CUUR0000SA0L1E"  # CPI-U less food and energy, NSA
    UNEMPLOYMENT_RATE: Final[str] = "LNS14000000"  # SA
    NONFARM_PAYROLLS: Final[str] = "CES0000000001"  # SA, thousands
    AVG_HOURLY_EARNINGS_PRIVATE: Final[str] = "CES0500000003"  # SA, $/hr
    JOLTS_JOB_OPENINGS: Final[str] = "JTS000000000000000JOL"  # SA, thousands
    LABOR_FORCE_PARTICIPATION: Final[str] = "LNS11300000"  # SA, %
    EMPLOYMENT_COST_INDEX: Final[str] = "CIU1010000000000A"  # NSA, all civilian


class BLSDataPoint(BaseModel):
    """One observation from a BLS time series."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    series_id: str
    year: Annotated[int, Field(ge=1900, le=2100)]
    period: Annotated[str, Field(pattern=r"^[MQAS]\d{2}$")]
    period_name: str
    value: float
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION


class BLSSeries(BaseModel):
    """All observations for one series in a single API response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    series_id: str
    data: list[BLSDataPoint]
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION


def parse_bls_response(payload: dict) -> list[BLSSeries]:
    """Parse the BLS v2 JSON envelope into a list of BLSSeries.

    Raises RuntimeError if the response indicates failure (`status` !=
    `REQUEST_SUCCEEDED`); raises ValueError if the structure is malformed.
    """
    status = payload.get("status")
    if status != "REQUEST_SUCCEEDED":
        messages = payload.get("message", [])
        raise RuntimeError(f"BLS API error: status={status!r}, messages={messages}")

    results = payload.get("Results", {})
    series_blocks = results.get("series")
    if not isinstance(series_blocks, list):
        raise ValueError("BLS response missing Results.series array")

    out: list[BLSSeries] = []
    for sb in series_blocks:
        sid = sb.get("seriesID")
        if not sid:
            raise ValueError("series block missing seriesID")
        rows = sb.get("data", [])
        points = [
            BLSDataPoint(
                series_id=sid,
                year=int(row["year"]),
                period=row["period"],
                period_name=row.get("periodName", ""),
                value=float(row["value"]),
            )
            for row in rows
        ]
        out.append(BLSSeries(series_id=sid, data=points))
    return out


def _default_fetch(url: str, headers: dict[str, str], body: bytes) -> bytes:
    req = urllib.request.Request(url, headers=headers, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


class BLSClient:
    """Thin BLS Public Data API v2 client.

    `registration_key` is optional; without it you are on the free 25/day
    tier. With it you are on the 500/day tier. The key is sent in the
    request body, not as a header.
    """

    def __init__(
        self,
        *,
        registration_key: str | None = None,
        fetcher: PostFetcher | None = None,
    ) -> None:
        self.registration_key = registration_key
        self._fetcher = fetcher or _default_fetch

    def _build_body(
        self,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> bytes:
        body: dict = {
            "seriesid": list(series_ids),
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self.registration_key:
            body["registrationkey"] = self.registration_key
        return json.dumps(body).encode("utf-8")

    def fetch_series_raw(
        self,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> dict:
        """POST to BLS and return the raw JSON envelope as a dict."""
        if not series_ids:
            raise ValueError("series_ids must be non-empty")
        if start_year > end_year:
            raise ValueError("start_year must be <= end_year")
        if start_year < 1900 or end_year > 2100:
            raise ValueError("years must be in [1900, 2100]")
        headers = {"Content-Type": "application/json", "Host": BLS_HOST}
        body = self._build_body(series_ids, start_year, end_year)
        return json.loads(self._fetcher(BLS_API_URL, headers, body))

    def get_series(
        self,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> list[BLSSeries]:
        """Fetch and parse the series in one call."""
        return parse_bls_response(
            self.fetch_series_raw(series_ids, start_year, end_year)
        )
