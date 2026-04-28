"""SEC EDGAR ingestion adapter.

Per master directive section 3.1: 10-K, 10-Q, 8-K filings are a primary
Alpha Ledger input. SEC requires a User-Agent header that identifies the
requester (https://www.sec.gov/os/accessing-edgar-data); requests without
one will be blocked.

The client is split into pure URL / parsing helpers (testable without the
network) and a thin fetch shim. Tests inject a mock fetcher so no real
HTTP is performed in CI.

EDGAR's submissions JSON shape (relevant subset):
    {
      "cik": "0000320193",
      "name": "Apple Inc.",
      "tickers": ["AAPL"],
      "filings": {
        "recent": {
          "accessionNumber": ["0000320193-24-000123", ...],
          "form": ["10-K", ...],
          "filingDate": ["2024-11-01", ...],
          "primaryDocument": ["aapl-20240928.htm", ...]
        }
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
SUBMISSIONS_HOST = "data.sec.gov"
ARCHIVE_HOST = "www.sec.gov"

Fetcher = Callable[[str, dict[str, str]], bytes]


class FilingMetadata(BaseModel):
    """One row from EDGAR's submissions feed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cik: Annotated[int, Field(ge=0, le=99_99999_99999)]
    accession_number: Annotated[str, Field(pattern=r"^\d{10}-\d{2}-\d{6}$")]
    form: str
    filing_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    primary_document: str
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION


class CompanyFilings(BaseModel):
    """All filings (in the recent feed) for a single CIK."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cik: Annotated[int, Field(ge=0, le=99_99999_99999)]
    name: str
    tickers: list[str]
    filings: list[FilingMetadata]
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION


def submissions_url(cik: int) -> str:
    """URL for a CIK's submissions JSON. CIK is left-padded to 10 digits."""
    if cik < 0:
        raise ValueError("cik must be non-negative")
    return f"https://{SUBMISSIONS_HOST}/submissions/CIK{cik:010d}.json"


def filing_url(cik: int, accession_number: str, primary_document: str) -> str:
    """URL for the primary document of a filing.

    EDGAR strips the dashes from the accession number for the archive path.
    """
    if cik < 0:
        raise ValueError("cik must be non-negative")
    if not accession_number:
        raise ValueError("accession_number must be non-empty")
    if not primary_document:
        raise ValueError("primary_document must be non-empty")
    acc = accession_number.replace("-", "")
    return (
        f"https://{ARCHIVE_HOST}/Archives/edgar/data/{cik}/{acc}/{primary_document}"
    )


def parse_company_submissions(data: dict, cik: int) -> CompanyFilings:
    """Parse the raw EDGAR submissions JSON into a CompanyFilings record."""
    name = str(data.get("name", ""))
    tickers = list(data.get("tickers", []))
    recent = data.get("filings", {}).get("recent", {})
    accession_numbers = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(len(accession_numbers), len(forms), len(filing_dates), len(primary_docs))
    filings = [
        FilingMetadata(
            cik=cik,
            accession_number=accession_numbers[i],
            form=forms[i],
            filing_date=filing_dates[i],
            primary_document=primary_docs[i],
        )
        for i in range(n)
    ]
    return CompanyFilings(cik=cik, name=name, tickers=tickers, filings=filings)


def filter_by_form(
    filings: list[FilingMetadata], forms: set[str]
) -> list[FilingMetadata]:
    """Keep only filings whose `form` is in `forms` (e.g. {"10-K", "10-Q"})."""
    return [f for f in filings if f.form in forms]


def filter_by_date_range(
    filings: list[FilingMetadata],
    start_date: str,
    end_date: str,
) -> list[FilingMetadata]:
    """Keep filings with `start_date <= filing_date <= end_date` (ISO YYYY-MM-DD)."""
    return [f for f in filings if start_date <= f.filing_date <= end_date]


def content_hash(content: bytes) -> str:
    """SHA-256 hex digest of fetched filing content -- the corpus dedup key."""
    return hashlib.sha256(content).hexdigest()


def _default_fetch(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


class EDGARClient:
    """Thin SEC EDGAR client. Pure URL / parsing logic + injectable fetcher.

    `user_agent` MUST identify the requester per SEC policy. Standard form:
        "Project Name <email@example.com>"
    """

    def __init__(
        self,
        user_agent: str,
        *,
        fetcher: Fetcher | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError(
                "user_agent must be non-empty -- SEC EDGAR rejects requests "
                "without an identifying User-Agent header"
            )
        self.user_agent = user_agent
        self._fetcher = fetcher or _default_fetch

    def _headers(self, host: str) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/html, */*",
            "Host": host,
        }

    def fetch_submissions(self, cik: int) -> dict:
        """Fetch and JSON-decode the submissions feed for a CIK."""
        url = submissions_url(cik)
        body = self._fetcher(url, self._headers(SUBMISSIONS_HOST))
        return json.loads(body)

    def get_company_filings(self, cik: int) -> CompanyFilings:
        """Fetch and parse a company's submissions feed in one call."""
        return parse_company_submissions(self.fetch_submissions(cik), cik)

    def fetch_filing(self, filing: FilingMetadata) -> bytes:
        """Fetch the primary-document bytes for a filing."""
        url = filing_url(filing.cik, filing.accession_number, filing.primary_document)
        return self._fetcher(url, self._headers(ARCHIVE_HOST))
