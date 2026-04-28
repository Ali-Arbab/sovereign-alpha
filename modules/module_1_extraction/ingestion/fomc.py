"""FOMC ingestion adapter.

Per master directive section 3.1: Federal Reserve statements, minutes, the
Summary of Economic Projections (SEP / dot plot), and press-conference
transcripts are primary inputs to the hawkish_fed_strategist persona. Fed
URLs follow predictable patterns keyed off the meeting end date in
YYYYMMDD form, so URL construction is pure-functional and the network
surface is a thin injectable shim.

URL patterns (verified against federalreserve.gov circa 2024):
- Statement:        /newsevents/pressreleases/monetary{YYYYMMDD}a.htm
- Minutes:          /monetarypolicy/fomcminutes{YYYYMMDD}.htm
- SEP / dot plot:   /monetarypolicy/files/fomcprojtabl{YYYYMMDD}.pdf
- Press conference: /monetarypolicy/fomcpresconf{YYYYMMDD}.htm

Note: SEP is only published at the four quarterly meetings (March, June,
September, December). Press conferences happen at every meeting since 2019.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
FED_HOST = "www.federalreserve.gov"

ArtifactType = Literal["statement", "minutes", "projections", "press_conference"]
Fetcher = Callable[[str, dict[str, str]], bytes]


def _parse_date(d: date | str) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def statement_url(meeting_end_date: date | str) -> str:
    """URL for the policy statement released at the end of an FOMC meeting."""
    d = _parse_date(meeting_end_date)
    return (
        f"https://{FED_HOST}/newsevents/pressreleases/"
        f"monetary{d.strftime('%Y%m%d')}a.htm"
    )


def minutes_url(meeting_end_date: date | str) -> str:
    """URL for the minutes (typically released ~3 weeks after the meeting)."""
    d = _parse_date(meeting_end_date)
    return f"https://{FED_HOST}/monetarypolicy/fomcminutes{d.strftime('%Y%m%d')}.htm"


def projections_url(meeting_end_date: date | str) -> str:
    """URL for the SEP / dot-plot PDF. Only valid at quarterly meetings."""
    d = _parse_date(meeting_end_date)
    return (
        f"https://{FED_HOST}/monetarypolicy/files/"
        f"fomcprojtabl{d.strftime('%Y%m%d')}.pdf"
    )


def press_conference_url(meeting_end_date: date | str) -> str:
    """URL for the press-conference page (links to video, statement, transcript)."""
    d = _parse_date(meeting_end_date)
    return f"https://{FED_HOST}/monetarypolicy/fomcpresconf{d.strftime('%Y%m%d')}.htm"


_URL_FUNCS: dict[ArtifactType, Callable[[date | str], str]] = {
    "statement": statement_url,
    "minutes": minutes_url,
    "projections": projections_url,
    "press_conference": press_conference_url,
}


class FOMCArtifact(BaseModel):
    """One downloadable artifact from a single FOMC meeting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    meeting_end_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    artifact_type: ArtifactType
    url: str
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION


def all_artifacts_for_meeting(meeting_end_date: date | str) -> list[FOMCArtifact]:
    """Return FOMCArtifacts for every URL pattern for a meeting.

    Includes `projections` even at non-quarterly meetings (the URL exists
    but will 404 on fetch). Callers should filter by artifact_type or use
    response status to detect missing artifacts.
    """
    d = _parse_date(meeting_end_date)
    iso = d.isoformat()
    return [
        FOMCArtifact(meeting_end_date=iso, artifact_type=t, url=fn(d))
        for t, fn in _URL_FUNCS.items()
    ]


def _default_fetch(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


class FOMCClient:
    """Thin federalreserve.gov client. Pure URL helpers + injectable fetcher.

    A descriptive User-Agent is good citizenship even though FOMC URLs do
    not strictly require it. Construct with
        `user_agent="Sovereign Alpha Research <email>"`.
    """

    def __init__(
        self,
        user_agent: str,
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
            "Accept": "text/html, application/pdf, */*",
            "Host": FED_HOST,
        }

    def fetch_artifact(self, artifact: FOMCArtifact) -> bytes:
        """Fetch the raw bytes of an artifact (HTML for most, PDF for projections)."""
        return self._fetcher(artifact.url, self._headers())
