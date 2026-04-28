"""Default ticker -> sector mapping for the bootstrap-phase synthetic Alpha Ledger.

Small mega-cap subset; not a research-grade universe. Module II baseline strategies
will use a richer SPX point-in-time membership table once acquired (per directive
section 0.5.1.D). Sector tags are deliberately lowercase_snake to match the
example in directive section 3.4.
"""

from __future__ import annotations

from typing import Final

DEFAULT_BOOTSTRAP_UNIVERSE: Final[dict[str, str]] = {
    "AAPL": "consumer_electronics",
    "MSFT": "software",
    "GOOGL": "internet",
    "AMZN": "ecommerce",
    "META": "internet",
    "NVDA": "semiconductors",
    "TSLA": "automotive",
    "JPM": "financials",
    "V": "financials",
    "JNJ": "healthcare",
    "PG": "consumer_staples",
    "XOM": "energy",
    "UNH": "healthcare",
    "HD": "retail",
    "WMT": "retail",
}
