"""Parse LLM output into an AlphaLedgerRecord-shaped dict.

Real backends rarely return clean JSON -- they wrap it in prose, code
fences, or interleave reasoning. This parser handles the three common
shapes: plain JSON, fenced JSON (```json ... ```), and JSON embedded
in surrounding prose (extracts the first balanced object).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _first_balanced_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_alpha_ledger_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of an Alpha Ledger record dict from LLM output.

    Tries, in order: plain JSON, fenced JSON, first balanced object inside prose.
    Raises ValueError with a short preview if all three fail.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    s = text.strip()
    if not s:
        raise ValueError("empty inference output")

    # 1. Plain JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Code fence
    m = _FENCE_RE.search(s)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. First balanced object
    obj = _first_balanced_object(s)
    if obj is not None:
        return obj

    raise ValueError(
        f"could not extract a JSON object from inference output "
        f"(preview: {s[:200]!r})"
    )
