"""Inference backend protocol + a deterministic stub backend for tests.

The Protocol decouples the smoke-test harness from any specific LLM
runtime. Real backends (vLLM, llama-cpp-python, OpenAI-compatible servers)
implement `InferenceBackend` and plug in at the call site.

NullBackend produces a valid AlphaLedgerRecord-shaped JSON string keyed
on a seeded hash of the inputs. It is deterministic, schema-conformant,
and ENTIRELY CONTENT-FREE -- the values are noise. Used for CI plumbing
checks and for the bootstrap-test entrypoint until real backends are
wired in (post-hardware).
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol


class InferenceBackend(Protocol):
    """One-shot generative LLM interface."""

    @property
    def name(self) -> str:
        """Versioned backend name; included in run manifest hashing."""
        ...

    @property
    def model_id(self) -> str:
        """Model identifier (e.g. 'deepseek-r1-32b-q6_k', 'qwen2.5-7b-q6')."""
        ...

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 1024,
        seed: int = 0,
    ) -> str:
        """Run inference and return the raw model output as text."""
        ...


class NullBackend:
    """Deterministic stub that emits valid AlphaLedgerRecord JSON.

    Output values are derived from a SHA-256 of the inputs + seed, so
    same-input -> same-output without any model dependency. Tag every
    artifact produced with this backend as bootstrap-only -- the values
    are NOT inferred, they are deterministic noise.
    """

    name: str = "null_v1"
    model_id: str = "null"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 1024,
        seed: int = 0,
    ) -> str:
        h = hashlib.sha256(
            f"{system_prompt}\x00{user_prompt}\x00{seed}".encode()
        ).hexdigest()
        macro = (int(h[0:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        sector = max(-1.0, min(1.0, macro * 0.9))
        conf = int(h[8:16], 16) / 0xFFFFFFFF
        ci_low = max(0.0, conf - 0.05)
        ci_hi = min(1.0, conf + 0.05)
        regime = (int(h[16:24], 16) / 0xFFFFFFFF) < 0.05
        record = {
            "doc_hash": "sha256:" + hashlib.sha256(user_prompt.encode()).hexdigest(),
            "timestamp": "2024-01-01T00:00:00Z",
            "epoch_ns": 1_704_067_200_000_000_000,
            "entities": ["NULL"],
            "sector_tags": ["null"],
            "macro_sentiment": macro,
            "sector_sentiment": sector,
            "confidence_interval": [ci_low, ci_hi],
            "confidence_score": conf,
            "regime_shift_flag": regime,
            "horizon_days": 30,
            "reasoning_trace": "[null backend stub: content is deterministic noise]",
            "persona_id": "null_persona_v1",
            "model_id": self.model_id,
            "schema_version": "1.0.0",
        }
        return json.dumps(record)
