"""Tests for the Module I inference subsystem -- backend, parser, smoke."""

from __future__ import annotations

import json

from modules.module_1_extraction.inference.backend import (
    InferenceBackend,
    NullBackend,
)


# --- backend tests -------------------------------------------------------


def test_null_backend_emits_valid_json() -> None:
    backend = NullBackend()
    out = backend.generate("system", "user")
    parsed = json.loads(out)
    assert "doc_hash" in parsed
    assert "macro_sentiment" in parsed


def test_null_backend_is_deterministic_per_input() -> None:
    backend = NullBackend()
    a = backend.generate("sys", "user", seed=0)
    b = backend.generate("sys", "user", seed=0)
    assert a == b


def test_null_backend_seed_changes_output() -> None:
    backend = NullBackend()
    a = backend.generate("sys", "user", seed=0)
    b = backend.generate("sys", "user", seed=1)
    assert a != b


def test_null_backend_protocol_compliance() -> None:
    """NullBackend is structurally an InferenceBackend (Protocol satisfaction)."""
    backend: InferenceBackend = NullBackend()
    assert backend.name == "null_v1"
    assert backend.model_id == "null"
