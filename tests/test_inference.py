"""Tests for the Module I inference subsystem -- backend, parser, smoke."""

from __future__ import annotations

import json

import pytest

from modules.module_1_extraction.inference.backend import (
    InferenceBackend,
    NullBackend,
)
from modules.module_1_extraction.inference.parser import parse_alpha_ledger_json


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


# --- parser tests --------------------------------------------------------


def test_parser_handles_plain_json() -> None:
    out = parse_alpha_ledger_json('{"a": 1}')
    assert out == {"a": 1}


def test_parser_handles_code_fence() -> None:
    text = "Here is the record:\n```json\n{\"a\": 1, \"b\": 2}\n```\nDone."
    out = parse_alpha_ledger_json(text)
    assert out == {"a": 1, "b": 2}


def test_parser_handles_unfenced_inline_json() -> None:
    text = "Sure, here you go: {\"a\": 1, \"b\": 2} -- end."
    out = parse_alpha_ledger_json(text)
    assert out == {"a": 1, "b": 2}


def test_parser_handles_nested_object() -> None:
    text = 'prefix {"a": {"b": 1, "c": [2, 3]}} suffix'
    out = parse_alpha_ledger_json(text)
    assert out == {"a": {"b": 1, "c": [2, 3]}}


def test_parser_handles_braces_inside_strings() -> None:
    """Braces inside string literals must not confuse the depth counter."""
    text = 'prefix {"a": "{ not a brace }", "b": 1} suffix'
    out = parse_alpha_ledger_json(text)
    assert out == {"a": "{ not a brace }", "b": 1}


def test_parser_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_alpha_ledger_json("")


def test_parser_rejects_no_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_alpha_ledger_json("the model just answered in prose, no json")


def test_parser_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        parse_alpha_ledger_json(123)  # type: ignore[arg-type]


def test_parser_takes_first_object_when_multiple() -> None:
    text = 'first: {"a": 1} second: {"a": 2}'
    out = parse_alpha_ledger_json(text)
    assert out == {"a": 1}
