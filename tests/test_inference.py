"""Tests for the Module I inference subsystem -- backend, parser, smoke."""

from __future__ import annotations

import hashlib
import json

import pytest

from modules.module_1_extraction.inference.backend import (
    InferenceBackend,
    NullBackend,
)
from modules.module_1_extraction.inference.parser import parse_alpha_ledger_json
from modules.module_1_extraction.inference.smoke import (
    DEFAULT_USER_PROMPT_TEMPLATE,
    run_smoke_test,
)
from shared.personas.registry import load_persona

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


# --- smoke-test runner tests --------------------------------------------


def _docs(n: int) -> list[tuple[str, str]]:
    """Build (doc_hash, text) fixtures with real SHA-256 hashes."""
    return [
        (
            "sha256:" + hashlib.sha256(f"doc-{i}".encode()).hexdigest(),
            f"This is document number {i} for the smoke test.",
        )
        for i in range(n)
    ]


def test_smoke_test_with_null_backend_validates_all_records() -> None:
    persona = load_persona("supply_chain_analyst_v1")
    backend = NullBackend()
    result = run_smoke_test(
        backend=backend, persona=persona, documents=_docs(10), max_docs=10
    )
    assert result.n_docs_attempted == 10
    assert result.n_records_validated == 10
    assert result.failures == []
    assert result.success_rate == 1.0
    assert len(result.sample_records) == 5  # default sample_size


def test_smoke_test_overwrites_metadata_with_run_truth() -> None:
    """A model can lie about persona_id / model_id; the harness must override."""
    persona = load_persona("supply_chain_analyst_v1")
    backend = NullBackend()
    result = run_smoke_test(
        backend=backend, persona=persona, documents=_docs(3), max_docs=3
    )
    for rec in result.sample_records:
        assert rec["persona_id"] == "supply_chain_analyst_v1"
        assert rec["model_id"] == "null"


def test_smoke_test_collects_failures_from_a_broken_backend() -> None:
    class _BrokenBackend:
        name = "broken_v1"
        model_id = "broken"

        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            max_tokens: int = 1024,
            seed: int = 0,
        ) -> str:
            return "the model returned prose, no json here"

    persona = load_persona("supply_chain_analyst_v1")
    result = run_smoke_test(
        backend=_BrokenBackend(), persona=persona, documents=_docs(5), max_docs=5
    )
    assert result.n_records_validated == 0
    assert len(result.failures) == 5
    for f in result.failures:
        assert "parse" in f.reason


def test_smoke_test_caps_at_max_docs() -> None:
    persona = load_persona("supply_chain_analyst_v1")
    result = run_smoke_test(
        backend=NullBackend(),
        persona=persona,
        documents=_docs(50),
        max_docs=10,
    )
    assert result.n_docs_attempted == 10


def test_smoke_test_handles_backend_exception_gracefully() -> None:
    class _RaisingBackend:
        name = "raising_v1"
        model_id = "raising"

        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            max_tokens: int = 1024,
            seed: int = 0,
        ) -> str:
            raise RuntimeError("simulated backend failure")

    persona = load_persona("supply_chain_analyst_v1")
    result = run_smoke_test(
        backend=_RaisingBackend(), persona=persona, documents=_docs(3), max_docs=3
    )
    assert result.n_records_validated == 0
    assert len(result.failures) == 3
    for f in result.failures:
        assert "backend.generate raised" in f.reason


def test_smoke_test_default_user_prompt_template_includes_text_placeholder() -> None:
    """Sanity: the template must format with a `text` kwarg without raising."""
    formatted = DEFAULT_USER_PROMPT_TEMPLATE.format(text="example doc body")
    assert "example doc body" in formatted


def test_smoke_test_invalid_args() -> None:
    persona = load_persona("supply_chain_analyst_v1")
    with pytest.raises(ValueError, match="max_docs"):
        run_smoke_test(
            backend=NullBackend(), persona=persona, documents=[], max_docs=-1
        )
    with pytest.raises(ValueError, match="sample_size"):
        run_smoke_test(
            backend=NullBackend(),
            persona=persona,
            documents=[],
            max_docs=0,
            sample_size=-1,
        )
