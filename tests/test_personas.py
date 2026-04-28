"""Tests for the persona library -- registry, schema, content invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.personas.registry import (
    DEFINITIONS_DIR,
    list_personas,
    load_all_personas,
    load_persona,
)
from shared.personas.schema import PersonaSpec

EXPECTED_PERSONAS: frozenset[str] = frozenset(
    {"supply_chain_analyst_v1", "hawkish_fed_strategist_v1"}
)


def test_definitions_directory_exists() -> None:
    assert DEFINITIONS_DIR.is_dir()


def test_list_personas_includes_expected_set() -> None:
    available = set(list_personas())
    missing = EXPECTED_PERSONAS - available
    assert not missing, f"missing personas: {sorted(missing)}"


def test_load_each_known_persona() -> None:
    for pid in EXPECTED_PERSONAS:
        spec = load_persona(pid)
        assert spec.persona_id == pid


def test_load_all_personas_round_trip() -> None:
    all_personas = load_all_personas()
    assert set(all_personas.keys()) >= EXPECTED_PERSONAS
    for pid, spec in all_personas.items():
        assert spec.persona_id == pid


def test_each_persona_has_substantive_prompt_and_metadata() -> None:
    for pid in list_personas():
        spec = load_persona(pid)
        assert spec.name, f"{pid}: empty name"
        assert spec.description, f"{pid}: empty description"
        assert spec.behavioral_signature, f"{pid}: empty behavioral_signature"
        assert spec.bias, f"{pid}: empty bias"
        assert len(spec.keywords) >= 3, f"{pid}: needs at least 3 keywords"
        assert len(spec.system_prompt) >= 200, f"{pid}: system prompt too short"


def test_each_persona_prompt_mentions_alpha_ledger_schema() -> None:
    """Every persona must instruct the model to follow the Alpha Ledger output
    contract from directive section 3.4 -- otherwise downstream validation
    will silently reject every record."""
    for pid in list_personas():
        spec = load_persona(pid)
        assert "Alpha Ledger" in spec.system_prompt, (
            f"{pid}: system prompt does not reference the Alpha Ledger schema"
        )


def test_each_persona_prompt_self_identifies() -> None:
    """The persona must instruct the model to set persona_id correctly --
    without this, attribution back to the prompt is broken."""
    for pid in list_personas():
        spec = load_persona(pid)
        assert pid in spec.system_prompt, (
            f"{pid}: system prompt does not instruct setting persona_id={pid}"
        )


def test_prompt_hash_is_deterministic() -> None:
    available = list_personas()
    if not available:
        pytest.skip("no personas registered yet")
    pid = available[0]
    a = load_persona(pid)
    b = load_persona(pid)
    assert a.prompt_hash == b.prompt_hash
    assert len(a.prompt_hash) == 64


def test_prompt_hashes_differ_across_personas() -> None:
    hashes = {load_persona(pid).prompt_hash for pid in list_personas()}
    assert len(hashes) == len(list_personas()), "two personas share a prompt hash"


def test_persona_spec_is_frozen() -> None:
    available = list_personas()
    if not available:
        pytest.skip("no personas registered yet")
    pid = available[0]
    spec = load_persona(pid)
    with pytest.raises(ValidationError):
        spec.system_prompt = "modified"  # type: ignore[misc]


def test_persona_id_pattern_enforced() -> None:
    with pytest.raises(ValidationError):
        PersonaSpec(
            persona_id="bad name with spaces",
            name="x",
            description="x",
            behavioral_signature="x",
            bias="x",
            keywords=["a"],
            system_prompt="x" * 300,
        )


def test_missing_persona_raises_with_helpful_message() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_persona("nonexistent_persona_v99")
