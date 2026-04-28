"""Tests for the run-manifest reproducibility primitive."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.manifests.run_manifest import (
    RunManifest,
    hash_corpus,
    hash_file,
    hash_string,
    manifest_id,
    read_manifest,
    write_manifest,
)


def _sha(s: str) -> str:
    return hash_string(s)


def _example() -> RunManifest:
    return RunManifest(
        corpus_hash=_sha("corpus"),
        persona_hash=_sha("persona-v1"),
        model_hash=_sha("deepseek-r1-32b-q6_k"),
        lockfile_hash=_sha("uv.lock"),
        seed=42,
    )


def test_hash_string_is_deterministic_and_correct_length() -> None:
    a = hash_string("foo")
    b = hash_string("foo")
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_hash_string_differs_for_different_inputs() -> None:
    assert hash_string("foo") != hash_string("bar")


def test_hash_file(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert hash_file(f) == hash_string("hello")


def test_hash_corpus_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha")
    b.write_text("beta")
    assert hash_corpus([a, b]) == hash_corpus([b, a])


def test_hash_corpus_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("v1")
    h1 = hash_corpus([a])
    a.write_text("v2")
    h2 = hash_corpus([a])
    assert h1 != h2


def test_manifest_id_deterministic() -> None:
    m = _example()
    assert manifest_id(m) == manifest_id(m)


def test_manifest_id_changes_with_any_field() -> None:
    base = _example()
    base_id = manifest_id(base)

    bumps = [
        base.model_copy(update={"corpus_hash": _sha("other-corpus")}),
        base.model_copy(update={"persona_hash": _sha("other-persona")}),
        base.model_copy(update={"model_hash": _sha("other-model")}),
        base.model_copy(update={"lockfile_hash": _sha("other-lock")}),
        base.model_copy(update={"seed": 43}),
    ]
    ids = {manifest_id(b) for b in bumps}
    assert base_id not in ids
    assert len(ids) == len(bumps)


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    m = _example()
    p = tmp_path / "subdir" / "manifest.json"
    write_manifest(m, p)
    assert p.exists()
    rebuilt = read_manifest(p)
    assert rebuilt == m


def test_manifest_rejects_short_hash() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            corpus_hash="too-short",
            persona_hash=_sha("p"),
            model_hash=_sha("m"),
            lockfile_hash=_sha("l"),
            seed=0,
        )


def test_manifest_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            corpus_hash=_sha("c"),
            persona_hash=_sha("p"),
            model_hash=_sha("m"),
            lockfile_hash=_sha("l"),
            seed=0,
            unsanctioned="leak",
        )
