"""Run-manifest hashing -- the reproducibility primitive.

Per master directive sections 6.2 and 6.3. A `RunManifest` captures the
five-tuple `(corpus_hash, persona_hash, model_hash, seed, lockfile_hash)`
that uniquely identifies a Sovereign Alpha run; `manifest_id()` collapses
it to a single content-addressed string suitable for naming the run's
output directory.

Two runs with byte-identical manifests must produce byte-identical outputs
(directive section 6.2). Two runs with different manifests must not collide.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
_HASH_PATTERN = r"^[0-9a-f]{64}$"


class RunManifest(BaseModel):
    """Five-tuple identifying a single Sovereign Alpha run, plus seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION
    corpus_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]
    persona_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]
    model_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]
    lockfile_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]
    seed: int


def hash_string(s: str) -> str:
    """SHA-256 hex digest of a string (UTF-8)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents (streamed, 64KB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_corpus(paths: Iterable[Path]) -> str:
    """Hash a corpus by hashing each file then hashing the sorted concatenation
    of those digests. Order-independent given a fixed file set."""
    digests = sorted(hash_file(Path(p)) for p in paths)
    h = hashlib.sha256()
    for d in digests:
        h.update(d.encode("ascii"))
    return h.hexdigest()


def manifest_id(manifest: RunManifest) -> str:
    """Single content-addressed ID for a manifest -- usable as a run dir name.

    Stable under JSON sort; changing any field changes the id."""
    payload = json.dumps(manifest.model_dump(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(manifest: RunManifest, out_path: Path) -> None:
    """Persist a manifest as pretty-printed JSON, creating parent dirs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True))


def read_manifest(path: Path) -> RunManifest:
    return RunManifest(**json.loads(Path(path).read_text()))
