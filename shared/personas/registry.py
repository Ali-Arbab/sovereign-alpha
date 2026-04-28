"""Persona registry -- discovers and loads TOML-defined personas."""

from __future__ import annotations

import tomllib
from pathlib import Path

from shared.personas.schema import PersonaSpec

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def list_personas() -> list[str]:
    """Return all persona ids found in the definitions directory, sorted."""
    return sorted(p.stem for p in DEFINITIONS_DIR.glob("*.toml"))


def load_persona(persona_id: str) -> PersonaSpec:
    """Load a single persona by id. Raises FileNotFoundError if missing."""
    path = DEFINITIONS_DIR / f"{persona_id}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"persona {persona_id!r} not found at {path}; "
            f"available: {', '.join(list_personas()) or '(none)'}"
        )
    with path.open("rb") as f:
        data = tomllib.load(f)
    return PersonaSpec(**data)


def load_all_personas() -> dict[str, PersonaSpec]:
    """Load every persona in the registry, keyed by persona_id."""
    return {pid: load_persona(pid) for pid in list_personas()}
