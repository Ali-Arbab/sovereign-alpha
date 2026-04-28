# ADR-0008: Persona definitions in TOML, parsed via stdlib `tomllib`

**Status:** Accepted
**Date:** 2026-04-28

## Context
Master directive §3.2 elevates personas to first-class versioned objects: "system prompts inject analyst archetypes that constrain reasoning style. Personas are versioned and hash-tracked." Per §7, "the persona space is the alpha space" — persona prompts are the search dimension that downstream Module I inference sweeps across. They are written and reviewed by both code authors and prompt engineers; both groups need to be able to edit them without Python expertise.

## Decision
**Persona definitions live in TOML files** under `shared/personas/definitions/<persona_id>.toml`, parsed via stdlib `tomllib`. Each file declares the persona's `persona_id` (regex-validated to enforce the trailing `_vN` suffix), name, description, behavioral signature, bias label, keyword watch-list, and the `system_prompt` itself. A `prompt_hash` is auto-computed by `PersonaSpec` (Pydantic `@computed_field`).

## Alternatives considered
- **Python dicts in a module.** Easiest for code authors but locks out non-Python contributors. Also: changing a prompt requires a code review on a `.py` file rather than a content review on a config file.
- **YAML.** More expressive than TOML but adds a non-stdlib dependency (`pyyaml`). TOML covers our needs and `tomllib` is stdlib in 3.11+.
- **JSON.** No multi-line string support for prompts; would force escape-hell.
- **Database (sqlite, postgres).** Overkill; loses git-versioning of prompt history.

## Consequences
- Zero new runtime dependencies (TOML is stdlib).
- Non-Python contributors edit prompts directly in a config-style file.
- Versioning is mechanical via filename: change a prompt → bump from `_v1.toml` to `_v2.toml` → both files coexist → run manifests of older runs hash differently and remain reproducible.
- `prompt_hash` is auto-derived, so no chance of a stale hash field drifting from the actual prompt content.
