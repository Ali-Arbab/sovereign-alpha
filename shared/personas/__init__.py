"""Versioned analyst personas -- the alpha-search dimension.

Per master directive section 3.2: "Persona injection: System prompts inject
analyst archetypes that constrain reasoning style. Personas are versioned
and hash-tracked." Per section 7: "the persona space is the alpha space."

Each persona is a TOML file under `definitions/`, loaded via
`shared.personas.registry.load_persona` into a frozen `PersonaSpec`.
Changing a persona's system prompt MUST bump its `persona_id` (e.g.
`supply_chain_v1` -> `supply_chain_v2`), so the run-manifest hash
correctly diverges and old runs remain reproducible.
"""
