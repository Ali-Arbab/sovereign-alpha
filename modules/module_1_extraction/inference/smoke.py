"""Inference smoke-test harness -- end-to-end Module I plumbing validation.

Per master directive section 0.5.1.A bullet 13. Runs a small document
sample through (persona prompt + LLM backend + output parse + schema
validation) and reports counts plus a few sample records. Output is
tagged with the backend's `model_id` and the persona's `persona_id`;
callers are expected to mark the run with `phase=bootstrap,
discard_for_research=True` until a research-grade backend lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.module_1_extraction.inference.backend import InferenceBackend
from modules.module_1_extraction.inference.parser import parse_alpha_ledger_json
from shared.personas.schema import PersonaSpec
from shared.schemas.alpha_ledger import AlphaLedgerRecord

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Analyze the following document and produce a single Alpha Ledger "
    "record as a JSON object matching the schema described in your "
    "system prompt. Return ONLY the JSON object, no prose.\n\n"
    "Document:\n{text}\n\nJSON output:"
)


@dataclass(frozen=True)
class SmokeFailure:
    """One failed document, with the reason and a short preview."""

    doc_hash: str
    reason: str
    preview: str = ""


@dataclass(frozen=True)
class SmokeTestResult:
    n_docs_attempted: int
    n_records_validated: int
    failures: list[SmokeFailure]
    sample_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.n_docs_attempted == 0:
            return 0.0
        return self.n_records_validated / self.n_docs_attempted


def run_smoke_test(
    *,
    backend: InferenceBackend,
    persona: PersonaSpec,
    documents: list[tuple[str, str]],
    max_docs: int = 100,
    seed: int = 0,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    sample_size: int = 5,
) -> SmokeTestResult:
    """Run the smoke-test pipeline on `documents`.

    `documents` is a list of `(doc_hash, text)` pairs. For each document:
        1. Inject `persona.system_prompt` as the system message.
        2. Format the user message with the document text.
        3. Call `backend.generate()`.
        4. Parse the output as JSON.
        5. Override `doc_hash`, `persona_id`, `model_id`, `schema_version`
           with the run's actual values (so a malformed model output
           cannot poison metadata).
        6. Validate against AlphaLedgerRecord.

    Failures are collected with a short preview of the offending output;
    they do not abort the run.
    """
    if max_docs < 0:
        raise ValueError("max_docs must be non-negative")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    docs = documents[:max_docs]
    failures: list[SmokeFailure] = []
    samples: list[dict[str, Any]] = []
    n_validated = 0

    for doc_hash, text in docs:
        try:
            output = backend.generate(
                persona.system_prompt,
                user_prompt_template.format(text=text),
                seed=seed,
            )
        except Exception as e:
            failures.append(
                SmokeFailure(doc_hash=doc_hash, reason=f"backend.generate raised: {e!r}")
            )
            continue

        try:
            record_dict = parse_alpha_ledger_json(output)
        except ValueError as e:
            failures.append(
                SmokeFailure(
                    doc_hash=doc_hash,
                    reason=f"parse failure: {e}",
                    preview=output[:200],
                )
            )
            continue

        # Overwrite metadata so a malformed model output cannot lie about
        # provenance. Real backends sometimes hallucinate persona_id values.
        record_dict["doc_hash"] = doc_hash
        record_dict["persona_id"] = persona.persona_id
        record_dict["model_id"] = backend.model_id
        record_dict["schema_version"] = "1.0.0"

        try:
            AlphaLedgerRecord.model_validate(record_dict)
        except Exception as e:
            failures.append(
                SmokeFailure(
                    doc_hash=doc_hash,
                    reason=f"schema validation failed: {e!s}"[:300],
                    preview=str(record_dict)[:200],
                )
            )
            continue

        n_validated += 1
        if len(samples) < sample_size:
            samples.append(record_dict)

    return SmokeTestResult(
        n_docs_attempted=len(docs),
        n_records_validated=n_validated,
        failures=failures,
        sample_records=samples,
    )
