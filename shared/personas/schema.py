"""PersonaSpec -- the versioned, hash-tracked contract for analyst personas."""

from __future__ import annotations

import hashlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field

SCHEMA_VERSION = "1.0.0"


class PersonaSpec(BaseModel):
    """A versioned analyst persona used as the system prompt for Module I inference.

    A change to `system_prompt` MUST bump `persona_id` (e.g. trailing _v1 -> _v2)
    so downstream run manifests hash differently and old reproducibility is
    preserved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*_v\d+$")]
    name: str
    description: str
    behavioral_signature: str
    bias: str
    keywords: list[str]
    system_prompt: str
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prompt_hash(self) -> str:
        """SHA-256 of the system prompt -- the content-addressed fingerprint
        used in run manifests."""
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()
