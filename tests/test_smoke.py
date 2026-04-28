"""Smoke test — proves all packages import and the harness is wired correctly."""

import importlib

PACKAGES = (
    "modules",
    "modules.module_1_extraction",
    "modules.module_2_quant",
    "modules.module_3_twin",
    "shared",
    "shared.manifests",
    "shared.schemas",
)


def test_packages_import() -> None:
    for name in PACKAGES:
        assert importlib.import_module(name) is not None
