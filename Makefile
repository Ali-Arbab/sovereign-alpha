.PHONY: bootstrap-test test lint fmt ci clean

# Top-level targets. The CI workflow runs `make ci`. Local development uses
# whichever target matches the current task.

bootstrap-test:
	uv run python scripts/bootstrap_test.py

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

ci: lint test bootstrap-test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist *.egg-info
