format:
	uv run ruff format .

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .

type:
	uv run ty check

test:
	uv run pytest -v -s --cov=agent_core tests

.PHONY: format lint type test
