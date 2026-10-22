USER_NAME := $(shell whoami)

export UV_CACHE_DIR				:= /goinfre/$(USER_NAME)/.cache/uv
export UV_PROJECT_ENVIRONMENT	:= /goinfre/$(USER_NAME)/venvs/RAG

all: run

# Keep the venv on /goinfre: .venv is a symlink to it, so plain `uv` / the IDE
# never install into the (tiny) home partition.
venv:
	@mkdir -p $(UV_PROJECT_ENVIRONMENT)
	@if [ -d .venv ] && [ ! -L .venv ]; then \
		echo "Removing local .venv (home quota), using $(UV_PROJECT_ENVIRONMENT)"; \
		rm -rf .venv; \
	fi
	@ln -sfn $(UV_PROJECT_ENVIRONMENT) .venv

run: venv
	uv run python3 -m src

install: venv
	uv sync

debug: venv
	uv run -m pdb -m src

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache

lint: venv
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: venv
	uv run flake8 .
	uv run mypy . --strict


.PHONY: all venv run install debug clean lint lint-strict
