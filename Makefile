USER_NAME := $(shell whoami)

export UV_CACHE_DIR				:= /goinfre/$(USER_NAME)/.cache/uv
export UV_PROJECT_ENVIRONMENT	:= /goinfre/$(USER_NAME)/venvs/RAG

all: run

run:
	uv run python3 -m src

install:
	uv sync

debug:
	uv run -m pdb -m src

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict


.PHONY: all run install debug clean lint lint-strict
