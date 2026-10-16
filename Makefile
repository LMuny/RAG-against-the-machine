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
	rm -rf -R src/__pycache__ .mypy_cache

lint:
	uv run mypy -m src && uv run flake8 src


.PHONY: all run install debug clean lint
