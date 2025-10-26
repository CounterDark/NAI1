SHELL := /bin/sh

.PHONY: setup run clean

setup:
	uv sync

run:
	uv run python src/main.py $(ARGS)

clean:
	rm -rf .venv
	rm -rf uv.lock
