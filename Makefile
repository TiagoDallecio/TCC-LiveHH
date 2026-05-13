.PHONY: install test lint run

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	black poker_vision tests
	ruff check poker_vision tests
	mypy poker_vision

run:
	python -m poker_vision --help