PYTHON ?= python

.PHONY: install lint test train score

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) -m alphachurn.cli train

score:
	$(PYTHON) -m alphachurn.cli score
