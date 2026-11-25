.PHONY: test lint train

test:
	python -m pytest

lint:
	ruff check alphachurn documind tests

train:
	python -m alphachurn.train
