.PHONY: commit-check
commit-check:
	uvx ruff check --fix .
	uvx ruff format .
