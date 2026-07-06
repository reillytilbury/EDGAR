.PHONY: commit-check
commit-check:
	@STAGED_FILES=$$(git diff --name-only --cached --diff-filter=d | grep '\.py$$' || true); \
	if [ -n "$$STAGED_FILES" ]; then \
		uvx ruff check --fix $$STAGED_FILES; \
		uvx ruff format $$STAGED_FILES; \
	else \
		echo "No staged python files to check."; \
	fi

