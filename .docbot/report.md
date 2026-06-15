---COMMIT_TITLE---
Docs: Clarify `task_spec.py` git context; general docstring updates
---PR_BODY---
The EDGAR documentation bot has completed its latest update, incorporating new docstrings across several modules and refining existing descriptions.

### Key Changes in the Architectural/Mathematical Summary

The primary update to the global architectural summary (in `code_summary.md`) is a clarification within the `edgar/io/task_spec.py` module. The description of how git information (`git_sha`, `git_dirty`) is captured for full reproducibility now explicitly states: **"The git commands now explicitly use `REPO_ROOT` for context."** This adds more precise detail regarding the mechanism employed for ensuring consistent run reproducibility.

While `edgar/evolution/island.py` and the newly identified `edgar/scoring/hello.py` received updated docstrings within their respective source files, no corresponding new information or content changes were incorporated into their high-level descriptions within the `code_summary.md`. This indicates that the internal documentation for these files was enhanced, without altering their representation in the global architectural overview.

### Documented Files

The following files received new or updated docstrings during this documentation pass:

*   `edgar/evolution/island.py`
*   `edgar/io/task_spec.py`
*   `edgar/scoring/hello.py`