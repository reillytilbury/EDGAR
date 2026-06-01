"""Dashboard: live + post-hoc inspection of EDGAR runs.

The dashboard is a self-contained FastAPI server (server.py) that reads
artefacts written by edgar.run from a `<run_dir>/`. No shared-memory IPC.

Launch via:
    python -m edgar.cli dashboard [<run_dir>]
"""

from .data import (
    list_runs,
    load_run_summary,
    load_live_state,
    load_program_list,
    load_program_detail,
)

__all__ = [
    "list_runs",
    "load_run_summary",
    "load_live_state",
    "load_program_list",
    "load_program_detail",
]
