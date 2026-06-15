import subprocess
from typing import List, Optional


def run_git_command(args: List[str]) -> str:
    """Run a git command and return its output."""
    result = subprocess.run(["git"] + args, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_last_doc_commit() -> Optional[str]:
    """Find the last commit that modified .docbot/code_summary.md."""
    try:
        # Get the commit hash of the last commit that modified code_summary.md
        # If the file is empty or never committed, this might fail or return nothing.
        commit = run_git_command(
            ["log", "-1", "--format=%H", "--", ".docbot/code_summary.md"]
        )
        return commit if commit else None
    except subprocess.CalledProcessError:
        return None


def get_diff(since_commit: str) -> str:
    """Return the git diff for the edgar/ directory since since_commit."""
    return run_git_command(["diff", since_commit, "HEAD", "--", "edgar/"])


def get_changed_files(since_commit: str) -> List[str]:
    """Return a list of changed .py files in edgar/ since since_commit."""
    output = run_git_command(
        ["diff", "--name-only", since_commit, "HEAD", "--", "edgar/"]
    )
    files = output.splitlines()
    return [f for f in files if f.endswith(".py")]
