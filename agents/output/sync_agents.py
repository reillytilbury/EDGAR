#!/usr/bin/env python3
"""sync_agents.py - Sync agent instructions from external master markdown file
to Gemini CLI and Claude Code subagent definitions.
"""

import sys
from pathlib import Path

# Gemini frontmatter
GEMINI_FRONTMATTER = """---
name: edgar-analyzer
description: Explores EDGAR run directories, retrieves run specifications, lists available runs, and inspects numpy/JAX model codes.
tools:
  - mcp_edgar_analyzer_list_runs
  - mcp_edgar_analyzer_get_run_specs
  - mcp_edgar_analyzer_get_top_models
  - mcp_edgar_analyzer_inspect_model
mcp_servers:
  edgar_analyzer:
    command: "uv"
    args: [
      "run",
      "--with",
      "mcp",
      "python",
      "agents/output/tools/mcp_server.py"
    ]
model: gemini-2.5-flash
temperature: 0.1
max_turns: 15
---
"""

# Claude frontmatter
CLAUDE_FRONTMATTER = """---
name: "EDGAR Analyzer 📊"
color: "blue"
---
"""


def sync():
    # Resolve paths relative to repository root
    repo_root = Path(__file__).resolve().parents[2]
    instructions_path = repo_root / "agents/output/edgar_analyzer_instructions.md"
    gemini_path = repo_root / ".gemini/agents/edgar-analyzer.md"
    claude_path = repo_root / ".claude/agents/edgar-analyzer.md"

    if not instructions_path.exists():
        print(f"Error: Could not find master instructions at {instructions_path}")
        sys.exit(1)

    # Read instructions
    instructions = instructions_path.read_text().strip()

    # Generate Gemini subagent file
    gemini_path.parent.mkdir(parents=True, exist_ok=True)
    gemini_path.write_text(GEMINI_FRONTMATTER + "\n" + instructions + "\n")
    print(f"Generated Gemini Subagent configuration at: {gemini_path}")

    # Generate Claude subagent file
    claude_path.parent.mkdir(parents=True, exist_ok=True)
    claude_path.write_text(CLAUDE_FRONTMATTER + "\n" + instructions + "\n")
    print(f"Generated Claude Subagent configuration at: {claude_path}")

    print("Subagents successfully synced from master instructions!")


if __name__ == "__main__":
    sync()
