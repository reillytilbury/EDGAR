#!/usr/bin/env python3
"""sync_agents.py - Sync agent instructions from external master markdown file
to Gemini CLI and Claude Code subagent definitions.
"""

import json
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
name: "edgar-analyzer"
description: "Analyzes EDGAR run outputs — lists runs, compares model code, and inspects numpy/JAX models and parameter estimators via the edgar_analyzer MCP server. Use when the user wants to study or compare the results of an EDGAR run."
color: "blue"
---
"""


def sync():
    # Resolve paths relative to repository root
    repo_root = Path(__file__).resolve().parents[3]
    instructions_path = (
        repo_root / "edgar-experimental/agents/output/edgar_analyzer_instructions.md"
    )
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
    # For Claude, replace Gemini/custom MCP tool prefix with Claude's double underscore format
    claude_instructions = instructions.replace(
        "mcp_edgar_analyzer_", "mcp__edgar_analyzer__"
    )
    claude_path.parent.mkdir(parents=True, exist_ok=True)
    claude_path.write_text(CLAUDE_FRONTMATTER + "\n" + claude_instructions + "\n")
    print(f"Generated Claude Subagent configuration at: {claude_path}")

    # Ensure .mcp.json is correctly configured
    mcp_path = repo_root / ".mcp.json"
    expected_config = {
        "command": "uv",
        "args": ["run", "python", "agents/output/tools/mcp_server.py"],
    }

    mcp_data = {}
    if mcp_path.exists():
        try:
            mcp_data = json.loads(mcp_path.read_text())
            if not isinstance(mcp_data, dict):
                mcp_data = {}
        except Exception as e:
            print(
                f"Warning: Failed to parse existing .mcp.json ({e}). Re-initializing."
            )
            mcp_data = {}

    if "mcpServers" not in mcp_data or not isinstance(mcp_data["mcpServers"], dict):
        mcp_data["mcpServers"] = {}

    current_server_config = mcp_data["mcpServers"].get("edgar_analyzer")
    if current_server_config != expected_config:
        mcp_data["mcpServers"]["edgar_analyzer"] = expected_config
        # Write back updated .mcp.json with pretty formatting
        mcp_path.write_text(json.dumps(mcp_data, indent=2) + "\n")
        print(f"Updated .mcp.json configuration at: {mcp_path}")
    else:
        print(".mcp.json is already configured correctly.")

    print("Subagents successfully synced from master instructions!")


if __name__ == "__main__":
    sync()
