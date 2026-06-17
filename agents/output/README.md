# EDGAR Output Run Analyzer Agent Setup

This folder contains a custom **Model Context Protocol (MCP)** server and a platform-agnostic subagent system designed to allow researchers to query, summarize, and retrieve NumPy model codes from EDGAR evolutionary runs using natural language.

---

## Folder Architecture

- **`tools/mcp_server.py`**: A lightweight Python FastMCP server defining the programmatic analysis tools. It interfaces directly with EDGAR's internal `Population`, `Program`, `read_status`, and `read_metrics` modules.
- **`edgar_analyzer_instructions.md`**: **The Single Source of Truth** containing the core persona and scientific analysis instructions for the agent.
- **`sync_agents.py`**: A synchronization script that loads the master instructions and generates CLI-specific subagent configuration files with their respective metadata and frontmatters.

---

## Setup Instructions

### 1. Synchronization Utility
Before setting up the individual platforms, make sure both platform-specific configuration files are freshly generated and updated by running:
```bash
uv run python agents/output/sync_agents.py
```
This automatically builds/updates:
1. **`.gemini/agents/edgar-analyzer.md`** (Gemini CLI)
2. **`.claude/agents/edgar-analyzer.md`** (Claude Code CLI)

---

### 2. Gemini CLI Subagent Setup (Zero Configuration)
The Gemini CLI subagent utilizes an **inline** MCP server definition in its generated frontmatter. 

No registration commands are required! Simply start your Gemini CLI session in the repository:
```bash
gemini
```
And ask questions using the `@` prefix:
```bash
@edgar-analyzer "list the available runs"
```

---

### 3. Claude Code Setup (Two-Step Configuration)
To make your custom agent and tools available inside Anthropic's **Claude Code** CLI:

#### Step A: Register the MCP Server
Run the following command in your terminal to register the python FastMCP server globally with Claude Code:
```bash
claude mcp add edgar-analyzer -- uv run --with mcp python /home/rajah/repos/EDGAR/agents/output/tools/mcp_server.py
```

#### Step B: Launch Claude Code
Start Claude Code in your terminal:
```bash
claude
```
The custom `.claude/agents/edgar-analyzer.md` subagent will automatically be loaded. You can delegate tasks to it in your session:
> **You:** "Ask the EDGAR Analyzer subagent to summarize the specs of the run `06-15/17-18-43` and show me the top model."

---

## Maintenance & Updating Instructions

If you need to change the system prompt, persona, or logical rules of the analyzer subagent:
1. Edit **`agents/output/edgar_analyzer_instructions.md`**.
2. Run the sync command:
   ```bash
   uv run python agents/output/sync_agents.py
   ```
Both the Gemini and Claude agent files will instantly update in perfect lockstep.
