import sys
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP
from edgar.evolution.population import Population
from edgar.evolution.program import Program
from edgar.io.status import read_status

# Ensure the root of the repository is in Python's search path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Initialize FastMCP Server
mcp = FastMCP("EDGAR Run Analyzer")

DEFAULT_RUNS_DIR = REPO_ROOT / "program_databases"


def _find_run_dir(run_identifier: str, base_dir: Optional[str] = None) -> Path:
    """Resolves a run identifier (folder name, subpath, or full path) to a run directory,
    looking inside the specified base_dir (which defaults to DEFAULT_RUNS_DIR).
    """
    path = Path(run_identifier)
    if path.is_absolute() and path.exists():
        return path

    target_base_dir = Path(base_dir) if base_dir else DEFAULT_RUNS_DIR

    # Check inside the target base directory
    direct_subpath = target_base_dir / run_identifier
    if direct_subpath.exists():
        return direct_subpath

    # Search recursively for a folder matching the pattern or suffix
    all_runs = sorted(target_base_dir.glob("**/*"), reverse=True)
    for r in all_runs:
        if r.is_dir() and run_identifier in str(r.relative_to(target_base_dir)):
            return r

    raise FileNotFoundError(
        f"Could not resolve run directory for identifier: '{run_identifier}' under '{target_base_dir}'"
    )


@mcp.tool()
def list_runs(base_dir: Optional[str] = None) -> str:
    """Lists all available EDGAR runs in the program databases directory (or arbitrary base_dir) along with their statuses."""
    target_dir = Path(base_dir) if base_dir else DEFAULT_RUNS_DIR
    if not target_dir.exists():
        return f"Base runs directory does not exist: {target_dir}"

    # Find all directories that contain a population.jsonl
    runs = []
    # Search recursively
    for path in sorted(target_dir.glob("**/population.jsonl"), reverse=True):
        run_folder = path.parent
        rel_path = run_folder.relative_to(target_dir)

        status_info = "Status Unknown"
        status_data = read_status(run_folder)
        if status_data:
            state = status_data.get("state", "unknown")
            gen = status_data.get("current_gen", 0)
            n_gens = status_data.get("n_gens", 0)
            status_info = f"State: {state} ({gen}/{n_gens} gens)"
            if status_data.get("error"):
                status_info += f" | Error: {status_data['error']}"

        runs.append(f"- **{rel_path}** ({status_info})")

    if not runs:
        return f"No completed or active EDGAR runs found under {target_dir}."

    return f"### Available EDGAR Runs under `{target_dir.name}`:\n" + "\n".join(runs)


@mcp.tool()
def get_run_specs(run_id: str, base_dir: Optional[str] = None) -> str:
    """Gets execution status, run location, and configuration specs from task_spec.yaml for a specific EDGAR run."""
    try:
        run_dir = _find_run_dir(run_id, base_dir=base_dir)
    except FileNotFoundError as e:
        return str(e)

    status_data = read_status(run_dir) or {}
    state = status_data.get("state", "unknown")
    current_gen = status_data.get("current_gen")
    n_gens = status_data.get("n_gens")

    # Load task_spec.yaml config info
    spec_path = run_dir / "task_spec.yaml"
    config_info = "No task_spec.yaml found."
    if spec_path.exists():
        try:
            import yaml

            with open(spec_path) as f:
                spec_doc = yaml.safe_load(f) or {}

            evo_cfg = spec_doc.get("evolution", {})
            llm_cfg = spec_doc.get("llms", {})
            scoring_cfg = spec_doc.get("scoring", {})

            config_info = (
                f"### Evolution Config:\n"
                f"- **Islands**: {evo_cfg.get('n_islands', 'N/A')}\n"
                f"- **Population Size**: {evo_cfg.get('population_size', 'N/A')}\n"
                f"- **Generations**: {evo_cfg.get('n_generations', 'N/A')}\n"
                f"- **Crossover Rate**: {evo_cfg.get('crossover_rate', 'N/A')}\n"
                f"\n### LLM Config:\n"
                f"- **Model LLM**: {llm_cfg.get('model', {}).get('model_name', 'N/A')}\n"
                f"- **Param Est LLM**: {llm_cfg.get('param_est', {}).get('model_name', 'N/A')}\n"
                f"\n### Scoring Config:\n"
                f"- **Metrics**: {scoring_cfg.get('metrics', 'N/A')}\n"
                f"- **Optimisation Method**: {scoring_cfg.get('opt_method', 'N/A')}\n"
            )
        except Exception as e:
            config_info = f"Failed to parse task_spec.yaml: {e}"

    summary = [
        f"## Run Specification & Config: `{run_id}`",
        f"- **Resolved Path**: `{run_dir}`",
        f"- **Execution Status**: {state.upper()}",
    ]
    if current_gen is not None and n_gens is not None:
        summary.append(f"- **Progress**: Generation {current_gen} / {n_gens}")
    if status_data.get("error"):
        summary.append(f"- **Error**: {status_data['error']}")

    summary.append(f"\n{config_info}")
    return "\n".join(summary)


@mcp.tool()
def get_top_models(run_id: str, limit: int = 5, base_dir: Optional[str] = None) -> str:
    """Retrieves metadata and numpy code of the top N best-performing models in a run."""
    try:
        run_dir = _find_run_dir(run_id, base_dir=base_dir)
    except FileNotFoundError as e:
        return str(e)

    pop_file = run_dir / "population.jsonl"
    if not pop_file.exists():
        return f"No population found in run directory: {run_dir}"

    try:
        pop = Population.load(str(pop_file))
    except Exception as e:
        return f"Failed to load population: {e}"

    # Sort the programs
    try:
        sorted_progs = pop.get_sorted()
    except Exception:
        # Fallback if final rank is not computed yet
        def sort_key(p: Program):
            loss = p.program_losses.discover.final
            return loss if isinstance(loss, (int, float)) else float("inf")

        sorted_progs = sorted(pop._programs, key=sort_key)

    output = [f"## Top {limit} Models for Run `{run_id}`:\n"]
    for i, prog in enumerate(sorted_progs[:limit]):
        loss_val = prog.program_losses.discover.final
        val_loss = prog.program_losses.validate.final

        output.append(
            f"### {i + 1}. Model '{prog.name}' (Population Index: {prog.idx})\n"
            f"- **Rank**: {prog.rank if prog.rank is not None else 'N/A'}\n"
            f"- **Lineage**: Gen {prog.birth.generation}, Island {prog.birth.island}\n"
            f"- **Loss (Discover)**: {loss_val}\n"
            f"- **Loss (Validate)**: {val_loss}\n"
            f"- **Numpy Model Code**:\n"
            f"```python\n{prog.code.model or '# No numpy code available'}\n```\n"
            f"{'=' * 50}\n"
        )

    return "\n".join(output)


@mcp.tool()
def inspect_model(run_id: str, index: int, base_dir: Optional[str] = None) -> str:
    """Inspects all available source codes and metadata for a specific model by its population index."""
    try:
        run_dir = _find_run_dir(run_id, base_dir=base_dir)
    except FileNotFoundError as e:
        return str(e)

    pop_file = run_dir / "population.jsonl"
    if not pop_file.exists():
        return f"No population found in run directory: {run_dir}"

    try:
        pop = Population.load(str(pop_file))
    except Exception as e:
        return f"Failed to load population: {e}"

    if index < 0 or index >= len(pop._programs):
        return (
            f"Invalid index {index}. Population contains {len(pop._programs)} programs."
        )

    prog = pop._programs[index]

    return (
        f"## Detailed Model Inspection: Model '{prog.name}' (Index: {prog.idx})\n"
        f"- **Lineage**: Gen {prog.birth.generation}, Island {prog.birth.island}, Mode: {prog.birth.mode}, LLM: {prog.birth.llm_name}\n"
        f"- **Discover Loss**: Initial={prog.program_losses.discover.init}, Final={prog.program_losses.discover.final}\n"
        f"- **Validate Loss**: Initial={prog.program_losses.validate.init}, Final={prog.program_losses.validate.final}\n"
        f"- **Number of Parameters**: {prog.n_params}\n"
        f"\n### 1. Model (JAX):\n```python\n{prog.code.model_jax}\n```\n"
        f"\n### 2. Model (Numpy):\n```python\n{prog.code.model}\n```\n"
        f"\n### 3. Parameter Estimator:\n```python\n{prog.code.param_est}\n```\n"
    )


if __name__ == "__main__":
    mcp.run()
