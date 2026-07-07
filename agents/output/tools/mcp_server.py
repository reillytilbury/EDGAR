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
                f"- **Generations**: {evo_cfg.get('n_generations', 'N/A')}\n"
                f"- **Programs generated per island per generation**: {evo_cfg.get('batch_size', 'N/A')}\n"
                f"- **Critical island population size**: {evo_cfg.get('critical_population_size', 'N/A')}\n"
                f"- **Number of migrants per island per generation**: {evo_cfg.get('n_migrants', 'N/A')}\n"
                f"\n### LLM Config:\n"
                f"- **Model LLM**: {llm_cfg.get('model_llm', 'N/A')}\n"
                f"- **Param Est LLM**: {llm_cfg.get('param_est_llm', 'N/A')}\n"
                f"- **JAX translator LLM**: {llm_cfg.get('jax_model_translator_llm', 'N/A')}\n"
                f"\n### Scoring Config:\n"
                f"- **Parameter penalty**: {scoring_cfg.get('param_penalty_weight', 'N/A')}\n"
                f"- **Gradient descent params**: {scoring_cfg.get('gradient_descent', {})}\n"
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


@mcp.tool()
def filter_models_by_parameters(
    run_id: str, n_params: int, base_dir: Optional[str] = None
) -> str:
    """Filters the programs of an EDGAR run by their exact number of parameters.

    Args:
        run_id: The identifier of the run folder.
        n_params: The exact number of parameters to filter by.
        base_dir: Optional custom base directory path for runs.

    Returns:
        A formatted string listing the population indices, names, lineages,
        and losses of matching models.
    """
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

    matching = []
    for idx, prog in enumerate(pop._programs):
        if prog.n_params == n_params:
            matching.append((idx, prog))

    if not matching:
        return f"No models found with exactly {n_params} parameters in run `{run_id}`."

    output = [
        f"### Found {len(matching)} models with exactly {n_params} parameters in run `{run_id}`:\n"
    ]
    for idx, prog in matching:
        disc_loss = prog.program_losses.discover.final
        val_loss = prog.program_losses.validate.final
        name = prog.name or "Unnamed Model"
        output.append(
            f"- **Population Index**: `{idx}`\n"
            f"  - **Name**: {name}\n"
            f"  - **Lineage**: Gen {prog.birth.generation}, Island {prog.birth.island}\n"
            f"  - **Loss**: Discover={disc_loss} | Validate={val_loss}"
        )

    return "\n".join(output)


@mcp.tool()
def compare_model_syntax_trees(
    run_id: str,
    n_params: Optional[int] = None,
    indices: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> str:
    """Compares the Python abstract syntax trees of selected programs to find unique mathematical implementations.

    It strips docstrings, comments, and formatting to group programs that are
    mathematically/structurally identical. You can filter by parameter count
    or provide a comma-separated list of population indices.

    Args:
        run_id: The identifier of the run folder.
        n_params: Optional parameter count to filter and compare.
        indices: Optional comma-separated string of population indices to compare (e.g., "2,5,10").
        base_dir: Optional custom base directory path for runs.

    Returns:
        A structured string report detailing the unique variations, their frequency,
        indices, and the canonical code for each variation.
    """
    import ast
    import collections

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

    programs_to_compare = []

    # 1. Resolve which programs to compare
    if indices is not None:
        try:
            target_indices = [int(x.strip()) for x in indices.split(",") if x.strip()]
        except ValueError:
            return "Invalid format for indices. Please provide a comma-separated list of integers (e.g., '2,5,10')."

        for idx in target_indices:
            if idx < 0 or idx >= len(pop._programs):
                return f"Index {idx} is out of bounds for population of size {len(pop._programs)}."
            programs_to_compare.append((idx, pop._programs[idx]))
    elif n_params is not None:
        for idx, prog in enumerate(pop._programs):
            if prog.n_params == n_params:
                programs_to_compare.append((idx, prog))
    else:
        return "Please specify either 'n_params' or 'indices' to compare."

    if not programs_to_compare:
        return "No programs found matching the comparison criteria."

    # 2. Canonical AST normalization helper
    def clean_code(code_str: str) -> str:
        try:
            tree = ast.parse(code_str)
            # Remove docstrings from functions, classes, and module
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        node.body.pop(0)
            return ast.unparse(tree).strip()
        except Exception:
            # Fallback to simple cleaning if AST parsing fails
            lines = [line.strip() for line in code_str.strip().splitlines()]
            return "\n".join([l for l in lines if l and not l.startswith("#")])

    # 3. Group by canonical AST string
    groups = collections.defaultdict(list)
    for idx, prog in programs_to_compare:
        norm_code = clean_code(prog.model_code)
        groups[norm_code].append((idx, prog))

    # Helper to find the chronologically earliest member of a group in the evolutionary algorithm
    def get_earliest_member(members_list):
        return min(
            members_list,
            key=lambda x: (
                x[1].birth.generation if x[1].birth.generation is not None else 9999,
                x[0],
            ),
        )

    # 4. Generate report
    total_progs = len(programs_to_compare)
    output = [
        f"## AST Syntax Tree Comparison Report for Run `{run_id}`",
        f"- Compared a total of **{total_progs}** programs.",
        f"- Found **{len(groups)}** unique mathematical code variations.\n",
        f"{'=' * 80}\n",
    ]

    # Sort groups by count descending
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for g_idx, (canonical_code, members) in enumerate(sorted_groups):
        earliest_idx, earliest_prog = get_earliest_member(members)
        member_indices = [idx for idx, _ in members]
        output.append(
            f"### VARIATION {g_idx + 1}:\n"
            f"- **Occurrence Count**: {len(members)} ({(len(members) / total_progs) * 100:.1f}% of compared)\n"
            f"- **Earliest Discovery**: Population Index `{earliest_idx}` (Gen {earliest_prog.birth.generation}, Island {earliest_prog.birth.island})\n"
            f"- **Associated Indices**: {member_indices}\n"
            f"- **Representative Model Name**: {earliest_prog.name or 'Unnamed'}\n"
            f"- **Canonical Code (AST Normalized)**:\n"
            f"```python\n{canonical_code}\n```\n"
            f"{'-' * 80}\n"
        )

    return "\n".join(output)


if __name__ == "__main__":
    mcp.run()
