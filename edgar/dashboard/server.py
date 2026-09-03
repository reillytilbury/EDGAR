"""FastAPI server for the EDGAR dashboard.

Serves a single static SPA at `/` and a JSON API under `/api/`. Reads from
disk only — no shared-memory IPC with the runner. Run it via:

    python -m edgar.cli dashboard [<run_dir>] [--port PORT] [--no-open]
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data as dd
from .latex_cache import get_or_generate_latex


HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"


def build_app(run_roots: list[Path], default_run_dir: Path | None = None) -> FastAPI:
    """Build a FastAPI app rooted at the given program_databases directories.

    Initializes the FastAPI application, configures the run roots for discovering
    EDGAR experiments, and registers all API endpoints and static file serving.
    It ensures that the `default_run_dir`'s parent root is included in the
    searchable `run_roots`.

    Args:
        run_roots: A list of base directories where EDGAR run data (e.g.,
            'program_databases/') can be found. These directories are scanned
            to discover available EDGAR experiments.
        default_run_dir: An optional default run directory to be loaded
            automatically when the dashboard starts. Its root will be added to
            `run_roots` if not already present, ensuring it is discoverable.

    Returns:
        A configured FastAPI application instance ready to serve the EDGAR
        dashboard.
    """
    run_roots = [Path(r).resolve() for r in run_roots if Path(r).exists()]
    if default_run_dir is not None:
        # ensure the run's parent root is searchable
        default_run_dir = Path(default_run_dir).resolve()
        root = _root_for(default_run_dir)
        if root not in run_roots:
            run_roots.append(root)

    app = FastAPI(title="EDGAR Dashboard", version="0.1.0")

    def _resolve(run_id: str) -> Path:
        """Resolves a run_id to its corresponding on-disk directory.

        This helper function maps a unique run identifier (e.g., a timestamped
        directory name) to its absolute file system path within the configured
        `run_roots`.

        Args:
            run_id: The unique identifier for an EDGAR run (e.g., '2023-10-27/15-30-00').

        Returns:
            The `Path` object pointing to the resolved run directory.

        Raises:
            HTTPException: If the `run_id` does not correspond to an existing
                run directory within the configured `run_roots`, a 404 Not Found
                error is raised.
        """
        run_dir = dd.resolve_run_dir(run_id, run_roots)
        if run_dir is None:
            raise HTTPException(404, f"unknown run_id: {run_id}")
        return run_dir

    # ── meta ──

    @app.get("/api/health")
    def health():
        """Returns a simple health check status of the dashboard server.

        This endpoint can be used to verify that the server is running and
        identify the configured `run_roots` that the dashboard is monitoring.

        Returns:
            A dictionary indicating the server's operational status and a list
            of the absolute paths to its configured root directories.
        """
        return {"ok": True, "roots": [str(r) for r in run_roots]}

    @app.get("/api/config")
    def config():
        """Returns the dashboard's current configuration.

        This includes the list of `run_roots` that are being monitored and the
        `default_run_id` if one was specified at startup, which is used to
        automatically load a specific run.

        Returns:
            A dictionary containing the dashboard's configuration details,
            including `roots` (list of paths) and `default_run_id` (string or None).
        """
        return {
            "roots": [str(r) for r in run_roots],
            "default_run_id": dd._run_id(default_run_dir) if default_run_dir else None,
        }

    # ── runs ──

    @app.get("/api/runs")
    def runs():
        """Returns a list of all discoverable EDGAR runs.

        The runs are identified by scanning the configured `run_roots` for
        directories that contain a `task_spec.yaml` file, which signifies
        an EDGAR experiment.

        Returns:
            A list of dictionaries, where each dictionary represents an EDGAR
            run with its ID and potentially other summary information suitable
            for display in a list view.
        """
        return dd.list_runs(run_roots)

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str):
        """Returns a summary of a specific EDGAR run.

        This includes high-level information such as the experiment's start time,
        the total number of programs generated, the best overall loss achieved,
        Git status at the time of the run, and the LLM configuration used.
        It is primarily for overview cards in the dashboard.

        Args:
            run_id: The unique identifier for the EDGAR run.

        Returns:
            A dictionary containing the summary details for the specified run.

        Raises:
            HTTPException: If the `run_id` is not found within the configured
                `run_roots`, a 404 Not Found error is raised.
        """
        return dd.load_run_summary(_resolve(run_id))

    @app.get("/api/runs/{run_id}/state")
    def state(run_id: str):
        """Returns the live state of a specific EDGAR run.

        This endpoint provides real-time metrics and progress indicators, such as
        the current generation number, information about the best programs found
        so far, and success rates for LLM generations per island. It is designed
        for live monitoring and updates in the dashboard.

        Args:
            run_id: The unique identifier for the EDGAR run.

        Returns:
            A dictionary containing the live state information for the specified run.

        Raises:
            HTTPException: If the `run_id` is not found within the configured
                `run_roots`, a 404 Not Found error is raised.
        """
        return dd.load_live_state(_resolve(run_id))

    @app.get("/api/runs/{run_id}/programs")
    def programs(run_id: str):
        """Returns a list of all programs generated in a specific EDGAR run.

        The programs are returned as a sorted list, primarily by their final
        rank, then by their validation loss, and finally by their global index.
        This provides an ordered view of the evolutionary progress and outcomes.

        Args:
            run_id: The unique identifier for the EDGAR run.

        Returns:
            A list of dictionaries, where each dictionary represents a program
            with its key attributes such as index, name, and losses.

        Raises:
            HTTPException: If the `run_id` is not found within the configured
                `run_roots`, a 404 Not Found error is raised.
        """
        return dd.load_program_list(_resolve(run_id))

    @app.get("/api/runs/{run_id}/family_tree")
    def family_tree(run_id: str):
        """Returns data for the family tree (lineage) visualization.

        This endpoint provides the necessary data to render a lineage graph,
        including hierarchical layout positions for programs, parent-child
        relationships (edges), and metadata for each program node, which can
        be used for visualizing the evolutionary history of programs.

        Args:
            run_id: The unique identifier for the EDGAR run.

        Returns:
            A dictionary containing Plotly-compatible trace data and a parent
            map suitable for lineage visualization.

        Raises:
            HTTPException: If the `run_id` is not found within the configured
                `run_roots`, a 404 Not Found error is raised.
        """
        return dd.load_family_tree_data(_resolve(run_id))

    @app.get("/api/runs/{run_id}/programs/{idx}")
    def program_detail(run_id: str, idx: int):
        """Returns detailed information for a specific program within an EDGAR run.

        This includes comprehensive details such as the program's source code,
        various loss values, optimized parameters, its `BirthCertificate`
        (origin and lineage), and paths to any associated visualization images.

        Args:
            run_id: The unique identifier for the EDGAR run.
            idx: The global unique index of the program within the run's population.

        Returns:
            A dictionary containing detailed information for the specified program.

        Raises:
            HTTPException: If the `run_id` or `idx` is not found within the
                configured `run_roots` or population, a 404 Not Found error is raised.
        """
        det = dd.load_program_detail(_resolve(run_id), idx)
        if det is None:
            raise HTTPException(404, f"unknown program idx: {idx}")
        return det

    @app.post("/api/runs/{run_id}/programs/{idx}/latex")
    async def program_latex(run_id: str, idx: int, force: bool = False):
        """Generates or retrieves the LaTeX mathematical representation of a program's model.

        This function leverages an LLM (specifically the `jax_model_translator_llm`
        configured in the `task_spec.yaml`) to translate the program's JAX-compatible
        model source code into a mathematical equation in LaTeX format. The result
        is aggressively cached on disk to ensure fast, free retrieval for subsequent
        requests. It can optionally force regeneration even if a cached version exists.

        Args:
            run_id: The unique identifier for the EDGAR run.
            idx: The global unique index of the program.
            force: If True, forces the regeneration of the LaTeX equation,
                bypassing the cache and re-querying the LLM.

        Returns:
            A dictionary containing the generated or retrieved LaTeX string.

        Raises:
            HTTPException:
                - 404: If the `run_id` or `idx` is not found within the configured
                  `run_roots` or population.
                - 502: If there's an issue with the LLM API (e.g., missing
                  API key, quota limits, or unexpected model behavior) during
                  LaTeX generation, providing a clean error message.
        """
        run_dir = _resolve(run_id)
        det = dd.load_program_detail(run_dir, idx)
        if det is None:
            raise HTTPException(404, f"unknown program idx: {idx}")
        try:
            result = await get_or_generate_latex(run_dir, idx, det, force=force)
        except RuntimeError as e:
            # LLM key missing / quota etc — surface a 502 with a clean message
            raise HTTPException(502, str(e))
        return result

    # ── images ──

    @app.get("/api/runs/{run_id}/image/gen_{gen}/island_{island}/batch_{batch}")
    def image(run_id: str, gen: str, island: str, batch: str):
        """Serves LLM image feedback plots generated during the evolutionary process.

        These images are typically provided to LLMs as multimodal input to guide the
        generation of new programs, often showing model predictions against data.
        The path components (`gen`, `island`, `batch`) are expected to be integers,
        allowing for zero-padded (e.g., `gen_001`) or non-padded formats. The images
        are stored in a structured directory: `image_feedback/gen_NNN/island_NNN/batch_NNN/image.png`.

        Args:
            run_id: The unique identifier for the EDGAR run.
            gen: The generation number, as a string (e.g., '1' or '001').
            island: The island index, as a string (e.g., '0' or '000').
            batch: The batch index within the island and generation, as a string.

        Returns:
            A `FileResponse` containing the requested image with `media_type="image/png"`.

        Raises:
            HTTPException:
                - 404: If the `run_id` is not found or the specified image path does not exist.
                - 400: If `gen`, `island`, or `batch` are not valid integers, indicating a malformed request.
        """
        run_dir = _resolve(run_id)
        # path components are zero-padded ints; accept either "0" or "000"
        try:
            g, isl, b = int(gen), int(island), int(batch)
        except ValueError:
            raise HTTPException(400, "gen/island/batch must be integers")
        img_path = (
            run_dir
            / "image_feedback"
            / f"gen_{g:03d}"
            / f"island_{isl:03d}"
            / f"batch_{b:03d}"
            / "image.png"
        )
        if not img_path.exists():
            raise HTTPException(404, f"no image at {img_path}")
        return FileResponse(img_path, media_type="image/png")

    @app.get("/api/runs/{run_id}/fit_image/{idx}")
    def fit_image(run_id: str, idx: int):
        """Serves model fit visualization images for individual programs.

        These images typically show the initial and optimized model fits against
        the experimental data, providing a visual assessment of the program's performance
        and how well it captures the underlying patterns. The images are stored at
        `image_fits/P{idx:04d}.png`.

        Args:
            run_id: The unique identifier for the EDGAR run.
            idx: The global unique index of the program.

        Returns:
            A `FileResponse` containing the requested image with `media_type="image/png"`.

        Raises:
            HTTPException: If the `run_id` is not found or the specified fit image path does not exist.
        """
        run_dir = _resolve(run_id)
        img_path = run_dir / "image_fits" / f"P{idx:04d}.png"
        if not img_path.exists():
            raise HTTPException(404, f"no fit image at {img_path}")
        return FileResponse(img_path, media_type="image/png")

    @app.get("/api/runs/{run_id}/trajectory_image/{idx}")
    def trajectory_image(run_id: str, idx: int):
        """Serves optimization trajectory visualization images for individual programs.

        These images show the loss history of all parallel estimators during the
        gradient descent optimization process, highlighting how different
        initial parameter sets converge (or fail to converge) and indicating the
        best-performing trajectory. The images are stored at `image_trajectories/P{idx:04d}_traj.png`.

        Args:
            run_id: The unique identifier for the EDGAR run.
            idx: The global unique index of the program.

        Returns:
            A `FileResponse` containing the requested image with `media_type="image/png"`.

        Raises:
            HTTPException: If the `run_id` is not found or the specified trajectory image path does not exist.
        """
        run_dir = _resolve(run_id)
        img_path = run_dir / "image_trajectories" / f"P{idx:04d}_traj.png"
        if not img_path.exists():
            raise HTTPException(404, f"no trajectory image at {img_path}")
        return FileResponse(img_path, media_type="image/png")

    # ── static frontend ──

    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def root_index():
        """Serves the main `index.html` file for the Single Page Application (SPA) frontend.

        This is the entry point for the EDGAR dashboard's user interface,
        loading the core application logic and presentation.

        Returns:
            A `FileResponse` containing the `index.html` file located in the
            `static` directory.
        """
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def _root_for(run_dir: Path) -> Path:
    """Given a run directory path, return its corresponding 'program_databases' root.

    This helper function is used to determine the top-level directory where
    multiple EDGAR runs are organized. It assumes a hierarchical structure for
    run directories, typically `<root>/<task_name>/YYYY-MM-DD/HH-MM-SS/`.

    For example, given `program_databases/my_task/2023-01-01/12-34-56`, it
    would return `program_databases/my_task`. If the path structure does not
    match the expected pattern, it attempts to return a reasonable parent
    directory as a fallback.

    Args:
        run_dir: The `Path` object of an EDGAR run directory
            (e.g., `program_databases/my_task/2023-01-01/12-34-56`).

    Returns:
        The `Path` object representing the root directory (e.g.,
        `program_databases/my_task`). If the path structure does not match the expected
        pattern, it returns the parent directory of `run_dir` or `run_dir` itself
        as a fallback.
    """
    parts = run_dir.parts
    # Expected structure: <root>/<task_name>/YYYY-MM-DD/HH-MM-SS/
    # So, we expect at least 4 parts from the conceptual root, meaning
    # we need to go up 3 levels to reach <root>/<task_name>
    if len(parts) >= 3: # Corrected from 2 to 3 to get `<root>/<task_name>`
        # This will return `program_databases/my_task` for the example
        return Path(*parts[:-3])
    return run_dir.parent

```
