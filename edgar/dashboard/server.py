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
    """Build a FastAPI app rooted at the given program_databases directories."""
    run_roots = [Path(r).resolve() for r in run_roots if Path(r).exists()]
    if default_run_dir is not None:
        # ensure the run's parent root is searchable
        default_run_dir = Path(default_run_dir).resolve()
        root = _root_for(default_run_dir)
        if root not in run_roots:
            run_roots.append(root)

    app = FastAPI(title="EDGAR Dashboard", version="0.1.0")

    def _resolve(run_id: str) -> Path:
        run_dir = dd.resolve_run_dir(run_id, run_roots)
        if run_dir is None:
            raise HTTPException(404, f"unknown run_id: {run_id}")
        return run_dir

    # ── meta ──

    @app.get("/api/health")
    def health():
        return {"ok": True, "roots": [str(r) for r in run_roots]}

    @app.get("/api/config")
    def config():
        return {
            "roots": [str(r) for r in run_roots],
            "default_run_id": dd._run_id(default_run_dir) if default_run_dir else None,
        }

    # ── runs ──

    @app.get("/api/runs")
    def runs():
        return dd.list_runs(run_roots)

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str):
        return dd.load_run_summary(_resolve(run_id))

    @app.get("/api/runs/{run_id}/state")
    def state(run_id: str):
        return dd.load_live_state(_resolve(run_id))

    @app.get("/api/runs/{run_id}/programs")
    def programs(run_id: str):
        return dd.load_program_list(_resolve(run_id))

    @app.get("/api/runs/{run_id}/programs/{idx}")
    def program_detail(run_id: str, idx: int):
        det = dd.load_program_detail(_resolve(run_id), idx)
        if det is None:
            raise HTTPException(404, f"unknown program idx: {idx}")
        return det

    @app.post("/api/runs/{run_id}/programs/{idx}/latex")
    async def program_latex(run_id: str, idx: int, force: bool = False):
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
        run_dir = _resolve(run_id)
        img_path = run_dir / "image_fits" / f"P{idx:04d}.png"
        if not img_path.exists():
            raise HTTPException(404, f"no fit image at {img_path}")
        return FileResponse(img_path, media_type="image/png")

    # ── static frontend ──

    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def root_index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def _root_for(run_dir: Path) -> Path:
    """Given a run dir program_databases/MM-DD/HH-MM-SS, return program_databases."""
    parts = run_dir.parts
    if len(parts) >= 2:
        return Path(*parts[:-2])
    return run_dir.parent
