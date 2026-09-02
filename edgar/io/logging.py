"""
src/io/logging.py

Human-readable run logging for EDGAR experiments.

Creates a single run.log file. The runner emits a streaming generation banner
+ per-stage start/end lines (via ``edgar.io.metrics.stage_timer``) and one
summary block at the END of each generation via ``log_generation``.

Verbosity is controlled by the level argument:

  compact  — streaming progress + end-of-gen summary
  code     — compact + generated code for each program born this generation
  prompts  — code + reconstructed LLM prompts and image paths

Prompts are reconstructed post-hoc from program birth metadata and the spec's
prompt schemas.

Warnings emitted via warnings.warn() during a generation are buffered and
appended to the end of that generation's block in the log.
"""

from __future__ import annotations

import os
import time
import warnings
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evolution.population import Population
    from ..io.task_spec import TaskSpec
    from ..io.metrics import RunMetrics


LEVELS = ("compact", "code", "prompts")


def print_and_log(log: RunLog, message: str) -> None:
    """Prints a message to the console and appends it to the run log file.

    Ensures that important messages are visible in real-time and
    persisted in the run's log file for later review.

    Args:
        log: The `RunLog` object managing the log file.
        message: The string message to print and log.
    """
    print(message, flush=True)
    log.file.write(message + "\n")
    log.file.flush()


def _llm_display(v: Any) -> str:
    """Coerce an LLM field (string name OR a pydantic-ai Model instance) into
    a short display string. Falls back to the type name for opaque objects."""
    if isinstance(v, str):
        return v
    name = getattr(v, "model_name", None)
    if name:
        return str(name)
    return type(v).__name__


def gen_banner(
    log: RunLog,
    gen: int,
    n_gens: int,
    mode: str,
    temperature: float,
    llms: Any,
    n_spawn: int,
) -> None:
    """Banner written at the START of each generation. Includes the schedule
    so the user can correlate behaviour shifts (explore→exploit, temp decay,
    LLM rotation) with what they see in the log.
    """
    model_llm = (
        llms.model[gen % len(llms.model)]
        if isinstance(llms.model, list)
        else llms.model
    )
    print_and_log(log, "")
    print_and_log(
        log,
        f"┌── Generation {gen}/{n_gens - 1}  "
        f"mode={mode}  temp={temperature:.3f}  "
        f"spawn={n_spawn}  "
        f"llms[model={_llm_display(model_llm)} "
        f"param_est={_llm_display(llms.param_est)} "
        f"jax={_llm_display(llms.model_jax)}]",
    )


@dataclass
class RunLog:
    """A dataclass to hold the state and file handle for the EDGAR run log.

    Attributes:
        file: A file-like object (TextIO) opened for writing the log.
        level: The verbosity level of the log ("compact", "code", or "prompts").
        start_time: The monotonic time when the log was opened, used for
            calculating total elapsed time.
        previous_gen_time: The monotonic time at the end of the previous
            generation, used to calculate generation-specific elapsed time.
        warnings_buffer: A list of buffered warning messages to be flushed
            at the end of each generation.
        prev_showwarning: Stores the original `warnings.showwarning` hook
            to restore it when the log is closed.
    """

    file: TextIO
    level: str
    start_time: float
    previous_gen_time: float = 0.0
    warnings_buffer: list[str] = field(default_factory=list)
    prev_showwarning: Any = None


def open_log(output_dir: str, level: str = "compact", append: bool = False) -> RunLog:
    """Creates `run.log` in the specified output directory and returns a RunLog handle.

    This function initializes the logging system for an EDGAR run. It also
    installs a custom `warnings.showwarning` hook that buffers any warnings
    emitted during a generation. These buffered warnings are then appended
    to the end of the current generation's log block, providing contextualized
    warning messages. The original warning hook is stored so it can be
    restored by `close_log()`.

    Args:
        output_dir: The run output directory (e.g., `spec.output_dir`) where `run.log` will be created.
        level: The verbosity level for the log, must be one of "compact", "code", or "prompts".
        append: If True, open run.log in append mode and write a "RESUMED"
            banner instead of the fresh-run header. Used by `edgar resume`.

    Returns:
        A `RunLog` object containing the log file handle and state.

    Raises:
        ValueError: If `level` is not one of the allowed verbosity levels.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, "run.log"), "a" if append else "w")
    if append:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"\n{'=' * 60}\n──── RESUMED at {ts}  |  level={level} ────\n{'=' * 60}\n\n"
        )
    else:
        f.write(f"EDGAR run log  |  level={level}\n{'=' * 60}\n\n")
    f.flush()
    log = RunLog(file=f, level=level, start_time=time.monotonic())

    original: Callable = warnings.showwarning

    def _hook(message, category, filename, lineno, file=None, line=None):
        original(message, category, filename, lineno, file, line)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        log.warnings_buffer.append(f"  [{ts}] {category.__name__}: {message}\n")

    log.prev_showwarning = original
    warnings.showwarning = _hook
    return log


def close_log(log: RunLog) -> None:
    """Flushes any remaining buffered warnings, closes the log file, and restores the original warnings hook.

    This function should be called at the end of an EDGAR run to ensure
    all log messages and warnings are persisted and system state is cleaned up.

    Args:
        log: The `RunLog` object to close.
    """
    _flush_warnings(log)
    log.file.close()
    if log.prev_showwarning is not None:
        warnings.showwarning = log.prev_showwarning


def _flush_warnings(log: RunLog) -> None:
    """Writes all buffered warning messages to the log file and clears the buffer.

    Args:
        log: The `RunLog` object containing the warnings buffer.
    """
    if log.warnings_buffer:
        log.file.write("  --- Warnings ---\n")
        log.file.writelines(log.warnings_buffer)
        log.warnings_buffer.clear()
        log.file.flush()


def log_generation(
    log: RunLog,
    gen: int,
    population: Population,
    islands: list[set[int]],
    spec: TaskSpec,
    metrics: "RunMetrics | None" = None,
) -> None:
    """Appends a summary block for one generation to the run log file.

    This function compiles and writes a comprehensive summary of the current
    evolutionary generation to the `run.log` file. All statistics and details
    are dynamically derived from the `population`, `islands`, and `spec`
    objects, eliminating the need for intermediate state capture.

    The verbosity of the logged information depends on `log.level`:
    *   **"compact"**: Logs generation index, mode, temperature, elapsed times,
        LLM names, program spawning success rates, and the global best discover
        loss, along with the best program on each island.
    *   **"code"**: Includes all "compact" information, plus the generated
        `model`, `parameter_estimator`, and JAX `model_jax` code for all
        programs born in this generation.
    *   **"prompts"**: Includes all "code" information, plus the reconstructed
        LLM prompts (for model, parameter estimator, and JAX translation)
        used to generate each new program, along with paths to any associated
        feedback images.

    If ``metrics`` is provided, also surfaces per-stage timing, per-role LLM
    token totals + retry counts, and scoring outcome breakdown from the
    active accumulator's bucket for this generation.

    Args:
        log: The `RunLog` handle obtained from `open_log()`.
        gen: The current generation index (0-based).
        population: The current `Population` object containing all evolved programs.
        islands: A list of sets, where each set contains the indices of programs
            currently residing on a specific island (after pruning and deduplication).
        spec: The `TaskSpec` object containing global configuration and callables for the run.
        metrics: optional RunMetrics — used to surface live per-gen stats.
    """
    f = log.file
    born = [
        population[i]
        for i in range(len(population))
        if population[i].birth.generation == gen
    ]
    n = len(born)

    def pct(k):
        return f"{100 * k / n:.0f}%" if n else "n/a"

    n_model = sum(1 for p in born if p.code.model is not None)
    n_param_ests = spec.flat_config.get("n_param_ests")
    total_param_ests_expected = n * n_param_ests
    total_param_ests_generated = sum(
        len(p.code.param_est) for p in born if isinstance(p.code.param_est, list)
    )
    n_jax = sum(1 for p in born if p.code.model_jax is not None)
    n_scored = sum(
        1 for p in born if p.program_losses.discover.final not in (None, float("inf"))
    )

    elapsed = time.monotonic() - log.start_time
    this_gen_time = elapsed - log.previous_gen_time
    log.previous_gen_time = elapsed

    def pct_gen(k, expected):
        return f"{100 * k / expected:.0f}%" if expected else "n/a"

    f.write(f"└── Gen {gen:3d} summary  ({this_gen_time:.1f}s, total {elapsed:.1f}s)\n")
    f.write(
        f"    Spawn   {n}  |  model={pct(n_model)}  param_est={pct_gen(total_param_ests_generated, total_param_ests_expected)}  jax={pct(n_jax)}  scored={pct(n_scored)}\n"
    )

    if metrics is not None:
        row = metrics._build_gen_row()
        stage_times = row.get("stage_times") or {}
        if stage_times:
            parts = [f"{name}={t:.1f}s" for name, t in stage_times.items()]
            f.write(f"    Stages  {'  '.join(parts)}\n")
        llm_calls = row.get("llm_calls") or {}
        if llm_calls:
            for role, st in llm_calls.items():
                lat = st.get("latency_ms") or {}
                p50 = lat.get("p50")
                p90 = lat.get("p90")
                p50_s = f"{p50 / 1000:.1f}s" if p50 is not None else "-"
                p90_s = f"{p90 / 1000:.1f}s" if p90 is not None else "-"
                f.write(
                    f"    LLM[{role:<9}] n={st['n']} ok={st['ok']} retried={st['retried']}  "
                    f"tokens in={st['in_tokens_total']} out={st['out_tokens_total']}  "
                    f"latency p50={p50_s} p90={p90_s}\n"
                )
        sc = row.get("scoring") or {}
        if sc.get("n"):
            lat = sc.get("latency_ms") or {}
            p50 = lat.get("p50")
            p90 = lat.get("p90")
            mx = lat.get("max")
            p50_s = f"{p50 / 1000:.1f}s" if p50 is not None else "-"
            p90_s = f"{p90 / 1000:.1f}s" if p90 is not None else "-"
            mx_s = f"{mx / 1000:.1f}s" if mx is not None else "-"
            f.write(
                f"    Scoring n={sc['n']} ok={sc['ok']} timeout={sc['timeout']} inf={sc['inf']}  "
                f"latency p50={p50_s} p90={p90_s} max={mx_s}\n"
            )

    f.write("    Best per island:\n")
    for idx, island in enumerate(islands):
        progs = [population[i] for i in island]
        best = min(progs, key=lambda p: p.program_losses.discover.final or float("inf"))
        loss_val = best.program_losses.discover.final
        loss_str = f"{loss_val:.6f}" if isinstance(loss_val, float) else str(loss_val)
        f.write(
            f"      Island {idx}  size={len(island)}  best=#{best.idx} {best.name!r}  loss={loss_str}\n"
        )
    f.write("\n")

    if log.level not in ("code", "prompts"):
        _flush_warnings(log)
        f.flush()
        return

    f.write("Newly-generated programs:\n")
    for p in born:
        f.write(f"  --- Program #{p.idx} (island={p.birth.island}) ---\n")
        f.write(f"  [model]\n{p.code.model or '(none)'}\n")
        f.write(f"  [param_est]\n{p.param_est_code or '(none)'}\n")
        f.write(f"  [model_jax]\n{p.code.model_jax or '(none)'}\n\n")

    if log.level != "prompts":
        _flush_warnings(log)
        f.flush()
        return

    for p in born:
        parents = [population[i] for i in p.birth.parent_indices]
        mode_p = p.birth.mode or "explore"
        f.write(f"  --- Prompts for Program #{p.idx} ---\n")
        model_cfg = {
            **spec.flat_config,
            "ideas-injection-point": "\n".join(getattr(p.birth, "ideas", []) or []),
        }
        f.write(
            f"  [model prompt]\n{spec.model_prompt_schema.build_prompt(mode_p, parents, model_cfg)}\n\n"
        )
        f.write(
            f"  [param_est prompt]\n{spec.param_est_prompt_schema.build_prompt('explore', parents, spec.flat_config, current_program=p)}\n\n"
        )
        f.write(
            f"  [jax model prompt]\n{spec.jax_model_prompt_schema.build_prompt('explore', current_program=p, config=spec.flat_config)}\n\n"
        )
        if p.image_path:
            f.write(f"  [image] {p.image_path}\n")
        f.write("\n")

    _flush_warnings(log)
    f.flush()
