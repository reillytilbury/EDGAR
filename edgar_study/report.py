"""Summarise a run: best-so-far discovery loss per proposal, program funnel, requests and spend."""

import collections
import json
import re
import statistics

from edgar_study.go_client import list_cost
from edgar_study.upstream import MSE_SCALE, PENALTY, combined_file

# A fitted exponent applied to a distance-like term, e.g. `** p`, `np.power(x, shape)`, `** self_p`.
EXPONENT = re.compile(r"\*\*\s*\(?\s*[A-Za-z_]\w*|np\.power\(\s*[^,]+,\s*[A-Za-z_]\w*")

SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, LIGHT_BLUE = "#2a78d6", "#86b6ef"


def _census(run_dir):
    # Prefer the final census; fall back to the latest per-generation one while a run is in progress.
    final = combined_file(run_dir, "census.npy")
    found = [final] if final else sorted(
        run_dir.glob("program_databases/*/*/iteration_updates/iteration_*/census.npy"),
        key=lambda path: int(path.parent.name.split("_")[1]))
    if not found:
        raise SystemExit(f"No census.npy under {run_dir}.")
    import numpy as np
    return np.load(found[-1], allow_pickle=True)


def _scored_programs(run_dir):
    found = sorted(run_dir.glob("program_databases/*/*/hypothesis_engine.log"))
    if not found:
        return []
    programs, generation, lines, collecting = [], None, [], False
    for line in found[0].read_text(errors="replace").splitlines():
        if line.startswith("id="):
            generation = int(line[3:].split(",")[0])
        elif line.startswith("Neuron Model: "):
            collecting, lines = True, []
        elif line.startswith("Neuron Model (JAX): ") and collecting:
            programs.append((generation, "\n".join(lines)))
            collecting = False
        elif collecting:
            lines.append(line)
    return programs


def report(args):
    import numpy as np

    run_dir = args.run.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    batch = manifest.get("batch_size", 6)  # runs before these options existed used upstream's 8 × 6
    per_generation_count = manifest.get("islands", 8) * batch
    census = _census(run_dir)
    seeds = {int(row[2]): float(row[4]) for row in census if row[0] == -1}  # 0: single, 1: double Gaussian
    learned = [row for row in census if row[0] >= 0]
    generations = int(max([row[0] for row in learned], default=-1)) + 1

    # Proposal index counts every attempted slot, so failed proposals leave flat segments.
    index = np.array([row[0] * per_generation_count + row[1] * batch + row[2] for row in learned], dtype=int)
    loss = np.array([float(row[4]) for row in learned])
    attempted = generations * per_generation_count
    best = np.full(attempted, seeds[1])
    np.minimum.at(best, index, loss)
    best = np.minimum.accumulate(best)

    calls = [json.loads(line) for line in (run_dir / "api_calls.jsonl").read_text().splitlines()]
    by_role = collections.defaultdict(list)
    for call in calls:
        by_role[call["role"]].append(call)

    # Every scored program's NumPy source is in upstream's log, after an "id=generation,island,batch" line.
    scored_code = _scored_programs(run_dir)
    exponent_generations = sorted(g for g, code in scored_code if EXPONENT.search(code))
    first_exponent_generation = exponent_generations[0] if exponent_generations else None

    per_generation = []
    for generation in range(generations):
        scored = loss[(index // per_generation_count) == generation]
        per_generation.append({"generation": generation, "scored": int(scored.size),
                               "failed": per_generation_count - int(scored.size),
                               "best": float(scored.min()) if scored.size else None,
                               "best_so_far": float(best[(generation + 1) * per_generation_count - 1])})

    result = {
        "run_id": manifest["run_id"], "models": manifest["models"], "status": manifest.get("status"),
        "wall_hours": round((manifest.get("finished", manifest["started"]) - manifest["started"]) / 3600, 2),
        "generations": generations, "proposals_attempted": attempted, "programs_scored": int(loss.size),
        "seed_losses": seeds, "best_discovery_loss": float(loss.min()) if loss.size else None,
        "improvement_over_double_gaussian": seeds[1] - float(loss.min()) if loss.size else None,
        "per_generation": per_generation,
        "requests": {role: {"count": len(rows),
                            "ok": sum(r["status"] == 200 and r["content_chars"] > 0 for r in rows),
                            "truncated": sum(r.get("finish_reason") == "length" for r in rows),
                            "median_latency_s": statistics.median(r["latency_s"] for r in rows),
                            "output_tokens": sum(r["usage"].get("completion_tokens") or 0 for r in rows),
                            "input_tokens": sum(r["usage"].get("prompt_tokens") or 0 for r in rows)}
                     for role, rows in by_role.items()},
        "list_cost_usd": sum(list_cost(c["usage"]) for c in calls),
        "scored_programs_with_fitted_exponent": len(exponent_generations),
        "first_generation_with_fitted_exponent": first_exponent_generation,
    }
    (run_dir / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    _plot(run_dir / "best_so_far.png", index, loss, best, seeds, manifest, per_generation_count)
    print(json.dumps({k: v for k, v in result.items() if k != "per_generation"}, indent=2, default=float))
    return result


def _plot(path, index, loss, best, seeds, manifest, per_generation_count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    double = seeds[1]
    visible = loss < double + 5  # very poor programs would squash the interesting range
    hidden = int((~visible).sum())
    axis.scatter(index[visible], loss[visible], s=14, color=LIGHT_BLUE, edgecolor="none",
                 label="Each scored program" + (f" ({hidden} above {double + 5:.0f} not shown)" if hidden else ""),
                 zorder=2)
    axis.step(range(len(best)), best, where="post", color=BLUE, linewidth=2, label="Best so far", zorder=3)
    axis.axhline(double, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1,
                 label=f"Double Gaussian seed ({double:.2f})")
    for generation in range(1, len(best) // per_generation_count):
        axis.axvline(generation * per_generation_count, color="#e6e5e0", linewidth=0.8, zorder=0)
    axis.set_xlabel("Proposals attempted (failed proposals included)", color=TEXT)
    axis.set_ylabel(f"Discovery loss ({MSE_SCALE}·MSE + {PENALTY}·params, lower is better)", color=TEXT)
    axis.set_title(f"EDGAR on {manifest['models']['proposer']}: best discovery loss", color=TEXT,
                   loc="left", fontsize=11)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED)
    axis.legend(frameon=False, loc="upper right", labelcolor=TEXT)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
