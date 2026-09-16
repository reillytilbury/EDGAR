"""Post-run analysis for a stopped/finished FHN evolution.

Usage:
    python projects/fhn_excitable/scripts/post_run_analysis.py <run_dir>

Produces:
    <run_dir>/post_analysis.md    text summary
    <run_dir>/figures/top4_fits.png  best-program predictions vs true traces
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import yaml

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.fhn_excitable.data_loader.load_data import (   # noqa: E402
    apply_model, TEST_WARMUP_STEPS, load_data,
)


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) < 1e8


def analyse(run_dir: Path) -> None:
    pop = [json.loads(l) for l in (run_dir / "population.jsonl").open()]
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    pp = dict(cfg.get("project_params", {}))
    (Xd_tr, Xd_te), _, _ = load_data(**pp)

    # Oracle — on the test window (discover.final is the held-out test loss now);
    y = np.asarray(Xd_te["y"], dtype=np.float64)
    V = np.asarray(Xd_te["_V_true"], dtype=np.float64)
    w = np.asarray(Xd_te["_w_true"], dtype=np.float64)
    y_shift = np.asarray(Xd_te["_y_shift"], dtype=np.float64)[:, None]
    y_scale = np.asarray(Xd_te["_y_scale"], dtype=np.float64)[:, None]
    dt, I0 = pp["dt"], pp["I0"]
    V_next_raw = V[:, :-1] + dt * (V[:, :-1] - V[:, :-1] ** 3 / 3.0 - w[:, :-1] + I0)
    V_next_y = (V_next_raw - y_shift) / y_scale
    resid = (y[:, 1:] - V_next_y)[:, TEST_WARMUP_STEPS:]
    sigma_mle = np.maximum(resid.std(axis=1), 1e-6)
    L_oracle = float((np.log(sigma_mle) + 0.5).mean())
    L_pers = float(np.asarray(Xd_te["_persistence_nll"]).mean())

    # Program rows
    rows = []
    for x in pop:
        keys = sorted((x.get("_default_params") or {}).keys())
        s0 = sorted([k[3:] for k in keys if k.startswith("s0_")])
        df = x["program_losses"].get("discover", {}).get("final")
        gen = x.get("birth", {}).get("generation")
        mode = x.get("birth", {}).get("mode")
        rows.append(dict(disc=df, idx=x["idx"], gen=gen, mode=mode,
                         name=x["name"], s0=s0, n_params=x.get("n_params"),
                         program=x))
    ok = [r for r in rows if _is_finite(r["disc"])]
    ok.sort(key=lambda r: r["disc"])

    # Seed floor: best finite discover.final among gen == -1 seeds
    seed_floors = [r["disc"] for r in ok if r["gen"] == -1]
    L_seed = min(seed_floors) if seed_floors else float("nan")

    observable_like = {"y_last", "x", "v", "freq_est", "ma"}
    has_hidden = [r for r in ok if any(k not in observable_like for k in r["s0"])]
    beats_seed = [r for r in ok if r["disc"] < L_seed]

    per_gen_best = {}
    per_gen_hidden = defaultdict(int)
    per_gen_total = defaultdict(int)
    for r in ok:
        g = r["gen"]
        if g is None or g == -1:
            continue
        per_gen_best[g] = min(per_gen_best.get(g, 1e9), r["disc"])
        per_gen_total[g] += 1
        if any(k not in observable_like for k in r["s0"]):
            per_gen_hidden[g] += 1

    # Model-name motif analysis
    name_counter = Counter()
    for r in ok:
        if r["gen"] is not None and r["gen"] != -1:
            n = r["name"].lower()
            for kw in ["fitzhugh", "fhn", "excitable", "rinzel", "adaptation",
                       "refractory", "recovery"]:
                if kw in n:
                    name_counter[kw] += 1

    # Emit report
    out = []
    out.append(f"# Post-run analysis: {run_dir.name}\n")
    out.append(f"Populated with {len(pop)} programs total; {len(ok)} produced "
               f"finite discover loss.\n")
    out.append("## Benchmarks\n")
    out.append(f"- **Oracle NLL floor**   : `{L_oracle:+.4f}`  (true dynamics + true w, per-traj MLE σ)")
    out.append(f"- **Best seed floor**    : `{L_seed:+.4f}`  (best of 4 linear-observation seeds)")
    out.append(f"- **Persistence baseline** : `{L_pers:+.4f}`")
    out.append(f"- **Discovery budget**   : {L_pers - L_oracle:+.4f} nat")
    out.append(f"- **Seed-vs-oracle gap** : {L_seed - L_oracle:+.4f} nat")
    out.append("")

    out.append("## Discovery signal\n")
    out.append(f"- Programs with a novel state key (not in {sorted(observable_like)}): "
               f"**{len(has_hidden)} / {len(ok)}** ({100*len(has_hidden)/len(ok):.0f}%)")
    out.append(f"- Programs beating best seed floor `{L_seed:+.4f}`: "
               f"**{len(beats_seed)} / {len(ok)}** ({100*len(beats_seed)/len(ok):.0f}%)")
    out.append(f"- Model-name motif counts (LLM-generated only):")
    for kw, n in name_counter.most_common():
        out.append(f"    * `{kw}` : {n}")
    out.append("")

    out.append("## Per-generation progress\n")
    out.append("| gen | mode-tag | n_finite | hidden-var rate | best disc NLL | gap to oracle |")
    out.append("|-----|----------|----------|-----------------|---------------|---------------|")
    for g in sorted(per_gen_best.keys()):
        best = per_gen_best[g]
        n = per_gen_total[g]
        hr = per_gen_hidden[g]
        mode = next((r["mode"] for r in ok if r["gen"] == g and r["mode"]), "-")
        out.append(f"| {g} | {mode} | {n} | {hr}/{n} | {best:+.4f} | {best - L_oracle:+.4f} |")
    out.append("")

    out.append("## Top-10 programs by discover NLL\n")
    out.append("| rank | idx | gen | mode | disc NLL | state keys | name |")
    out.append("|------|-----|-----|------|----------|------------|------|")
    for i, r in enumerate(ok[:10]):
        out.append(f"| {i+1} | {r['idx']} | {r['gen']} | {r['mode']} | "
                   f"{r['disc']:+.4f} | `{r['s0']}` | {r['name'][:60]} |")
    out.append("")

    out.append("## Best program's code\n")
    out.append(f"```python\n{ok[0]['program']['code']['model']}\n```\n")
    out.append(f"```python\n# param_est\n{ok[0]['program']['code']['param_est']}\n```\n")

    (run_dir / "post_analysis.md").write_text("\n".join(out))
    print(f"Wrote {run_dir / 'post_analysis.md'} ({len(out)} lines)")

    # ── Figure: top-4 fits vs true traces ──
    figpath = run_dir / "figures" / "top4_fits.png"
    figpath.parent.mkdir(exist_ok=True)

    # Reconstruct the full trajectory (test overlaps train by one boundary sample)
    # so we can show the fit on the train window AND the held-out test window in
    # one trace, with a boundary marker. 
    y_train = np.asarray(Xd_tr["y"])
    y_test = np.asarray(Xd_te["y"])
    split_t = y_train.shape[1]
    y_disc = np.concatenate([y_train, y_test[:, 1:]], axis=1)
    n_traj, T = y_disc.shape
    T_show = T - 1
    show = np.linspace(0, min(n_traj - 1, 3), 4).astype(int)
    top4 = ok[:4]

    fig, axes = plt.subplots(len(show), 1, figsize=(11, 2.4 * len(show)), squeeze=False)
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    def _compile_model(src: str):
        ns = {}
        # jax-translated source is what the pipeline actually scored; use it.
        exec(src, ns)
        return ns["model"]

    model_fns = []
    for r in top4:
        try:
            src = r["program"]["code"].get("model_jax") or r["program"]["code"]["model"]
            model_fns.append(_compile_model(src))
        except Exception as e:
            print(f"[warn] could not compile idx={r['idx']}: {e}")
            model_fns.append(None)

    for row_i, s in enumerate(show):
        ax = axes[row_i, 0]
        ax.plot(np.arange(T_show + 1), y_disc[s, :T_show + 1], "k-", lw=0.6,
                alpha=0.55, label="true V (normalised)")
        for j, (r, mf) in enumerate(zip(top4, model_fns)):
            if mf is None:
                continue
            try:
                params = {k: jnp.asarray(np.asarray(v)[s:s + 1])
                          for k, v in r["program"]["params"].items()}
                out_arr = np.asarray(apply_model(
                    mf, {"y": jnp.asarray(y_disc[s:s + 1])}, params
                ))
                means = out_arr[0, :T_show, 0]
                ax.plot(np.arange(1, T_show + 1), means,
                        color=colors[j], lw=0.7,
                        label=f"#{r['idx']} L={r['disc']:.3f}")
            except Exception as e:
                print(f"[warn] rendering idx={r['idx']} traj={s}: {e}")

        # Mark the train/test boundary: left of the line is the fit window,
        # right is the held-out test window (params frozen, state carried over).
        ax.axvline(split_t, color="gray", ls="--", lw=0.8,
                   label="train | test" if row_i == 0 else None)
        ax.set_ylabel(f"traj {s}")
        if row_i == 0:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title("Top-4 FHN-evolved fits — train fit | held-out test "
                         "(one-step predictions)")
        if row_i == len(show) - 1:
            ax.set_xlabel("time bin")

    fig.tight_layout()
    fig.savefig(figpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {figpath}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <run_dir>")
    analyse(Path(sys.argv[1]).resolve())
