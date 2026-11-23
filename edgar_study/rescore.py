"""Strict holdout after a run: pick the winner on validation cells, then score it once on locked cells.

Candidates are the top programs by discovery loss from upstream's final database. Both Gaussian
seeds are always scored as references. Settings match upstream's final evaluation (Adam, up to
2,000 steps, trial split seed 0, penalty 0.01 per parameter).
"""

import inspect
import json

from edgar_study.upstream import code_key, combined_file, import_upstream, score_program

MAX_ITER = 2_000  # upstream's final evaluation


def rescore(args):
    import numpy as np
    import pandas as pd

    from edgar_study.datasets import load_dataset, prepare

    upstream = import_upstream(args.upstream.resolve())
    utils, seeds = upstream["utils"], upstream["seed_programs"]
    run_dir = args.run.resolve()
    database = combined_file(run_dir, "programs_db.csv")
    if database is None:
        raise SystemExit(f"No combined programs_db.csv under {run_dir}; has the run finished?")

    response, angles, _ = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")

    translations = {}
    path = run_dir / "translations.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            translations[row["numpy_sha256"]] = row["jax_code"]

    frame = pd.read_csv(database).sort_values("train_loss").drop_duplicates("program_code_string")
    candidates = [{"id": f"seed_{k}", "iteration": -1,
                   "model": getattr(seeds, f"neuron_model_{k}_jax"),
                   "estimator": getattr(seeds, f"parameter_estimator_{k}")} for k in (1, 2)]
    for _, row in frame[frame["iteration_number"] >= 0].head(args.top).iterrows():
        jax_code = translations.get(code_key(row["program_code_string"]))
        candidates.append({
            "id": f"{row['iteration_number']}_{row['birth_island']}_{row['batch_index']}",
            "iteration": int(row["iteration_number"]), "discovery_loss": float(row["train_loss"]),
            "upstream_eval_loss": float(row["test_loss"]),
            "model": utils.str_to_func(jax_code, "neuron_model"),
            "estimator": utils.str_to_func(row["parameter_estimator_code_string"], "parameter_estimator"),
            "code": row["program_code_string"], "jax_code": jax_code,
        })

    def score(candidate, cells):
        if candidate["model"] is None or candidate["estimator"] is None:
            return None
        total, mse, count, params = score_program(upstream, candidate["model"], candidate["estimator"],
                                                  x[cells], y[cells], MAX_ITER)
        if not np.isfinite(total):
            return None
        names = [n for n in inspect.signature(candidate["model"]).parameters if n != "theta"]
        medians = np.median(np.asarray(params), axis=0).tolist()
        return {"score": total, "mse": mse, "parameters": count,
                "median_fitted_params": dict(zip(names, medians)) if len(names) == count else medians}

    for candidate in candidates:
        candidate["selection"] = score(candidate, groups["selection"])
        print(candidate["id"], candidate["selection"])

    learned = [c for c in candidates if c["iteration"] >= 0 and c["selection"]]
    winner = min(learned, key=lambda c: c["selection"]["score"], default=None)
    # The locked cells are used once: for the winner and the two references.
    for candidate in candidates[:2] + ([winner] if winner else []):
        candidate["locked_test"] = score(candidate, groups["locked_test"])
    for candidate in candidates:
        del candidate["model"], candidate["estimator"]

    single, double = candidates[:2]
    result = {
        "selection_rule": "lowest selection-cell penalized score among the top "
                          f"{args.top} unique learned programs by discovery loss",
        "cell_counts": {name: int(len(cells)) for name, cells in groups.items()},
        "winner": winner, "double_gaussian": double, "single_gaussian": single,
        "delta_locked_test_vs_double_gaussian": (
            double["locked_test"]["score"] - winner["locked_test"]["score"]
            if winner and winner["locked_test"] else None),
        "candidates": candidates,
    }
    out = run_dir / "rescore.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"Wrote {out}")
    return result
