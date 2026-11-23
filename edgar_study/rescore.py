"""Strict holdout after a run: pick the winner on validation cells, then score it once on locked cells.

Candidates are the top programs by discovery loss from upstream's final database. Both Gaussian
seeds are always scored as references. Settings match upstream's final evaluation (Adam, up to
2,000 steps, trial split seed 0, penalty 0.01 per parameter).
"""

import hashlib
import inspect
import json

PENALTY = 0.01


def _database(run_dir):
    found = sorted(run_dir.glob("program_databases/*/*/combined/programs_db.csv"))
    if len(found) != 1:
        raise SystemExit(f"Expected one combined programs_db.csv under {run_dir}, found {len(found)}.")
    return found[0]


def rescore(args, import_upstream):
    import numpy as np
    import pandas as pd

    from edgar_study.datasets import load_dataset, prepare

    upstream = import_upstream(args.upstream.resolve())
    engine, utils, seeds = upstream["hypothesis_engine"], upstream["utils"], upstream["seed_programs"]
    loss = upstream["loss_functions"].quadratic_loss
    run_dir = args.run.resolve()

    response, angles, _ = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")

    translations = {}
    path = run_dir / "translations.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            translations[row["numpy_sha256"]] = row["jax_code"]

    database = pd.read_csv(_database(run_dir))
    database = database.sort_values("train_loss").drop_duplicates("program_code_string")
    learned = database[database["iteration_number"] >= 0].head(args.top)

    candidates = [{"id": f"seed_{k}", "iteration": -1, "discovery_loss": None,
                   "model": getattr(seeds, f"neuron_model_{k}_jax"),
                   "estimator": getattr(seeds, f"parameter_estimator_{k}"),
                   "code": None} for k in (1, 2)]
    for _, row in learned.iterrows():
        jax_code = translations.get(hashlib.sha256(row["program_code_string"].encode()).hexdigest())
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
        np.random.seed(0)  # upstream samples cells for compatibility checks with NumPy's RNG
        _, _, total, params = engine.objective(
            candidate["model"], candidate["estimator"], loss, x[cells], y[cells],
            max_iter=2_000, tol=1e-6, param_penalty_weight=PENALTY)
        total = float(total)
        if not np.isfinite(total):
            return None
        count = int(params.shape[1])
        names = [n for n in inspect.signature(candidate["model"]).parameters if n != "theta"]
        medians = np.median(np.asarray(params), axis=0).tolist()
        return {"score": total, "mse": (total - PENALTY * count) / 10, "parameters": count,
                "median_fitted_params": dict(zip(names, medians)) if len(names) == count else medians}

    rows = []
    for candidate in candidates:
        row = {key: value for key, value in candidate.items() if key not in ("model", "estimator")}
        row["selection"] = score(candidate, groups["selection"])
        rows.append((candidate, row))
        print(candidate["id"], row["selection"])

    learned_rows = [(c, r) for c, r in rows if c["iteration"] >= 0 and r["selection"]]
    winner = min(learned_rows, key=lambda pair: pair[1]["selection"]["score"]) if learned_rows else None
    # The locked cells are used once: for the winner and the two references.
    for candidate, row in rows:
        if candidate["iteration"] < 0 or (winner and candidate is winner[0]):
            row["locked_test"] = score(candidate, groups["locked_test"])

    reference = next(r for c, r in rows if c["id"] == "seed_2")
    result = {
        "selection_rule": "lowest selection-cell penalized score among the top "
                          f"{args.top} unique learned programs by discovery loss",
        "cell_counts": {name: int(len(cells)) for name, cells in groups.items()},
        "winner": winner[1] if winner else None,
        "double_gaussian": reference, "single_gaussian": rows[0][1],
        "delta_locked_test_vs_double_gaussian": (
            reference["locked_test"]["score"] - winner[1]["locked_test"]["score"]
            if winner and winner[1].get("locked_test") else None),
        "candidates": [row for _, row in rows],
    }
    out = run_dir / "rescore.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"Wrote {out}")
    return result
