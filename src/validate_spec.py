"""Pre-flight validation for an EDGAR spec module.

Runs a sequence of checks (required symbols, data loading, splits, seed
models, parameter estimators, loss function, and optional plot function)
and prints [PASS] / [FAIL] for each.  Returns True when every check passes.
"""
import tempfile
import traceback
from pathlib import Path

import numpy as np


def validate_spec(
    spec_module,
    config: dict,
    data_processing_params: dict,
    load_and_process_data_fn,
    train_test_split_fn,
    loss_fn,
    plot_model_fits_fn,
) -> bool:
    """Run all validation checks against *spec_module*.

    Parameters
    ----------
    spec_module
        The imported ``projects.<task>.spec`` module.
    config : dict
        Merged experiment config (defaults + task overrides).
    data_processing_params : dict
        Kwargs forwarded to the data-loading wrapper.
    load_and_process_data_fn : callable
        Already-wrapped loader (from ``run._build_load_and_process_data_fn``).
    train_test_split_fn : callable
        Already-wrapped split function (from ``run._build_train_test_split_fn``).
    loss_fn : callable | None
        Already-validated loss (from ``run._build_loss_fn``).
    plot_model_fits_fn : callable | None
        Already-wrapped plotter (from ``run._build_plot_model_fits_fn``), or None.

    Returns
    -------
    bool
        True if every check passed.
    """
    passed = 0
    failed = 0

    def _pass(name: str, detail: str = ""):
        nonlocal passed
        passed += 1
        msg = f"  [PASS] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    def _fail(name: str, reason: str):
        nonlocal failed
        failed += 1
        print(f"  [FAIL] {name} — {reason}")

    def _squeeze_targets(yi):
        """Squeeze (1, n_trials) -> (n_trials,) to match engine behavior."""
        yi = np.asarray(yi)
        if yi.ndim == 2 and yi.shape[0] == 1:
            return yi[0]
        return yi

    print("\n=== EDGAR spec validation ===\n")

    # ------------------------------------------------------------------
    # Check 1: Required symbols
    # ------------------------------------------------------------------
    required_symbols = [
        "load_and_process_data",
        "train_test_split",
        "model_v1",
        "model_v2",
        "param_est_v1",
        "param_est_v2",
        "loss_fn",
    ]
    missing = [s for s in required_symbols if not callable(getattr(spec_module, s, None))]
    if missing:
        _fail("Required symbols", f"missing or not callable: {missing}")
        print(f"\n=== Summary: {passed} passed, {failed} failed ===")
        return False
    _pass("Required symbols", ", ".join(required_symbols))

    # ------------------------------------------------------------------
    # Check 2: Data loading
    # ------------------------------------------------------------------
    inputs = outputs = None
    try:
        inputs, outputs = load_and_process_data_fn(**data_processing_params)
        n_samples, n_features, n_trials = inputs.shape
        n_samples_y, n_targets, n_trials_y = outputs.shape
        _pass(
            "Data loading",
            f"inputs {inputs.shape}, outputs {outputs.shape}",
        )
    except Exception:
        _fail("Data loading", traceback.format_exc().splitlines()[-1])
        print(f"\n=== Summary: {passed} passed, {failed} failed ===")
        return False

    # ------------------------------------------------------------------
    # Check 3: Train/test split
    # ------------------------------------------------------------------
    random_seed = config.get("experiment_params", {}).get("random_seed", 42)
    try:
        train_samples, test_samples, train_trials, test_trials = train_test_split_fn(
            inputs, random_seed,
        )
        # bounds check
        all_samples = np.concatenate([train_samples, test_samples])
        all_trials = np.concatenate([train_trials, test_trials])
        if np.any(all_samples < 0) or np.any(all_samples >= n_samples):
            _fail("Train/test split", "sample indices out of bounds")
        elif np.any(all_trials < 0) or np.any(all_trials >= n_trials):
            _fail("Train/test split", "trial indices out of bounds")
        elif np.intersect1d(train_samples, test_samples).size > 0:
            _fail("Train/test split", "train/test sample indices overlap")
        elif np.intersect1d(train_trials, test_trials).size > 0:
            _fail("Train/test split", "train/test trial indices overlap")
        else:
            _pass(
                "Train/test split",
                f"train_samples={len(train_samples)}, test_samples={len(test_samples)}, "
                f"train_trials={len(train_trials)}, test_trials={len(test_trials)}",
            )
    except Exception:
        _fail("Train/test split", traceback.format_exc().splitlines()[-1])

    # ------------------------------------------------------------------
    # Check 4: Seed model outputs
    # ------------------------------------------------------------------
    model_outputs = {}
    for name in ("model_v1", "model_v2"):
        model_fn = getattr(spec_module, name)
        try:
            Xi = inputs[0]  # (n_features, n_trials)
            default_params = getattr(model_fn, "DEFAULT_PARAMS", None)
            if default_params is not None:
                y_pred = model_fn(Xi, default_params)
            else:
                # Fall back to the corresponding param estimator
                est_name = name.replace("model_", "param_est_")
                est_fn = getattr(spec_module, est_name)
                yi = _squeeze_targets(outputs[0])
                params = est_fn(Xi, yi)
                y_pred = model_fn(Xi, params)
            y_pred = np.asarray(y_pred)
            expected_shapes = [(n_trials,), (n_targets, n_trials)]
            if y_pred.shape not in expected_shapes:
                _fail(name, f"output shape {y_pred.shape}, expected one of {expected_shapes}")
            else:
                _pass(name, f"output shape {y_pred.shape}")
            model_outputs[name] = y_pred
        except Exception:
            _fail(name, traceback.format_exc().splitlines()[-1])

    # ------------------------------------------------------------------
    # Check 5: Parameter estimators
    # ------------------------------------------------------------------
    for name in ("param_est_v1", "param_est_v2"):
        est_fn = getattr(spec_module, name)
        try:
            Xi = inputs[0]
            yi = _squeeze_targets(outputs[0])
            result = est_fn(Xi, yi)
            if not isinstance(result, dict) or len(result) == 0:
                _fail(name, f"expected non-empty dict, got {type(result).__name__}")
            else:
                _pass(name, f"keys={list(result.keys())}")
        except Exception:
            _fail(name, traceback.format_exc().splitlines()[-1])

    # ------------------------------------------------------------------
    # Check 6: Loss function
    # ------------------------------------------------------------------
    if loss_fn is not None:
        try:
            y_dummy = np.ones((n_targets, n_trials))
            loss_val = loss_fn(y_dummy, y_dummy)
            loss_arr = np.asarray(loss_val)
            if not np.all(np.isfinite(loss_arr)):
                _fail("loss_fn", f"non-finite result: {loss_arr}")
            else:
                _pass("loss_fn", f"result shape {loss_arr.shape}")
        except Exception:
            _fail("loss_fn", traceback.format_exc().splitlines()[-1])
    else:
        _fail("loss_fn", "loss_fn is None after build")

    # ------------------------------------------------------------------
    # Check 7: Plot function (if present)
    # ------------------------------------------------------------------
    if plot_model_fits_fn is not None:
        tmp_path = None
        try:
            from src.data_structures import Inputs, Outputs
            from src import utils

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
            import os
            os.close(tmp_fd)

            # Build a minimal programs_list from the first seed model
            model_fn = getattr(spec_module, "model_v1")
            est_fn = getattr(spec_module, "param_est_v1")
            Xi = inputs[0]
            yi = _squeeze_targets(outputs[0])
            params = est_fn(Xi, yi)
            n_samp = inputs.shape[0]
            programs_list = [{
                "model": model_fn,
                "params": utils.broadcast_params(params, n_samp),
            }]

            X_eval = utils.build_evaluation_points(
                inputs=Inputs.from_array(inputs),
                n_bins=config.get("experiment_params", {}).get("n_bins", 100),
            )

            plot_model_fits_fn(
                X=inputs,
                Y=outputs,
                programs_list=programs_list,
                X_eval=X_eval,
                save_path=tmp_path,
                labels=("seed_v1",),
            )
            if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
                _pass("plot_model_fits", "plot file created")
            else:
                _fail("plot_model_fits", "plot file not created or empty")
        except Exception:
            _fail("plot_model_fits", traceback.format_exc().splitlines()[-1])
        finally:
            if tmp_path is not None and Path(tmp_path).exists():
                Path(tmp_path).unlink()
    else:
        print("  [SKIP] plot_model_fits — not defined in spec")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print(f"\n=== Summary: {passed}/{total} passed, {failed} failed ===\n")
    return failed == 0