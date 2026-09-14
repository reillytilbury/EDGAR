"""EDGAR entry points for the Wilson-Cowan (WC) discovery task.

The underlying system (see ``simulate_data.py`` — the source of truth) is the base
Wilson-Cowan model: two **fully observed** populations, excitatory ``E`` and
inhibitory ``I``, driven by a per-timestep external stimulus (an excitatory pulse
or an inhibitory pulse). There is no hidden variable.

Contract for the LLM's program (see ``seed_programs/wilson_cowan.py``):
    ``model(state, y_prev, params) -> (new_state, mean)``
    * ``y_prev`` is a **dict** ``{"E_prev","I_prev","stim_E_prev","stim_I_prev"}`` —
      the previous observation bundled with the previous stimulus.
    * ``new_state`` is an (empty) dict carry — the base model needs no hidden state.
    * ``mean`` is ``(E, I)`` — the predicted next observation.
    * ``params`` is a dict of the learnable WC parameters.

Prediction is one-step-ahead and teacher-forced: the prediction of ``y[t]`` is paired
with everything at ``t-1`` (``E[t-1], I[t-1], stim_E[t-1], stim_I[t-1]``). The scan
inputs are the ``[:-1]`` slice; the targets are ``[1:]``.

Loss is a heteroscedastic Gaussian NLL, averaged over stim conditions and time. The
observation noise is signal-dependent (the generator uses ``std = 0.1·sqrt(mean)``, so
``var ∝ mean``), which is large in the evoked transient and small at the ~0 resting
baseline. We model ``var_t = phi · max(mean_t, EPS_MEAN)`` with a single learnable
coefficient ``phi = exp(log_noise_coef)`` per sample, **shared across E and I** — so each
channel's variance follows its own predicted mean and the "natural weighting" of E vs I
falls out of the physics. This down-weights the high-variance transient by exactly the
right amount (the principled alternative to discarding the onset window) and replaces the
old per-sample ``scale`` term. The data is left in raw units so ``var ∝ mean`` is
meaningful and the mechanistic parameters stay interpretable.

``log_noise_coef`` reaches the (params-free) ``loss_fn`` the same way ``fhn_excitable``
threads its noise param: ``apply_model`` reads it from ``params`` and appends it as a
third output channel, which ``loss_fn`` reads back. The model function itself never sees
it. There are no ``s0_*`` initial-state parameters.

Cross-validation is over the 12 repeats: ``simulate_data.save_kfold_splits`` writes
``wc_fold{f}.npz`` files, each holding a repeat-averaged ``train_data`` / ``test_data``
pair (shape ``(n_samples, 2, T, 2)``). ``load_data`` reads one such file and additionally
splits the *samples* 50/50 into EDGAR's discover / validate sets.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# Floor on the mean inside the noise variance ``var = phi · max(mean, EPS_MEAN)``. The
# resting baseline is ~0 (and predictions can dip slightly negative), so this keeps the
# variance strictly positive; ~0.1 is around the resting activity level.
EPS_MEAN = 0.1


# ── EDGAR entry points ──


def load_data(
    data_path: str = "",
    sample_split_seed: int = 42,
    T_eval: int = 200,
    n_eval: int = 4,
):
    """Load one k-fold file and return EDGAR's ``(discover, validate, X_eval)`` split.

    ``data_path`` must point at a ``wc_fold{f}.npz`` produced by
    ``simulate_data.save_kfold_splits``. The 8 samples are split 50/50 into discover
    and validate; parameters are fit per sample on ``train_data`` and cross-validated
    on the held-out (repeat-averaged) ``test_data``.

    Every top-level dict value is a plain array with axis 0 = n_samples (the per-sample
    axis matched to per-sample params). ``"E"`` is the first key so the engine's
    ``next(iter(data.values())).shape[0]`` sees n_samples. The per-step dict ``y_prev``
    is assembled inside ``apply_model`` (a ``jax.lax.scan`` over a pytree), so no core
    EDGAR change is needed.
    """
    if not data_path:
        raise ValueError(
            "wilson_cowan load_data requires data_path pointing to a wc_fold*.npz file "
            "(generate with simulate_data.save_kfold_splits)."
        )
    raw = np.load(data_path)
    train_data = np.asarray(raw["train_data"])  # (n_samples, 2, T, 2)  last axis = (E, I)
    test_data = np.asarray(raw["test_data"])
    stimuli = np.asarray(raw["stimuli"])         # (2, T, 2)  [stim_cond, T, (stim_E, stim_I)]

    n_samples, n_stim, T, _ = train_data.shape

    # 50/50 sample split → discover / validate (params are fit per sample).
    perm = np.random.default_rng(sample_split_seed).permutation(n_samples)
    disc_idx = np.sort(perm[: n_samples // 2])
    val_idx = np.sort(perm[n_samples // 2:])

    # The stimulus is shared across samples; broadcast to (n, n_stim, T) per channel so
    # every array carries the n_samples axis (axis 0) that vmap/params map over.
    def _stim_arrays(n: int):
        sE = np.broadcast_to(stimuli[None, :, :, 0], (n, n_stim, T))
        sI = np.broadcast_to(stimuli[None, :, :, 1], (n, n_stim, T))
        return jnp.asarray(sE), jnp.asarray(sI)

    # No per-sample scale: the heteroscedastic NLL (see loss_fn) sets each residual's
    # weight from the fitted noise variance ``phi · mean_t``, which subsumes what the old
    # ``scale`` term did. The data is passed through in raw units.
    def _build(split_data: np.ndarray, idx: np.ndarray) -> dict:
        n = len(idx)
        sE, sI = _stim_arrays(n)
        return {
            # "E" first: engine reads n_samples from next(iter(data.values())).shape[0].
            "E": jnp.asarray(split_data[idx, :, :, 0]),      # (n, n_stim, T)
            "I": jnp.asarray(split_data[idx, :, :, 1]),
            "stim_E": sE,
            "stim_I": sI,
        }

    X_disc_train = _build(train_data, disc_idx)
    X_disc_test = _build(test_data, disc_idx)
    X_val_train = _build(train_data, val_idx)
    X_val_test = _build(test_data, val_idx)

    # X_eval: a small, short subset of the discover cells for fingerprint dedup.
    # _sample_indices index positions WITHIN the discover set (the scorer does
    # params[_sample_indices] against the per-discover-sample params).
    n_eval_actual = int(min(max(1, n_eval), len(disc_idx)))
    T_eval_actual = int(min(T_eval, T))
    eval_pos = np.sort(
        np.random.default_rng(sample_split_seed + 1).choice(
            len(disc_idx), n_eval_actual, replace=False
        )
    )
    disc_train_E = train_data[disc_idx, :, :, 0]
    disc_train_I = train_data[disc_idx, :, :, 1]
    sE_eval, sI_eval = _stim_arrays(len(disc_idx))
    X_eval = {
        "E": jnp.asarray(disc_train_E[eval_pos, :, :T_eval_actual]),
        "I": jnp.asarray(disc_train_I[eval_pos, :, :T_eval_actual]),
        "stim_E": sE_eval[eval_pos, :, :T_eval_actual],
        "stim_I": sI_eval[eval_pos, :, :T_eval_actual],
        "_sample_indices": eval_pos,
    }

    print(
        f"[wilson_cowan] {data_path}: n_samples={n_samples}, n_stim={n_stim}, T={T}; "
        f"discover/validate={len(disc_idx)}/{len(val_idx)} "
        f"(disc={disc_idx.tolist()}, val={val_idx.tolist()}); "
        f"X_eval n={n_eval_actual}, T={T_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def apply_model(model_fn, data, params):
    """Teacher-forced one-step-ahead scan of ``model_fn`` over every (sample, stim).

    Builds the per-step dict ``y_prev`` and scans over it (``jax.lax.scan`` handles the
    pytree natively). vmaps over samples (axis 0, matched to per-sample ``params``) and,
    inside, over the two stim conditions (params are shared across conditions — same
    cell). Returns ``(n_samples, n_stim, T-1, 2)``: predicted (E, I) at each step.

    Models with a hidden state (e.g. the WCS slow variable ``S``) expose an
    ``INITIAL_STATE`` attribute that seeds the scan carry; stateless models (base WC)
    have none, so the carry starts as an empty dict exactly as before.

    The fitted per-sample observation-noise coefficient ``log_noise_coef`` is read from
    ``params`` and appended as a constant third channel, so the params-free ``loss_fn``
    can recover it. Output is ``(n_samples, n_stim, T-1, 3)``: ``(E, I, log_noise_coef)``.
    The model function itself never sees ``log_noise_coef``.
    """
    E = data["E"]        # (n, n_stim, T)
    I = data["I"]
    sE = data["stim_E"]
    sI = data["stim_I"]

    init_state = getattr(model_fn, "INITIAL_STATE", {})

    def per_sample(E_s, I_s, sE_s, sI_s, p):
        def per_stim(E_c, I_c, sE_c, sI_c):
            xs = {
                "E_prev": E_c[:-1],
                "I_prev": I_c[:-1],
                "stim_E_prev": sE_c[:-1],
                "stim_I_prev": sI_c[:-1],
            }

            def step(state, y_prev):
                new_state, mean = model_fn(state, y_prev, p)
                E_next, I_next = mean
                return new_state, jnp.stack([E_next, I_next])

            _, means = jax.lax.scan(step, init_state, xs)  # (T-1, 2)
            return means

        means = jax.vmap(per_stim)(E_s, I_s, sE_s, sI_s)   # (n_stim, T-1, 2)
        log_nc = jnp.broadcast_to(p["log_noise_coef"], means.shape[:-1] + (1,))
        return jnp.concatenate([means, log_nc], axis=-1)   # (n_stim, T-1, 3)

    return jax.vmap(per_sample, in_axes=(0, 0, 0, 0, 0))(E, I, sE, sI, params)


def loss_fn(model_output, data):
    """Heteroscedastic Gaussian NLL, averaged over stim conditions and time.

    ``model_output`` is ``(n, n_stim, T-1, 3)``: ``(E_hat, I_hat, log_noise_coef)``, where
    ``log_noise_coef`` is the per-sample fitted noise coefficient carried through by
    ``apply_model``. Targets are the ``[1:]`` slice of the observed E/I.

    The observation variance is signal-dependent: ``var = phi · max(mean, EPS_MEAN)`` with
    ``phi = exp(log_noise_coef)`` shared by E and I. Each channel is weighted by its own
    predicted mean, so the high-variance evoked transient is smoothly down-weighted and the
    low-noise baseline dominates the fit. The mean inside the variance is detached
    (``stop_gradient``) so the model cannot lower the loss by inflating its predicted mean to
    buy variance; ``phi`` still gets a gradient through the ``log`` term. Returns ``(n,)``.
    """
    E_hat = model_output[..., 0]      # (n, n_stim, T-1)
    I_hat = model_output[..., 1]
    log_nc = model_output[..., 2]
    E_tgt = data["E"][:, :, 1:]
    I_tgt = data["I"][:, :, 1:]

    phi = jnp.exp(log_nc)
    var_E = phi * jnp.maximum(jax.lax.stop_gradient(E_hat), EPS_MEAN)
    var_I = phi * jnp.maximum(jax.lax.stop_gradient(I_hat), EPS_MEAN)

    nll_E = 0.5 * (jnp.log(var_E) + (E_tgt - E_hat) ** 2 / var_E)
    nll_I = 0.5 * (jnp.log(var_I) + (I_tgt - I_hat) ** 2 / var_I)
    return jnp.mean(nll_E + nll_I, axis=(1, 2))  # (n,)
