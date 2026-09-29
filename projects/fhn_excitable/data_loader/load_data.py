"""State-space DSL testbed: stochastic FitzHugh-Nagumo, observation of V only.

Contract (same as oscillator_ss):
    LLM writes ``model(state, y_prev, params) -> (new_state, mean)``.
    ``params`` includes ``log_sigma_obs`` (learnable observation noise) and any
    number of ``s0_*``-prefixed keys that declare the initial-state values for
    the scan carry (framework strips the prefix and passes them as ``state``).

What's different from oscillator_ss:
    The underlying system has a HIDDEN recovery variable ``w`` that is not
    observed. Predicting V well requires modelling ``w`` as latent state — a
    linear oscillator or AR model cannot represent FHN's excitability. This
    is the whole point: the "seed floor" is set by 1D-observation-only models,
    and evolution must introduce a slow hidden variable to beat them.

Underlying dynamics (Nagumo 1962 form):
    dV/dt = V - V^3/3 - w + I_0     + σ_p * η_V(t)
    dw/dt = ε * (V + a - b * w)
    y[t]  = V[t] + σ_obs * ε_obs(t)

For canonical excitable dynamics: a=0.7, b=0.8, ε=0.08, I_0=0.5. With those
parameters, V spikes ~30 times over T=2400 at dt=0.05.

The true ``w`` trajectory is preserved on the returned dict as ``_w_true_*``
so ``scripts/fhn_oracle_nll.py`` can compute the oracle Gaussian NLL as a
lower bound on any model's predictive performance.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ── module-level constants (baked into loss_fn's closure, jit-safe) ──

WARMUP_STEPS: int = 100
"""Number of leading predictions ignored by ``loss_fn``. Set higher than the
oscillator_ss default (50) because FHN's recovery variable ``w`` starts at
its DEFAULT_PARAMS prior and takes ~1 spike-cycle (~90 steps at ε=0.08,
dt=0.05) to stabilise. See docstring in oscillator_ss/load_data.py for why
this is a Python constant, not a config value."""

TEST_WARMUP_STEPS: int = 0
"""Leading predictions ignored by ``loss_fn`` when evaluating on the *test*
window. Zero: ``roll_state`` scans the whole train window (>> WARMUP_STEPS) from
``s0_*``, so the latent carried across the boundary is already converged — the
first test prediction has no init transient to discard. The test dicts stamp this
as ``_eval_warmup``; the train (fitting) path has no such key and falls back to
``WARMUP_STEPS``. Kept symmetric with ``WARMUP_STEPS`` so both are tunable in one
place. Safe as a data-dict value because the test path is eager (never jitted),
unlike the fit loss — so no ``ConcretizationError``."""


# ── data synthesis: stochastic FitzHugh-Nagumo ──


def _synth_fhn(
    rng: np.random.Generator,
    T: int,
    dt: float,
    I0: float,
    a: float,
    b: float,
    eps: float,
    proc_noise_std: float,
    obs_noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One trajectory of Euler-Maruyama-integrated stochastic FHN.

    Returns (y, w_raw, V_raw, y_shift, y_scale) where y = normalise(V + obs noise)
    to zero-mean unit-std, and (y_shift, y_scale) are the per-trajectory affine
    transform so that raw-scale predictions can be mapped back to y-scale:
        y_predicted = (raw_predicted - y_shift) / y_scale
    """
    V = np.zeros(T, dtype=np.float64)
    w = np.zeros(T, dtype=np.float64)
    V[0] = -1.0 + 0.1 * rng.standard_normal()
    w[0] = -0.5 + 0.1 * rng.standard_normal()

    for t in range(1, T):
        # process noise on V only, scaled by sqrt(dt) → dt-invariant SDE variance
        eta = proc_noise_std * rng.standard_normal() / np.sqrt(dt)
        dV = V[t - 1] - V[t - 1] ** 3 / 3.0 - w[t - 1] + I0 + eta
        dw = eps * (V[t - 1] + a - b * w[t - 1])
        V[t] = V[t - 1] + dt * dV
        w[t] = w[t - 1] + dt * dw

    y_raw = V + obs_noise_std * rng.standard_normal(T)
    y_shift = float(y_raw.mean())
    y_scale = float(y_raw.std() + 1e-8)
    y = (y_raw - y_shift) / y_scale
    return (y.astype(np.float32), w.astype(np.float32), V.astype(np.float32),
            y_shift, y_scale)


# ── EDGAR entry points ──


def load_data(
    data_path: str = "",
    n_trajectories: int = 32,
    T: int = 2400,
    train_frac: float = 0.5,
    dt: float = 0.05,
    I0: float = 0.5,
    a: float = 0.7,
    b: float = 0.8,
    eps: float = 0.08,
    proc_noise_std: float = 0.3,
    obs_noise_std: float = 0.05,
    T_eval: int = 200,
    n_eval: int = 4,
    seed: int = 42,
):
    """Generate stochastic FHN trajectories with observation of V only.

    Returns the standard EDGAR five-way split. Trials (timesteps) are split into
    two contiguous chunks at ``split_t = round(T * train_frac)``: train = the
    first ``train_frac`` of the trajectory, test = the rest. Params are fit on the
    train window and cross-validated on the held-out test window; the latent state
    is carried across the boundary by ``roll_state`` (so dynamics + noise params
    generalise across time). The initial-state ``s0_*`` params are *not*
    cross-validated.
    """
    del data_path  # synthetic
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac ({train_frac}) must be in (0, 1).")
    split_t = int(round(T * train_frac))
    min_win = WARMUP_STEPS + 40
    # Current set up : Both train and test must clear the warmup for a meaningful
    # post-warmup loss window.
    if split_t < min_win or (T - split_t + 1) < min_win:
        raise ValueError(
            f"train_frac={train_frac}, T={T} -> train={split_t} / "
            f"test={T - split_t + 1} timesteps; each must exceed WARMUP_STEPS "
            f"({WARMUP_STEPS}) + 40 = {min_win}. Increase T or move train_frac "
            "toward 0.5."
        )

    rng = np.random.default_rng(seed)
    ys, ws, Vs, shifts, scales = [], [], [], [], []
    for _ in range(n_trajectories):
        y_i, w_i, V_i, sh, sc = _synth_fhn(
            rng, T, dt, I0, a, b, eps, proc_noise_std, obs_noise_std
        )
        ys.append(y_i)
        ws.append(w_i)
        Vs.append(V_i)
        shifts.append(sh)
        scales.append(sc)
    all_y = np.stack(ys)
    all_w = np.stack(ws)
    all_V = np.stack(Vs)
    all_shift = np.asarray(shifts, dtype=np.float32)
    all_scale = np.asarray(scales, dtype=np.float32)

    split_rng = np.random.default_rng(seed + 1)
    perm = split_rng.permutation(n_trajectories)
    disc_idx = np.sort(perm[: n_trajectories // 2])
    val_idx = np.sort(perm[n_trajectories // 2 :])

    y_disc = jnp.asarray(all_y[disc_idx])
    y_val = jnp.asarray(all_y[val_idx])
    w_disc = jnp.asarray(all_w[disc_idx])
    w_val = jnp.asarray(all_w[val_idx])
    V_disc = jnp.asarray(all_V[disc_idx])
    V_val = jnp.asarray(all_V[val_idx])
    shift_disc = jnp.asarray(all_shift[disc_idx])
    shift_val = jnp.asarray(all_shift[val_idx])
    scale_disc = jnp.asarray(all_scale[disc_idx])
    scale_val = jnp.asarray(all_scale[val_idx])

    # ── temporal train/test split (contiguous chunks; split_t set above) ──
    def _train_win(y_full):
        return y_full[:, :split_t]

    def _test_win(arr):
        return arr[:, split_t - 1:]

    # Persistence-baseline NLL per trajectory — diagnostic, matches oscillator_ss.
    # Computed on the test window at TEST_WARMUP_STEPS so it stays comparable to
    # the model's (now warmup-free) held-out test loss.
    def _persistence_nll_per_trajectory(y_np: np.ndarray) -> np.ndarray:
        residuals = y_np[:, 1:] - y_np[:, :-1]
        sigma = np.maximum(residuals.std(axis=-1, keepdims=True), 1e-3)
        nll = np.log(sigma) + 0.5 * (residuals / sigma) ** 2
        return nll[:, TEST_WARMUP_STEPS:].mean(axis=-1).astype(np.float32)

    pers_disc = _persistence_nll_per_trajectory(np.asarray(_test_win(y_disc)))
    pers_val = _persistence_nll_per_trajectory(np.asarray(_test_win(y_val)))

    X_disc_train = {"y": _train_win(y_disc)}
    X_disc_test = {
        "y": _test_win(y_disc),
        "_eval_warmup": TEST_WARMUP_STEPS,
        "_persistence_nll": jnp.asarray(pers_disc),
        # Oracle sidecars for scripts/fhn_oracle_nll.py, sliced to the same test
        # window as "y"; scoring path never reads them (apply_model / loss_fn
        # only touch "y").
        "_w_true": _test_win(w_disc),
        "_V_true": _test_win(V_disc),
        "_y_shift": shift_disc,
        "_y_scale": scale_disc,
    }
    X_val_train = {"y": _train_win(y_val)}
    X_val_test = {
        "y": _test_win(y_val),
        "_eval_warmup": TEST_WARMUP_STEPS,
        "_persistence_nll": jnp.asarray(pers_val),
        "_w_true": _test_win(w_val),
        "_V_true": _test_win(V_val),
        "_y_shift": shift_val,
        "_y_scale": scale_val,
    }

    # X_eval: short traces for fingerprint dedup. A contiguous t=0 window of the
    # real discover cells (within the TRAIN region, so the fitted s0_* is the
    # correct init and no carry is needed).
    n_eval_actual = int(min(max(1, n_eval), len(disc_idx)))
    T_eval_actual = int(min(T_eval, split_t))
    eval_pos = np.sort(split_rng.choice(len(disc_idx), n_eval_actual, replace=False))
    X_eval = {
        "y": y_disc[eval_pos, :T_eval_actual],
        "_sample_indices": eval_pos,
        "_fingerprint_only": True,
    }

    print(
        f"[fhn_excitable] T={T}, dt={dt}, {n_trajectories} traj -> "
        f"disc/val={len(disc_idx)}/{len(val_idx)}; "
        f"split_t={split_t} (train {split_t} / test {T - split_t + 1} timesteps); "
        f"a={a}, b={b}, eps={eps}, I0={I0}; "
        f"proc/obs noise={proc_noise_std}/{obs_noise_std}; "
        f"WARMUP_STEPS={WARMUP_STEPS}; X_eval T={T_eval_actual}, n={n_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _split_params_s0(params: dict) -> tuple[dict, dict]:
    """Strip ``s0_``-prefixed keys → initial state dict; rest is dyn params.

    Only strips keys of the form ``s0_<name>`` with ``<name>`` non-empty.
    Duplicated (not imported) from oscillator_ss so this project stands alone.
    """
    init_state = {}
    dyn_params = {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        else:
            dyn_params[k] = v
    return init_state, dyn_params


def _scan_one(model_fn, y_traj, init_state, dyn_params):
    """Scan ``model_fn`` over one trajectory, returning ``(means, final_state)``.
    We keep the final carry so ``roll_state`` can hand it to the test window.

    Returns : 
        - means: (T-1,) array — the one-step ahead prediction at each step.
        - final_state: dictionary (pytree) of the same structure as ``init_state``.
    """
    def scan_step(state, y_prev):
        new_state, mean = model_fn(state, y_prev, dyn_params)
        return new_state, mean

    final_state, means = jax.lax.scan(scan_step, init_state, y_traj[:-1])
    return means, final_state


def apply_model(model_fn, data, params):
    """State-space scan wrapper.

    Returns:
      - Normal path: ``(n_samples, T-1, 2)`` — means and broadcast log_sigma.
      - Fingerprint path: ``(n_samples, T-1)`` — residuals (y_true - mean).

    When we apply the model to test trials, we don't want to use the fitted init_state. 
    Instead, we roll the state across the train/test boundary, which is achieved by
    ``roll_state`` on the train_window and saved as ``data["_init_carry"]``. The test window 
    then uses this carry as the initial state for its scan.
    
    Absent the key, behaviour is the plain-scan default - this is the behaviour on train trials. 
    """
    y = data["y"]
    fingerprint_only = bool(data.get("_fingerprint_only", False))
    init_carry = data.get("_init_carry")
    # None broadcasts to every sample; a per-sample carry is mapped over axis 0.
    carry_axis = None if init_carry is None else 0

    def per_sample(y_traj, p, carry):
        init_state, dyn_params = _split_params_s0(p)
        if carry is not None:
            init_state = carry
        means, _ = _scan_one(model_fn, y_traj, init_state, dyn_params)
        if fingerprint_only:
            targets = y_traj[1:]
            return targets - means
        log_sigma = jnp.full_like(means, dyn_params["log_sigma_obs"])
        return jnp.stack([means, log_sigma], axis=-1)

    return jax.vmap(per_sample, in_axes=(0, 0, carry_axis))(y, params, init_carry)


def roll_state(model_fn, data, params):
    """Return the per-sample final latent carry after scanning the train trials.

    Registered by the scorer (via ``TaskSpec.rollout_fn``) as the train→test
    hand-off: the returned pytree becomes ``data["_init_carry"]`` for the test
    evaluation. Scans from each sample's ``s0_*`` params — the train window always
    starts from the learnable initial state.
    """
    y = data["y"]

    def per_sample(y_traj, p):
        init_state, dyn_params = _split_params_s0(p)
        _, final_state = _scan_one(model_fn, y_traj, init_state, dyn_params)
        return final_state

    return jax.vmap(per_sample, in_axes=(0, 0))(y, params)


def loss_fn(model_output, data):
    """Per-sample Gaussian NLL over the post-warmup horizon.

    ``warmup`` defaults to the fit-time ``WARMUP_STEPS``; the test dicts override
    it to ``TEST_WARMUP_STEPS`` (=0) via ``_eval_warmup``. Only the eager test path
    supplies the key, so the jitted fit loss always sees the static default — no
    ``ConcretizationError``.
    """
    warmup = int(data.get("_eval_warmup", WARMUP_STEPS))
    y = data["y"][:, 1:]
    means = model_output[:, warmup:, 0]
    log_sigmas = model_output[:, warmup:, 1]
    tgt = y[:, warmup:]
    nll = log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2
    return jnp.mean(nll, axis=-1)


# ── debug / triage helper ──


def validate_step(model_fn, default_params: dict, program_code: str = "") -> None:
    """Eager sanity check for one program's model_fn (see oscillator_ss docs)."""
    assert isinstance(default_params, dict), (
        f"DEFAULT_PARAMS must be a dict, got {type(default_params).__name__}"
    )
    assert "log_sigma_obs" in default_params, (
        "DEFAULT_PARAMS must include 'log_sigma_obs' (observation-noise log-std). "
        f"Got keys: {sorted(default_params.keys())}"
    )

    init_state, dyn_params = _split_params_s0(default_params)
    init_state_j = jax.tree_util.tree_map(jnp.asarray, init_state)
    dyn_params_j = jax.tree_util.tree_map(jnp.asarray, dyn_params)

    try:
        new_state, mean = model_fn(init_state_j, jnp.asarray(0.5), dyn_params_j)
    except Exception as e:
        raise AssertionError(
            f"model_fn(init_state, y_prev=0.5, params) raised {type(e).__name__}: {e}\n"
            f"init_state={init_state}\ndyn_params={dyn_params}\n"
            f"---program code---\n{program_code}\n---end---"
        ) from e

    init_struct = jax.tree_util.tree_structure(init_state_j)
    new_struct = jax.tree_util.tree_structure(new_state)
    assert init_struct == new_struct, (
        f"model_fn returned a state with different pytree structure than init.\n"
        f"init state:   {init_state}\n"
        f"returned:     {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x).__name__, new_state)}\n"
        "lax.scan requires the carry structure to be invariant across iterations."
    )

    mean_arr = jnp.asarray(mean)
    assert mean_arr.shape == (), (
        f"model_fn must return a scalar mean, got shape {mean_arr.shape}"
    )
    assert bool(jnp.isfinite(mean_arr)), (
        f"model_fn returned non-finite mean: {mean_arr}"
    )
