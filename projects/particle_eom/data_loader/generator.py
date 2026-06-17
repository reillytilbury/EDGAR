"""Synthetic data generator for the overdamped pairwise force-law PoC.

Ground truth: dx_i/dt = sum_{j: |x_i-x_j|<r_c} [ A*sign(x_i-x_j)/(x_i-x_j)^2 + B*(x_j-x_i) ].
Mobility mu is fixed to 1 (unidentifiable from trajectory data alone — see journal).
One "recording session" = one independent simulated population (own RNG draw for
initial positions), generated `n_sessions` times so that fitted (A, B) must explain an
entire session's population at once (see journal/2026-06-16_eom_discovery_poc_spec.md,
"Design revision" section, for why this is the sample unit).
"""

from __future__ import annotations

import numpy as np


def minimum_image_diff(x: np.ndarray, L: float) -> np.ndarray:
    """Pairwise displacement x_j - x_i under periodic minimum-image convention.

    Args:
        x: Positions, shape (..., n_cells).
        L: Periodic domain length.

    Returns:
        np.ndarray: diff[..., i, j] = x_j - x_i, wrapped to [-L/2, L/2), shape
            (..., n_cells, n_cells).
    """
    diff = x[..., None, :] - x[..., :, None]
    return diff - L * np.round(diff / L)


def drop_diagonal(mat: np.ndarray) -> np.ndarray:
    """Removes the diagonal of the trailing (n, n) axes, e.g. self-interaction terms.

    Args:
        mat: Array with shape (..., n, n).

    Returns:
        np.ndarray: shape (..., n, n - 1), row i no longer contains entry [i, i].
    """
    n = mat.shape[-1]
    off_diag = ~np.eye(n, dtype=bool)
    return mat[..., off_diag].reshape(mat.shape[:-2] + (n, n - 1))


def rhs(x: np.ndarray, L: float, A: float, B: float, r_c: float, C: float = 0.0) -> np.ndarray:
    """Ground-truth dx_i/dt for every particle, given current positions.

    The pairwise kernel is repulsion + attraction + a non-reciprocal term. The first
    two are odd in the displacement `x_j - x_i`, so they obey Newton's third law and
    make the system a gradient flow that relaxes to a fixed point. The `C` term is even
    in the displacement (asymmetric attraction: pull toward right neighbors differs from
    left, since `B*d + C*|d| = (B+C)*d` for d>0 and `(B-C)*d` for d<0), which breaks
    reciprocity. A non-reciprocal kernel is no longer a gradient flow, so the dynamics
    need not freeze at equilibrium — they can sustain traveling-wave / drifting motion,
    keeping every recording window dynamically informative. C = 0 recovers the original
    reciprocal force law.

    Args:
        x: Positions, shape (n_cells,).
        L: Periodic domain length.
        A: Repulsion strength.
        B: Attraction strength.
        r_c: Interaction cutoff radius.
        C: Non-reciprocity strength (even, asymmetric-attraction term).

    Returns:
        np.ndarray: dx_i/dt for every particle, shape (n_cells,).
    """
    diff = minimum_image_diff(x, L)  # diff[i, j] = x_j - x_i
    r = np.abs(diff)
    np.fill_diagonal(r, np.inf)  # exclude self-interaction
    within_cutoff = r < r_c
    repulsion = -A * np.sign(diff) / r**2  # sign(x_i - x_j) = -sign(diff)
    attraction = B * diff
    non_reciprocal = C * np.abs(diff)  # even in diff -> breaks reciprocity
    contribution = np.where(within_cutoff, repulsion + attraction + non_reciprocal, 0.0)
    return np.sum(contribution, axis=-1)


def initial_positions(
    n_cells: int, L: float, d_star: float, rng: np.random.Generator, jitter_frac: float = 0.3
) -> np.ndarray:
    """Particles placed near equilibrium spacing with jitter, so the run starts away
    from (but close to) steady state, giving train/test windows different dynamical
    regimes to span.

    Args:
        n_cells: Number of particles.
        L: Periodic domain length.
        d_star: Target equilibrium spacing (A/B)^(1/3), used to scale the jitter.
        rng: NumPy random generator.
        jitter_frac: Jitter amplitude as a fraction of d_star.

    Returns:
        np.ndarray: Initial positions, shape (n_cells,).
    """
    base = np.arange(n_cells) * (L / n_cells)
    jitter = rng.uniform(-jitter_frac, jitter_frac, n_cells) * d_star
    return (base + jitter) % L


def simulate_session(
    n_cells: int,
    L: float,
    A: float,
    B: float,
    r_c: float,
    dt: float,
    n_steps: int,
    record_every: int,
    rng: np.random.Generator,
    noise_D: float = 0.0,
    C: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrates one independent recording session with explicit Euler-Maruyama.

    Overdamped Langevin dynamics: dx_i = v_i(x) dt + sqrt(2*noise_D*dt) * dW_i, where
    v_i = dx_i/dt is the deterministic drift from `rhs`. With noise_D > 0 the system
    never freezes at equilibrium but fluctuates around it, so every recorded window
    stays dynamically informative (vs. a pure relaxation that goes dead after the
    initial transient). noise_D = 0 recovers deterministic Euler integration.

    Args:
        n_cells: Number of particles.
        L: Periodic domain length.
        A: Repulsion strength.
        B: Attraction strength.
        r_c: Interaction cutoff radius.
        dt: Integration timestep.
        n_steps: Total number of integration steps.
        record_every: Record positions/velocities every this many steps.
        rng: NumPy random generator (one draw per session, for initial positions and,
            when noise_D > 0, the per-step Langevin noise).
        noise_D: Diffusion coefficient of the position process noise.
        C: Non-reciprocity strength passed to `rhs`.

    Returns:
        tuple: (positions, velocities), each shape (n_recorded, n_cells). velocities
            are the noiseless drift dx_i/dt = rhs evaluated at the recorded (noisy)
            positions — i.e. noise enters the positions, never the velocity target.
    """
    d_star = (A / B) ** (1.0 / 3.0)
    x = initial_positions(n_cells, L, d_star, rng)
    record_steps = set(range(0, n_steps + 1, record_every))
    noise_scale = np.sqrt(2.0 * noise_D * dt)

    recorded_positions = []
    if 0 in record_steps:
        recorded_positions.append(x.copy())
    for step in range(1, n_steps + 1):
        v = rhs(x, L, A, B, r_c, C)
        x = (x + dt * v + noise_scale * rng.normal(size=n_cells)) % L
        if step in record_steps:
            recorded_positions.append(x.copy())

    positions = np.stack(recorded_positions)  # (n_recorded, n_cells)
    velocities = np.stack([rhs(p, L, A, B, r_c, C) for p in positions])
    return positions, velocities


def generate_sessions(
    n_sessions: int,
    n_cells: int,
    L: float,
    A: float,
    B: float,
    r_c: float,
    dt: float,
    n_steps: int,
    record_every: int,
    seed: int,
    noise_D: float = 0.0,
    C: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates `n_sessions` independent recording sessions.

    Args:
        n_sessions: Number of independent sessions to simulate.
        n_cells, L, A, B, r_c, dt, n_steps, record_every, noise_D, C: See
            `simulate_session`.
        seed: Base RNG seed; session s uses its own child generator.

    Returns:
        tuple: (positions, velocities), each shape (n_sessions, n_recorded, n_cells).
            velocities are noiseless.
    """
    parent_rng = np.random.default_rng(seed)
    child_seeds = parent_rng.integers(0, 2**32 - 1, size=n_sessions)

    all_positions = []
    all_velocities = []
    for child_seed in child_seeds:
        rng = np.random.default_rng(child_seed)
        positions, velocities = simulate_session(
            n_cells, L, A, B, r_c, dt, n_steps, record_every, rng, noise_D, C
        )
        all_positions.append(positions)
        all_velocities.append(velocities)

    return np.stack(all_positions), np.stack(all_velocities)
