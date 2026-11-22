"""Prepare the pinned Stringer recording and persist trial and cell indices."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DATASET_NAME = "gratings_drifting_GT1_2019_04_12_1.npy"
DATASET_SHA256 = "dbfbb9cd7764bd15feddf8c0d79a8e17a51d956a978c01f87e54b472b42e1b57"
SPLIT_SEED = 42
TRIAL_SEED = 0
ACTIVITY_THRESHOLD = 0.4
CONCENTRATION_THRESHOLD = 0.55


def load_dataset(path, extract_response):
    # Verify the pinned recording before loading the pickled NumPy dictionary.
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
        if sha256.hexdigest() != DATASET_SHA256:
            raise ValueError("File checksum differs from the pinned Figshare recording.")
        stream.seek(0)
        recording = np.load(stream, allow_pickle=True).item()
    return extract_response(recording, n_pcs=0), np.asarray(recording["istim"]), sha256.hexdigest()


def prepare(response, angles, protocol):
    if response.ndim != 2 or angles.shape != (response.shape[1],):
        raise ValueError("Expected cells × trials responses and one angle per trial.")
    if not np.isfinite(response).all() or not np.isfinite(angles).all():
        raise ValueError("Recording contains nonfinite values.")

    active = response > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        concentration = np.abs(
            np.sum(np.exp(2j * angles)[None, :] * response, axis=1) / np.sum(response, axis=1)
        )
    cells = np.flatnonzero(
        (np.mean(active.astype(np.float32), axis=1) > ACTIVITY_THRESHOLD)
        & (concentration > CONCENTRATION_THRESHOLD)
    )
    trial_count = int(response.shape[1] * ACTIVITY_THRESHOLD)
    if trial_count < 2 or cells.size < (4 if protocol == "strict-holdout-v1" else 2):
        raise ValueError("Too few eligible cells or trials for the requested split.")

    trials = np.stack([np.flatnonzero(active[cell])[:trial_count] for cell in cells])
    y = jnp.asarray(response[cells[:, None], trials].astype(np.float64))
    y = 100 * y / jnp.linalg.norm(y, axis=1, keepdims=True)
    x = jnp.asarray(angles[trials].astype(np.float64))

    cell_order = np.asarray(jax.random.permutation(jax.random.PRNGKey(SPLIT_SEED), jnp.arange(cells.size)))
    cut = cells.size // 2
    groups = {"discovery": cell_order[:cut]}
    if protocol == "strict-holdout-v1":
        validation_end = cut + cells.size // 4
        groups["selection"] = cell_order[cut:validation_end]
        groups["locked_test"] = cell_order[validation_end:]
    else:
        groups["selection"] = cell_order[cut:]

    trial_order = np.asarray(
        jax.random.permutation(jax.random.PRNGKey(TRIAL_SEED), jnp.arange(trial_count))
    )
    indices = {
        "eligible_cell_ids": cells,
        "recording_trial_ids": trials,
        "fit_trial_columns": trial_order[:trial_count // 2],
        "score_trial_columns": trial_order[trial_count // 2:],
        **{f"{name}_cell_ids": cells[rows] for name, rows in groups.items()},
    }
    return x, y, groups, indices


def synthetic_recording():
    """Small deterministic input exercising the same preparation and evaluator."""
    rng = np.random.default_rng(412)
    angles = np.tile(np.linspace(0, 2 * np.pi, 36, endpoint=False), 4)
    centers = np.linspace(0, 2 * np.pi, 16, endpoint=False)[:, None]
    difference = (angles - centers + np.pi) % (2 * np.pi) - np.pi
    response = 0.01 + np.exp(-0.5 * (difference / 0.35) ** 2)
    response *= 1 + rng.normal(0, 0.02, response.shape)
    return response, angles
