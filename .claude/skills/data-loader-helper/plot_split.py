"""Visualise EDGAR's two-level data partition from a project's ``load_data`` output.

``load_data`` orchestrates two independent splits, and getting each one right for the
parametric equation at hand is the whole point of the data-loader-helper skill. This module
draws all four resulting partitions as a 2x2 grid of **data heatmaps** so you can eyeball —
*before* launching a run — both where each split fell and whether the values look sane
(normalised, no dead rows, no all-NaN columns).

The two splits:

1. **Across the sample axis — discover vs validate.** The first set of samples
   (``X_discover``) drives the evolutionary search; the second, disjoint set
   (``X_validate``) is held out for the final check that a discovered equation *form*
   transfers to fresh samples.
2. **Within each sample, along the in-sample axis — fit vs eval.** ``*_train`` positions are
   where each candidate's parameters are fit by gradient descent; ``*_test`` positions are
   where the fitted model is scored. Whether these are contiguous blocks or interleaved
   chunks is exactly what the masked regions make visible.

Why plot the real values rather than a flat "which partition" colour map: the masked-out
region already tells you *where* the split is, and showing the actual numbers underneath
catches normalisation/quality problems in the same glance. This mirrors the
"BZ015 Data Partitions (Normalized & Masked)" figure this helper is modelled on.

Vocabulary: the within-sample axis is the **in-sample position** axis here, not "trial" —
"trial" is overloaded and blurs the two split levels.

Typical use from the skill (or a scratch cell)::

    from plot_split import plot_split
    out = load_data(**project_params)          # (X_discover, X_validate, X_eval)
    plot_split(out, key="velocity", save_path="split.png")

If the loader splits by *reducing* arrays (train/test have fewer columns each, as in
``particle_eom``) rather than NaN-masking a full grid, pass the in-sample index arrays you
used so the four panels are reconstructed onto a shared grid with the held-out region masked
white — reproducing the reference figure's complementary-mask look and revealing
block-vs-interleave structure::

    plot_split(out, key="velocity", within_sample_index=(train_idx, test_idx),
               save_path="split.png")
"""

from __future__ import annotations

import numpy as np

_PANELS = (
    ("Discovery Train", 0, "train"),
    ("Discovery Test", 1, "test"),
    ("Validation Train", 0, "train"),
    ("Validation Test", 1, "test"),
)


def _first_data_key(split: dict) -> str:
    """Returns the first feature/response key in a split dict (skips ``_`` meta keys)."""
    for k in split:
        if not k.startswith("_"):
            return k
    raise ValueError("split dict has no non-underscore data key to read")


def _to_2d(arr, key: str):
    """Coerces a split array to a 2-D ``(n_samples, n_positions)`` view for display.

    **Limitation:** only axis 1 is shown as the in-sample/position axis. Keys with *any*
    trailing axes beyond that (e.g. ``neighbor_dx`` of shape
    ``(n_samples, n_positions, n_neighbors)``, or a layout that deliberately keeps
    ``(n_sample, n_cell, n_time, n_repeat)`` distinct) are **mean-reduced over axes 2+** into a
    single per-(sample, axis-1-position) summary. So a multi-axis layout still renders, but the
    heatmap collapses everything past axis 1 — you are *not* seeing those axes resolved. The
    caller prints exactly which keys/shapes were collapsed so this is never silent.

    Returns ``(grid, orig_shape)``; ``orig_shape`` is the array's original shape (whether or
    not anything was reduced — reduction happened iff ``len(orig_shape) > 2``).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim < 2:
        raise ValueError(f"key {key!r} must be at least 2-D; got shape {arr.shape}")
    if arr.ndim == 2:
        return arr, arr.shape
    return np.nanmean(arr, axis=tuple(range(2, arr.ndim))), arr.shape


def _expand(grid, idx, n_positions):
    """Scatters a reduced ``(n_samples, len(idx))`` grid back onto a full-width NaN canvas."""
    full = np.full((grid.shape[0], n_positions), np.nan)
    full[:, np.asarray(idx)] = grid
    return full


def plot_split(
    load_data_output,
    *,
    key: str | None = None,
    within_sample_index: tuple | None = None,
    max_samples: int | None = None,
    cmap: str = "viridis",
    suptitle: str = "Data Partitions (Masked)",
    save_path: str = "",
    axes=None,
):
    """Draws the four partitions (discover/validate x fit/eval) as masked data heatmaps.

    Each panel shows one partition's values for ``key``; entries not belonging to that
    partition are masked and render white. Panel titles report the array shape and the
    finite-value mean/std, so a partition that is accidentally empty, constant, or unnormalised
    stands out immediately.

    Args:
        load_data_output: The ``(X_discover, X_validate, X_eval)`` tuple returned by a
            project's ``load_data`` (``X_eval`` is ignored here).
        key: Which data key to display. Defaults to the first non-``_`` key. Keys with
            trailing feature axes are mean-reduced to ``(n_samples, n_positions)``.
        within_sample_index: Optional ``(train_index, test_index)`` — the 1-D arrays of
            original in-sample positions used to split. Pass these when the loader *reduces*
            arrays per split (different column counts) so the panels are reconstructed onto a
            shared full-width grid with the held-out columns masked. Omit when the loader
            already returns full-width NaN-masked arrays.
        max_samples: Cap on the number of sample rows drawn per panel (useful for very tall
            data). ``None`` draws all; start with a small number to keep the figure light.
        cmap: Matplotlib colormap for the data values.
        suptitle: Figure-level title.
        save_path: If non-empty, the figure is written here.
        axes: Optional 2x2 array of existing Axes to draw into.

    Returns:
        The 2x2 numpy array of matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    X_discover, X_validate = load_data_output[0], load_data_output[1]
    splits = {
        "discover": {"train": X_discover[0], "test": X_discover[1]},
        "validate": {"train": X_validate[0], "test": X_validate[1]},
    }
    key = key or _first_data_key(splits["discover"]["train"])

    reconstruct = within_sample_index is not None
    if reconstruct:
        train_idx, test_idx = (np.asarray(a) for a in within_sample_index)
        n_positions = int(max(train_idx.max(), test_idx.max())) + 1
        idx_for = {"train": train_idx, "test": test_idx}

    reduced_shapes: dict[str, tuple] = {}

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        axes = np.asarray(axes)
        fig = axes.flat[0].get_figure()

    for (label, _, kind), ax in zip(_PANELS, axes.flat):
        group = "discover" if label.startswith("Discovery") else "validate"
        grid, orig_shape = _to_2d(splits[group][kind][key], key)
        reduced = len(orig_shape) > 2
        if reduced:
            reduced_shapes[label] = orig_shape
        if reconstruct:
            grid = _expand(grid, idx_for[kind], n_positions)
        if max_samples is not None:
            grid = grid[:max_samples]

        masked = np.ma.masked_invalid(grid)
        finite = grid[np.isfinite(grid)]
        mean = float(finite.mean()) if finite.size else float("nan")
        std = float(finite.std()) if finite.size else float("nan")

        im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="none")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=key)
        ax.set_title(
            f"{label}\nShape: {grid.shape}\nMean: {mean:.4g}, Std: {std:.4g}",
            fontsize=10,
        )
        ax.set_xlabel(
            "in-sample position" + ("  (mean over feature axes)" if reduced else "")
        )
        ax.set_ylabel("sample")

    if reduced_shapes:
        shape = next(iter(reduced_shapes.values()))
        print(
            f"[plot_split] NOTE: key {key!r} has shape {shape} (> 2-D). The heatmap shows "
            f"axis 1 (size {shape[1]}) as the in-sample/position axis and MEAN-REDUCES "
            f"axes 2+ ({shape[2:]}) into it. You are NOT seeing those axes resolved — only "
            f"a per-(sample, axis-1) average. If those trailing axes are scientifically "
            f"distinct (e.g. cell vs time vs repeat), pick the axis you most want to "
            f"inspect, move it to axis 1, and re-plot (or plot each separately)."
        )

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return axes


def _demo(save_path: str = "split_demo.png"):
    """Self-test: fabricate a reduced-convention ``load_data`` output and render it.

    Uses a chunked-interleave in-sample split (the recommended default for an autocorrelated
    time axis) so the reconstructed masking shows alternating held-out stripes — a quick way
    to confirm the figure renders without needing a real project.
    """
    rng = np.random.default_rng(0)
    n_disc, n_val, n_pos = 200, 80, 120

    chunks = np.array_split(np.arange(n_pos), 6)
    train_idx = np.concatenate(chunks[0::2])
    test_idx = np.concatenate(chunks[1::2])

    base = np.abs(rng.standard_normal((max(n_disc, n_val), n_pos))) * 0.02

    def split(n_samples, idx):
        return {"response": base[:n_samples][:, idx]}

    out = (
        (split(n_disc, train_idx), split(n_disc, test_idx)),
        (split(n_val, train_idx), split(n_val, test_idx)),
        None,
    )
    plot_split(
        out,
        key="response",
        within_sample_index=(train_idx, test_idx),
        save_path=save_path,
    )
    print(f"wrote {save_path}")


if __name__ == "__main__":
    _demo()
