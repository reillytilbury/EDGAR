import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from src import utils


def plot_model_fits(
    data,
    programs_list,
    data_eval,
    save_path="",
    labels=("model_v1", "model_v2"),
):
    """
    Plot observed and predicted place-cell rate maps for up to 9 random samples.
    """
    if save_path == "":
        raise ValueError("Please provide a save_path for the plot")

    pos_x = np.asarray(data["pos_x"])
    pos_y = np.asarray(data["pos_y"])
    response = np.asarray(data["response"])
    pos_x_eval = np.asarray(data_eval["pos_x"])
    pos_y_eval = np.asarray(data_eval["pos_y"])

    n_samples = pos_x.shape[0]
    n_show = min(9, n_samples)
    show_idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    n_models = len(programs_list)
    fig, axes = plt.subplots(n_show, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n_show))
    axes = np.atleast_2d(axes)

    params_by_model = [
        utils.broadcast_params(program["params"], n_samples)
        for program in programs_list
    ]

    n_bins = min(50, pos_x_eval.shape[1])

    for row, sample_idx in enumerate(show_idx):
        x = pos_x[sample_idx]
        y = pos_y[sample_idx]
        y_obs = response[sample_idx]
        x_domain = (float(np.min(pos_x_eval[sample_idx])), float(np.max(pos_x_eval[sample_idx])))
        y_domain = (float(np.min(pos_y_eval[sample_idx])), float(np.max(pos_y_eval[sample_idx])))

        rm_obs = _bin_to_rate_map(x, y, y_obs, n_bins=n_bins, x_domain=x_domain, y_domain=y_domain)
        ax = axes[row, 0]
        im = ax.imshow(rm_obs.T, origin="lower",
                       extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]], cmap="viridis")
        ax.set_title(f"Sample {sample_idx} data")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for m_idx, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[m_idx], sample_idx)
            sample_data = {"pos_x": pos_x[sample_idx], "pos_y": pos_y[sample_idx]}
            y_pred = utils.call_model(model, sample_data, params)
            rm_pred = _bin_to_rate_map(x, y, y_pred, n_bins=n_bins, x_domain=x_domain, y_domain=y_domain)

            axm = axes[row, m_idx + 1]
            imm = axm.imshow(rm_pred.T, origin="lower",
                             extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]], cmap="viridis")
            label = labels[m_idx] if labels is not None and m_idx < len(labels) else f"Model {m_idx + 1}"
            if "losses" in program:
                label += f", loss={program['losses'][sample_idx]:.2f}"
            axm.set_title(label)
            fig.colorbar(imm, ax=axm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bin_to_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_bins: int = 50,
    x_domain: Tuple[float, float] = (-1.0, 1.0),
    y_domain: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    edges_x = np.linspace(x_domain[0], x_domain[1], n_bins + 1)
    edges_y = np.linspace(y_domain[0], y_domain[1], n_bins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=values)
    return weighted / (occ + 1e-8)
