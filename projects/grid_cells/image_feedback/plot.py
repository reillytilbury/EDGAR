import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple

from src import utils


def plot_model_fits(
    data,
    programs_list,
    data_eval,
    save_path="",
    labels=("model_v1", "model_v2"),
    n_bins: int = 40,
    smoothing_sigma: float = 1.0,
    max_show: int = 6,
    show_colorbar: bool = False,
):
    """
    Plot data and model-predicted 2D rate maps for up to 6 random samples.
    """
    if save_path == "":
        raise ValueError("Please provide a save_path for the plot")

    pos_x = np.asarray(data['pos_x'])
    pos_y = np.asarray(data['pos_y'])
    response = np.asarray(data['response'])
    eval_pos_x = np.asarray(data_eval['pos_x'])
    eval_pos_y = np.asarray(data_eval['pos_y'])

    n_samples = pos_x.shape[0]
    n_show = min(max_show, n_samples)
    show_idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    n_models = len(programs_list)
    fig, axes = plt.subplots(n_show, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n_show))
    axes = np.atleast_2d(axes)

    params_by_model = [
        utils.broadcast_params(program["params"], n_samples)
        for program in programs_list
    ]

    for row, s in enumerate(show_idx):
        x = pos_x[s]
        y = pos_y[s]
        y_obs = response[s]
        x_domain = (float(np.min(eval_pos_x[s])), float(np.max(eval_pos_x[s])))
        y_domain = (float(np.min(eval_pos_y[s])), float(np.max(eval_pos_y[s])))

        rm_obs = _bin_to_rate_map(x, y, y_obs, n_bins=n_bins, x_domain=x_domain,
                                  y_domain=y_domain, smoothing_sigma=smoothing_sigma)
        ax = axes[row, 0]
        im = ax.imshow(rm_obs.T, origin="lower",
                       extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]], cmap="viridis")
        ax.set_title(f"Sample {s} data")
        if show_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for m_idx, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[m_idx], s)
            sample_data = {'pos_x': pos_x[s], 'pos_y': pos_y[s]}
            y_pred = utils.call_model(model, sample_data, params)
            rm_pred = _bin_to_rate_map(x, y, y_pred, n_bins=n_bins, x_domain=x_domain,
                                       y_domain=y_domain, smoothing_sigma=smoothing_sigma)

            axm = axes[row, m_idx + 1]
            imm = axm.imshow(rm_pred.T, origin="lower",
                             extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]], cmap="viridis")
            label = labels[m_idx] if labels is not None and m_idx < len(labels) else f"Model {m_idx + 1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            axm.set_title(label)
            if show_colorbar:
                fig.colorbar(imm, ax=axm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)


def _bin_to_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_bins: int = 50,
    x_domain: Tuple[float, float] = (-1.0, 1.0),
    y_domain: Tuple[float, float] = (-1.0, 1.0),
    smoothing_sigma: float = 1.0,
) -> np.ndarray:
    edges_x = np.linspace(x_domain[0], x_domain[1], n_bins + 1)
    edges_y = np.linspace(y_domain[0], y_domain[1], n_bins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=values)
    if smoothing_sigma and smoothing_sigma > 0:
        occ = gaussian_filter(occ, sigma=smoothing_sigma)
        weighted = gaussian_filter(weighted, sigma=smoothing_sigma)
    return weighted / (occ + 1e-8)
