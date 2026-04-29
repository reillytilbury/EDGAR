import numpy as np
import matplotlib.pyplot as plt

from src.io import broadcast_params, call_model, slice_params


def plot_model_fits(
    data,
    programs_list,
    X_eval,
    save_path="",
    labels=("model_v1", "model_v2"),
):
    """
    Plot observed synthetic data and model predictions for up to 9 random samples.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    x_arr = data['x']
    y_arr = data['y']
    x_eval_arr = X_eval['x']
    n_samples = x_arr.shape[0]

    if len(programs_list) == 1:
        colours = ["red"]
    elif len(programs_list) == 2:
        colours = ["green", "red"]
    else:
        colours = ["purple", "green", "red"]
    binned_colour = "deepskyblue"

    n_show = min(9, n_samples)
    idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    params_by_model = [
        broadcast_params(program["params"], n_samples)
        for program in programs_list
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.reshape(3, 3)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        if i >= n_show:
            ax.axis("off")
            continue

        s = idx[i]
        x = x_arr[s]
        y_obs = y_arr[s]
        x_eval = np.asarray(x_eval_arr[s]).reshape(-1)

        y_mean = _compute_binned_means_on_eval(x, y_obs, x_eval)
        ax.scatter(x, y_obs, s=10, c="black", alpha=0.15, label="Observed")
        ax.plot(x_eval, y_mean, color=binned_colour, linewidth=4, label="Binned mean", alpha=0.8)

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = slice_params(params_by_model[j], s)
            y_pred = call_model(model, {'x': x_eval}, params)

            label = labels[j] if labels is not None and j < len(labels) else f"Model {j+1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            ax.plot(x_eval, np.asarray(y_pred).flatten(), color=colours[j % len(colours)],
                    linewidth=3, label=label, alpha=0.8)

        ax.set_xlim((float(np.min(x_eval)), float(np.max(x_eval))))
        ax.set_title(f"Sample {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=12)

    mean_loss_parts = []
    for j, program in enumerate(programs_list):
        if "losses" in program and np.size(program["losses"]) > 0:
            mean_loss_parts.append(f"Model {j+1} Loss: {np.mean(program['losses']):.2f}")
        else:
            mean_loss_parts.append(f"Model {j+1} Loss: n/a")
    plt.suptitle(f"Model Fits\n{chr(10).join(mean_loss_parts)}", fontsize=24)
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)


def _compute_binned_means_on_eval(theta, y, x_eval):
    x_eval = np.asarray(x_eval).reshape(-1)
    if x_eval.size == 0:
        return x_eval
    if x_eval.size == 1:
        return np.array([float(np.mean(y))])

    edges = np.empty(x_eval.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x_eval[:-1] + x_eval[1:])
    edges[0] = x_eval[0] - 0.5 * (x_eval[1] - x_eval[0])
    edges[-1] = x_eval[-1] + 0.5 * (x_eval[-1] - x_eval[-2])

    idx = np.digitize(theta, edges) - 1
    y_mean = np.full(x_eval.size, np.nan, dtype=float)
    for i in range(x_eval.size):
        vals = y[idx == i]
        if vals.size > 0:
            y_mean[i] = float(np.mean(vals))

    valid = np.isfinite(y_mean)
    if np.any(valid):
        y_mean = np.interp(x_eval, x_eval[valid], y_mean[valid],
                           left=float(y_mean[valid][0]), right=float(y_mean[valid][-1]))
    else:
        y_mean = np.zeros_like(x_eval, dtype=float)
    return y_mean
