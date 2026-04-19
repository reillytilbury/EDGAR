import numpy as np
import matplotlib.pyplot as plt

from src import utils


def plot_model_fits(
    data,
    programs_list,
    X_eval,
    save_path="",
    labels=("model_v1", "model_v2"),
):
    """
    Plot observed and predicted orientation tuning curves for up to 9 random samples.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    stimulus = data['stimulus']
    response = data['response']
    n_samples = stimulus.shape[0]

    if len(programs_list) == 1:
        colours = ["tab:red"]
    elif len(programs_list) == 2:
        colours = ["tab:green", "tab:red"]
    else:
        colours = ["tab:orange", "tab:green", "tab:red"]
    binned_colour = "deepskyblue"

    n_show = min(9, n_samples)
    idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    params_by_model = [
        utils.broadcast_params(program["params"], n_samples)
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
        x = stimulus[s]
        y_obs = response[s]
        eval_data = utils.get_data_sample(X_eval, s)
        x_eval = np.asarray(eval_data['stimulus']).reshape(-1)

        y_mean = _compute_binned_means_on_eval(x, y_obs, x_eval)
        ax.scatter(x, y_obs, s=10, c="black", alpha=0.2, label="Observed")
        ax.plot(x_eval, y_mean, color=binned_colour, linewidth=4, label="Binned mean", alpha=0.8)

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[j], s)
            y_pred = utils.call_model(model, eval_data, params)

            label = labels[j] if labels is not None and j < len(labels) else f"Model {j+1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            ax.plot(x_eval, np.asarray(y_pred).flatten(), color=colours[j % len(colours)],
                    linewidth=3, label=label, alpha=0.8)

        ax.set_xlim((float(np.min(x_eval)), float(np.max(x_eval))))
        ax.set_ylim((0, np.max(y_mean) * 1.5))
        ax.set_title(f"Cell {s}")
        ax.set_xlabel("stimulus (rad)")
        ax.set_ylabel("response")
        ax.legend(fontsize=12)

    mean_loss_parts = []
    for j, program in enumerate(programs_list):
        if "losses" in program and np.size(program["losses"]) > 0:
            mean_loss_parts.append(f"Model {j+1}: {np.mean(program['losses']):.2f}")
        else:
            mean_loss_parts.append(f"Model {j+1}: n/a")
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
