import inspect
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_rate_maps(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    def _stimuli_trials_first(stimuli):
        stim = np.asarray(stimuli)
        if stim.ndim == 1:
            return stim
        if stim.ndim != 2:
            raise ValueError(f"stimuli must be 1D or 2D, got {stim.ndim}D")
        trials_first = stim.T
        if trials_first.shape[1] == 1:
            return trials_first[:, 0]
        return trials_first

    def _to_time_series(arr, n_trials):
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return np.full(n_trials, float(arr))
        if arr.ndim == 1:
            return arr
        if arr.shape[0] == n_trials:
            return np.mean(arr, axis=1)
        if arr.shape[-1] == n_trials:
            return np.mean(arr, axis=0)
        return np.mean(arr, axis=-1)

    stim_model = _stimuli_trials_first(x)
    if stim_model.ndim == 1:
        n_trials = stim_model.shape[0]
    else:
        n_trials = stim_model.shape[0]

    models = programs_df["program"].tolist()
    params = programs_df["params"].tolist()
    n_models = len(models)
    labels = kwargs.get("labels", [f"model {i + 1}" for i in range(n_models)])
    trace_len = int(kwargs.get("trace_len", min(200, n_trials)))
    trace_start = int(kwargs.get("trace_start", 0))
    trace_end = min(trace_start + trace_len, n_trials)

    def _unpack_population_params(params_flat, n_source, n_target, n_model_params):
        params_flat = np.asarray(params_flat).ravel()
        if n_model_params == 1:
            A = params_flat.reshape(n_source, n_target)
            return (A,)
        if n_model_params == 2:
            size_A = n_source * n_target
            A = params_flat[:size_A].reshape(n_source, n_target)
            b = params_flat[size_A:size_A + n_target]
            return (A, b)
        if n_model_params == 4:
            size_A = n_source * n_target
            A = params_flat[:size_A].reshape(n_source, n_target)
            quad = params_flat[size_A:].reshape(3, n_target)
            return (A, quad[0], quad[1], quad[2])
        # fallback: split into vectors
        parts = []
        idx = 0
        for _ in range(n_model_params):
            parts.append(params_flat[idx:idx + n_target])
            idx += n_target
        return tuple(parts)

    def _call_model(model, stim, params_row, y_for_shape):
        n_model_params = len(inspect.signature(model).parameters) - 1
        if n_model_params == 0:
            return model(stim)
        if pop_mode:
            n_source = stim.shape[1] if stim.ndim > 1 else 1
            n_target = y_for_shape.shape[1] if y_for_shape.ndim == 2 else 1
            parts = _unpack_population_params(params_row, n_source, n_target, n_model_params)
            return model(stim, *parts)
        params_row = np.asarray(params_row)
        if n_model_params == 1:
            return model(stim, params_row)
        if params_row.ndim == 0:
            return model(stim, params_row)
        if params_row.size == n_model_params:
            return model(stim, *params_row)
        return model(stim, params_row)

    pop_mode = y.ndim == 3 and y.shape[0] == 1
    n_cells = len(cell_selection)
    if n_cells == 0 or n_models == 0:
        logger = kwargs.get("logger")
        if logger is not None:
            logger.info("Peer prediction diagnostics skipped: no cells or models to plot.")
        return

    n_rows = n_cells
    n_cols = n_models + 1
    fig_w = 3.8 * n_cols
    fig_h = 2.6 * n_rows
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    if pop_mode:
        y_full = np.asarray(y[0])
        t = np.arange(trace_start, trace_end)
        for i, c in enumerate(cell_selection):
            y_series = y_full[:, c]
            ax[i, 0].plot(t, y_series[trace_start:trace_end], color="black", lw=1)
            ax[i, 0].set_title(f"Target {c} | data")
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])

            for m, model in enumerate(models):
                params_row = np.asarray(params[m])[0]
                pred_full = np.asarray(_call_model(model, stim_model, params_row, y_full))
                pred_series = pred_full[:, c] if pred_full.ndim == 2 else _to_time_series(pred_full, n_trials)
                loss_val = loss_function(jnp.asarray(pred_series), jnp.asarray(y_series))
                loss_scalar = float(jnp.mean(loss_val))
                ax[i, m + 1].plot(t, y_series[trace_start:trace_end], color="gray", lw=0.8, alpha=0.6)
                ax[i, m + 1].plot(t, pred_series[trace_start:trace_end], color="tab:blue", lw=1)
                ax[i, m + 1].set_title(f"{labels[m]} | loss {loss_scalar:.2f}")
                ax[i, m + 1].set_xticks([])
                ax[i, m + 1].set_yticks([])
    else:
        for i, c in enumerate(cell_selection):
            y_raw = np.asarray(y[c])
            y_series = _to_time_series(y_raw, n_trials)
            t = np.arange(trace_start, trace_end)
            ax[i, 0].plot(t, y_series[trace_start:trace_end], color="black", lw=1)
            ax[i, 0].set_title(f"Cell {c} | data")
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])

            for m, model in enumerate(models):
                params_c = np.asarray(params[m])[c]
                pred_raw = np.asarray(_call_model(model, stim_model, params_c, y_raw))
                if pred_raw.shape == y_raw.shape:
                    pred_for_loss = pred_raw
                    y_for_loss = y_raw
                else:
                    pred_for_loss = _to_time_series(pred_raw, n_trials)
                    y_for_loss = _to_time_series(y_raw, n_trials)
                loss_val = loss_function(jnp.asarray(pred_for_loss), jnp.asarray(y_for_loss))
                loss_scalar = float(jnp.mean(loss_val))
                pred_series = _to_time_series(pred_raw, n_trials)
                ax[i, m + 1].plot(t, y_series[trace_start:trace_end], color="gray", lw=0.8, alpha=0.6)
                ax[i, m + 1].plot(t, pred_series[trace_start:trace_end], color="tab:blue", lw=1)
                ax[i, m + 1].set_title(f"{labels[m]} | loss {loss_scalar:.2f}")
                ax[i, m + 1].set_xticks([])
                ax[i, m + 1].set_yticks([])

    title = kwargs.get("title", "")
    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=kwargs.get("dpi", 120))
    else:
        plt.show()
    plt.close(fig)
