import numpy as np
import matplotlib.pyplot as plt


def plot_model_fits(
    data,
    parent_programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    *,
    rng: np.random.Generator,
):
    """
    Predicted-vs-observed dx_i/dt scatter for a few example sessions, for each parent
    program. No (cell, time) un-flattening is needed for this view — each point is one
    (cell, time) trial within a session.

    Two call sites exist (`edgar/io/plotting.py`): `generate_feedback_image` calls with
    just (data, parents, save_path) for live LLM feedback; `generate_program_fits`
    additionally passes `losses`/`sample_losses`/`program_names`/`params` (e.g. to plot
    the same program's init vs. final fit side by side) for the dashboard. All four
    default to the corresponding `Program` attributes when not given.

    Args:
        data: X_disc_train dict of JAX arrays. data['neighbor_dx'] shape
            (n_sessions, n_trials, n_neighbors), data['velocity'] shape
            (n_sessions, n_trials).
        parent_programs: list of Program objects, each with .compile_model().
        save_path: file path (not directory) to save the figure.
        losses: optional list of scalar loss values, one per program in
            `parent_programs`; defaults to program.program_losses.discover.final.
        sample_losses: optional list of per-session loss arrays (shape (n_sessions,)
            or None), one per program; defaults to program.sample_losses.
        program_names: optional list of display names, one per program; defaults to
            program.name.
        params: optional list of per-session param dicts, one per program; defaults
            to program.params.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in parent_programs]
    if sample_losses is None:
        sample_losses = [p.sample_losses for p in parent_programs]
    if program_names is None:
        program_names = [p.name for p in parent_programs]
    if params is None:
        params = [p.params for p in parent_programs]

    neighbor_dx = np.asarray(data["neighbor_dx"])  # (n_sessions, n_trials, n_neighbors)
    velocity = np.asarray(data["velocity"])  # (n_sessions, n_trials)
    n_sessions = neighbor_dx.shape[0]

    n_show = min(4, n_sessions)
    session_indices = rng.choice(n_sessions, size=n_show, replace=False)
    colours = ["tab:red", "tab:green", "tab:orange", "tab:purple"]

    model_fns = [program.compile_model() for program in parent_programs]

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5), squeeze=False)
    axes = axes[0]

    for i, s in enumerate(session_indices):
        ax = axes[i]
        sample_data = {"neighbor_dx": neighbor_dx[s], "velocity": velocity[s]}
        y_obs = velocity[s]

        lims = [y_obs.min(), y_obs.max()]
        ax.plot(
            lims,
            lims,
            color="black",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
            label="y=x",
        )

        for j, model_fn in enumerate(model_fns):
            params_s = {k: np.asarray(v[s]) for k, v in params[j].items()}
            y_pred = np.asarray(model_fn(sample_data, params_s))
            sl = sample_losses[j][s] if sample_losses[j] is not None else None
            label = (
                f"{program_names[j]} (loss={sl:.4f})"
                if sl is not None
                else program_names[j]
            )
            ax.scatter(
                y_obs,
                y_pred,
                s=8,
                alpha=0.4,
                color=colours[j % len(colours)],
                label=label,
            )

        ax.set_title(f"Session {s}")
        ax.set_xlabel("observed dx/dt")
        ax.set_ylabel("predicted dx/dt")
        ax.legend(fontsize=8)

    title_parts = [
        f"{program_names[j]}: loss={losses[j]:.4f}"
        if losses[j] is not None
        else f"{program_names[j]}: loss=n/a"
        for j in range(len(parent_programs))
    ]
    plt.suptitle("  |  ".join(title_parts), fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)
