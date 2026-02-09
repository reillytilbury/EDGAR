import diagnostic

SIGMA = 2.0
N_BINS = 24


def plot_rate_maps(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    return diagnostic.plot_model_fits(
        programs_df=programs_df,
        loss_function=loss_function,
        x=x,
        y=y,
        cell_selection=cell_selection,
        save_path=save_path,
        **kwargs,
    )

