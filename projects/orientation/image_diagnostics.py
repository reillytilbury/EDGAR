import diagnostic


def plot_model_prompt_image(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    return diagnostic.plot_model_fits(
        programs_df=programs_df,
        loss_function=loss_function,
        x=x,
        y=y,
        cell_selection=cell_selection,
        save_path=save_path,
        **kwargs,
    )


def plot_param_estimator_prompt_image(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    return diagnostic.plot_model_fits(
        programs_df=programs_df,
        loss_function=loss_function,
        x=x,
        y=y,
        cell_selection=cell_selection,
        save_path=save_path,
        **kwargs,
    )


def plot_seed_programs(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    return diagnostic.plot_model_fits(
        programs_df=programs_df,
        loss_function=loss_function,
        x=x,
        y=y,
        cell_selection=cell_selection,
        save_path=save_path,
        **kwargs,
    )


def plot_top_model_fits(programs_df, loss_function, x, y, cell_selection, title, save_path, **kwargs):
    return diagnostic.plot_model_fits(
        programs_df=programs_df,
        loss_function=loss_function,
        x=x,
        y=y,
        cell_selection=cell_selection,
        title=title,
        save_path=save_path,
        **kwargs,
    )


def plot_single_model_fit(model, loss_function, x, y, params, title, save_path, **kwargs):
    return diagnostic.plot_single_model_fit(
        model=model,
        loss_function=loss_function,
        x=x,
        y=y,
        params=params,
        title=title,
        save_path=save_path,
        **kwargs,
    )
