You are generating a Python visualization function for an EDGAR equation-discovery project.
The scientist is checking whether their train/test split logic (a) introduces systematic bias,
(b) samples unevenly and hurts generalisation, or (c) scrambles the structure the model should capture.

EDGAR's `load_data` returns `(X_discover, X_validate, X_eval)`:
- `X_discover = (train, test)` — discovery set, split along the within-sample (observation) axis
- `X_validate = (train, test)` — held-out validation set, same split
- Each of the four splits is a dict of arrays whose leading axis is samples.

Write `plot_split(load_data_output, save_path="split.png")`: a 2×2 grid (discover/validate × train/test)
that reveals where the split falls within a sample, that discover/validate are disjoint samples, and
that values look sane (not empty, constant, or unnormalised).

Here is the project's load_data.py:

```python
{load_data_source}
```

Work out these decisions from the source, then write the code:

1. SAMPLE SCOPE — show one representative sample. Exception: if there is only one independent
   variable, stack all samples and use (sample, that variable) as the two axes.
2. AXES — the model's independent variables. If one has structure (position, time, angle), put it on
   an axis (and sort it) so uneven sampling is visible. (train, test) splits data within a single sample.
   Represent the missing observations from each train and test as nan so that we can visually see which 
   observations are held out from either train or test. 
3. PLOT TYPE — scatter when the split dicts carry explicit coordinate keys whose spatial layout is
   meaningful; otherwise imshow. For imshow, NaN-mask held-out positions so they render white.
4. COLOR — colour by the regression target (what the equation predicts) to expose uneven target
   coverage between train/test. Share one colormap range across all four panels.
   
Examples:
- Map f: R²→R, split into a spatial checkerboard. One sample; x/y positions as axes; scatter (also
  shows sampling density); colour by the target scalar.
- N neuron populations, each (n_cells × n_stimulus), predicting held-out (cell × stim) firing. One
  population; cell_idx/stim_idx as axes (sort either if it has structure); imshow (scatter only if
  both axes have a clean ordering); colour by firing rate.

Write the complete function using this template:

```python
def _plot_panel(ax, split, title, vmin=None, vmax=None):
    # Draw one panel from one split dict (e.g. X_discover[0]): correct plot type, white masked
    # region, semantic axis labels, and a title with the name + one identifying label (sample ID /
    # count). Use the shared (vmin, vmax). If aspect="equal" is needed, pass adjustable="datalim"
    # (NOT "box", which makes panels different sizes). Return the mappable for the shared colorbar.
    ...
    return mappable


def plot_split(load_data_output, save_path="split.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    X_discover, X_validate = load_data_output[0], load_data_output[1]

    panels = [
        ("Discover TRAIN", X_discover[0]),
        ("Discover TEST", X_discover[1]),
        ("Validate TRAIN", X_validate[0]),
        ("Validate TEST", X_validate[1]),
    ]

    vmin, vmax = None, None  # shared range of the colour key across all four panels

    # constrained_layout places the shared colorbar cleanly; do NOT also call tight_layout().
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    mappable = None
    for ax, (title, split) in zip(axes.flat, panels):
        mappable = _plot_panel(ax, split, title, vmin=vmin, vmax=vmax)

    if mappable is not None:
        fig.colorbar(mappable, ax=axes, shrink=0.85, label="<colour key>")

    fig.suptitle("Data Partitions", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return axes
```

Return only the two functions. No explanation.
