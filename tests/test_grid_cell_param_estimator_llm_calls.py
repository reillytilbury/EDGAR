import numpy as np
import matplotlib.pyplot as plt
import io 
import os
from pathlib import Path
from datetime import datetime 
from dotenv import load_dotenv

from scipy import signal, ndimage
from scipy.spatial import KDTree
from scipy.optimize import curve_fit

from typing import Callable, List
import jax
import jax.numpy as jnp
from functools import partial

from google import genai
from google.genai import types, chats

from src.loss_functions import quadratic_loss
from experiments.grid_cells.diagnostics import _bin_to_rate_map

import src.utils as utils
from src.hypothesis_engine import compute_initial_params
# get the python functions in the experiments/grid_cells/seed_programs.py 
import experiments.grid_cells.seed_programs as grid_seed_programs
from experiments.grid_cells.data_parser import compute_rate_map, load_and_process_data

# Re-using helper functions from v3
def fit_2d_gaussian(patch):
    h, w = patch.shape
    y_coords, x_coords = np.mgrid[:h, :w]
    def gaussian_2d(coords, amplitude, y0, x0, sigma):
        y, x = coords
        g = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        return g.ravel()
    try:
        popt, _ = curve_fit(f=gaussian_2d, xdata=(y_coords.ravel(), x_coords.ravel()), ydata=patch.ravel(),
                            p0=[np.max(patch), h/2, w/2, min(h,w)/4],
                            bounds=([0, 0, 0, 0.1], [np.inf, h, w, min(h,w)]))
        return popt[3]
    except (RuntimeError, ValueError): return min(h,w)/4.0

def generate_lattice_points(lam, theta, phi_x, phi_y, extent=1.5):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v1 = R @ np.array([lam, 0]); v2 = R @ np.array([0.5 * lam, 0.5 * np.sqrt(3) * lam])
    points = []
    n_range = int(np.ceil(extent / lam)) + 2
    for n in range(-n_range, n_range + 1):
        for m in range(-n_range, n_range + 1):
            pt = n * v1 + m * v2 + np.array([phi_x, phi_y])
            if -extent < pt[0] < extent and -extent < pt[1] < extent:
                points.append(pt)
    return np.array(points)

def calculate_grid_score(sac, lam_px):
    """
    Calculates a 'gridness' score from the spatial autocorrelation map (SAC).
    It measures the 6-fold rotational symmetry in a ring around the center.
    """
    if lam_px <= 0: return 0.0
    
    cy, cx = sac.shape[0] // 2, sac.shape[1] // 2
    # Define a ring mask where the 6 peaks should be
    inner_radius = lam_px * 0.65
    outer_radius = lam_px * 1.35
    y, x = np.ogrid[-cy:sac.shape[0]-cy, -cx:sac.shape[1]-cx]
    mask = (x**2 + y**2 >= inner_radius**2) & (x**2 + y**2 <= outer_radius**2)
    
    ring_data = sac[mask]
    if ring_data.size == 0: return 0.0
        
    correlations = []
    # Correlate the ring with rotated versions of itself
    for angle in [30, 60, 90, 120, 150]:
        rotated_sac = ndimage.rotate(sac, angle, reshape=False, mode='constant', cval=np.mean(sac))
        rotated_ring_data = rotated_sac[mask]
        corr = np.corrcoef(ring_data, rotated_ring_data)[0, 1]
        correlations.append(corr)
    
    # Grid score = (mean of correlations at 60/120 deg) - (mean of others)
    score = (correlations[1] + correlations[3]) / 2.0 - (correlations[0] + correlations[2] + correlations[4]) / 3.0
    return max(0, score)

def parameter_estimator_v4(X, firing_rates):
    """
    Estimator v4: Sanity-Checked Lattice Fitting.

    This version adds a critical pre-processing step:
    1.  **Grid Score Calculation:** Before attempting a fit, it calculates a grid 
        score from the SAC to quantify if the data has hexagonal periodicity.
    2.  **Conditional Fitting:** If the score is below a threshold, it concludes the
        cell is not a grid cell and returns a simple baseline model (graceful failure).
    3.  **Robust Fitting:** If the score is high, it proceeds with the robust
        lattice fitting and field averaging logic from v3, with a more robust
        peak detection threshold.
    """
    x = X[0]; y = X[1]
    
    # --- 1. Rate Map Generation ---
    nbins = 60
    range_lims = [[-1, 1], [-1, 1]]
    heatmap, _, _ = np.histogram2d(x, y, bins=nbins, range=range_lims, weights=firing_rates)
    occupancy, x_edges, y_edges = np.histogram2d(x, y, bins=nbins, range=range_lims)
    
    ratemap = np.divide(heatmap, occupancy, out=np.zeros_like(heatmap), where=occupancy > 1e-5)
    ratemap_smooth = ndimage.gaussian_filter(ratemap, sigma=2.0)
    pixel_scale = 2.0 / nbins

    # --- 2. SAC and Initial Lambda Estimate ---
    rm_centered = ratemap_smooth - np.mean(ratemap_smooth)
    sac = signal.fftconvolve(rm_centered, rm_centered[::-1, ::-1], mode='same')
    cy, cx = sac.shape[0] // 2, sac.shape[1] // 2
    local_max_sac = ndimage.maximum_filter(sac, size=7) == sac
    peaks_y, peaks_x = np.where(local_max_sac & (sac > 0.15 * np.max(sac)))
    dists = np.sqrt((peaks_x - cx)**2 + (peaks_y - cy)**2)
    valid_dists = dists[dists > 5]
    lam_px = np.min(valid_dists) if len(valid_dists) > 0 else 0
    
    # --- 3. CRITICAL STEP: Gridness Sanity Check ---
    grid_score = calculate_grid_score(sac, lam_px)
    GRID_SCORE_THRESHOLD = 0.25

    if grid_score < GRID_SCORE_THRESHOLD:
        # Data is not grid-like. Return a simple baseline model.
        mean_rate = np.mean(firing_rates)
        return np.array([0.5, 0.0, 0.0, 0.0, mean_rate, 0.0, 0.1]) # lam, theta, phi, baseline, amplitude=0, sigma
    
    # --- 4. Proceed with Fitting for Grid-Like Cells ---
    lam = lam_px * pixel_scale
    
    # More robust peak detection using a high percentile
    if np.any(ratemap_smooth > 0):
        peak_threshold = np.percentile(ratemap_smooth[ratemap_smooth > 0], 90)
    else:
        peak_threshold = 0
    local_max_rm = ndimage.maximum_filter(ratemap_smooth, size=5) == ratemap_smooth
    peak_mask = local_max_rm & (ratemap_smooth > peak_threshold)
    peak_coords_px = np.argwhere(peak_mask)
    
    if len(peak_coords_px) < 3:
        mean_rate = np.mean(firing_rates)
        return np.array([lam, 0.0, 0.0, 0.0, mean_rate, 0.0, 0.1]) # Fallback if too few peaks

    peak_coords_spatial = (peak_coords_px[:, ::-1] + 0.5) * pixel_scale - 1.0
    
    # The rest is the robust lattice fitting from v3
    best_params = {'theta': 0, 'phi_x': 0, 'phi_y': 0}; min_error = np.inf
    for theta_candidate in np.linspace(0, np.pi / 3, 20):
        c, s = np.cos(theta_candidate), np.sin(theta_candidate)
        v1 = R @ np.array([lam, 0]); v2 = R @ np.array([0.5 * lam, 0.5 * np.sqrt(3) * lam])
        B_inv = np.linalg.inv(np.vstack([v1, v2]).T)
        
        # We can average phase estimates from multiple peaks to make it more robust
        phi_estimates = []
        for peak in peak_coords_spatial:
            n, m = np.round(B_inv @ peak)
            phi_est = peak - (n*v1 + m*v2)
            phi_estimates.append(phi_est)
        phi_candidate = np.mean(phi_estimates, axis=0)

        ideal_lattice = generate_lattice_points(lam, theta_candidate, phi_candidate[0], phi_candidate[1])
        if len(ideal_lattice) == 0: continue
        
        tree = KDTree(ideal_lattice)
        dist, _ = tree.query(peak_coords_spatial)
        error = np.mean(dist**2)

        if error < min_error:
            min_error = error
            best_params['theta'], best_params['phi_x'], best_params['phi_y'] = theta_candidate, phi_candidate[0], phi_candidate[1]

    theta, phi_x, phi_y = best_params['theta'], best_params['phi_x'], best_params['phi_y']

    baseline = np.median(ratemap[ratemap > 0]) if np.any(ratemap > 0) else 0
    peak_values = ratemap_smooth[peak_coords_px[:, 0], peak_coords_px[:, 1]]
    amplitude = np.mean(peak_values) - baseline

    patch_size_px = int(lam_px * 0.8); patch_size_px += (patch_size_px % 2 == 0)
    half_patch = patch_size_px // 2
    avg_field = np.zeros((patch_size_px, patch_size_px)); n_fields = 0
    
    for r, c in peak_coords_px:
        if (half_patch < r < nbins - half_patch and half_patch < c < nbins - half_patch):
            patch = ratemap[r-half_patch:r+half_patch+1, c-half_patch:c+half_patch+1]
            avg_field += (patch - baseline)
            n_fields += 1
    
    sigma = 0.08 # Fallback sigma
    if n_fields > 0:
        avg_field = np.maximum(0, avg_field / n_fields)
        if np.any(avg_field > 0):
            sigma_px = fit_2d_gaussian(avg_field)
            sigma = sigma_px * pixel_scale

    return np.array([np.clip(lam, 0.1, 2.0), np.clip(theta, 0, np.pi / 3), phi_x, phi_y,
                     np.clip(baseline, 0, None), np.clip(amplitude, 0, None), np.clip(sigma, 0.01, 0.5)])

def compute_total_loss(model, loss_function, inputs, params, response):
    """
    Compute total loss across all cells without storing intermediate predictions.
    Memory-efficient: computes loss per cell and immediately reduces.
    
    Returns: (mean_loss, per_cell_losses)
    """
    @jax.jit
    def loss_one_cell(inp_one, p_one, true_one):
        pred_one = model(inp_one, *p_one)
        return jnp.mean(loss_function(pred_one, true_one))
    
    # vmap over cells - JAX will fuse the operations and not store all predictions
    per_cell_losses = jax.vmap(loss_one_cell, in_axes=(0, 0, 0))(inputs, params, response)
    mean_loss = jnp.mean(per_cell_losses)
    return mean_loss, per_cell_losses

def compute_predictions_for_selection(model, inputs_sel, params_sel):
    """
    Compute predictions only for selected cells.
    
    Args:
        model: the model function
        inputs_sel: (n_selected, 2, n_trials)
        params_sel: (n_selected, n_params)
    
    Returns: (n_selected, n_trials) predictions
    """
    def predict_one(inp_one, p_one):
        return model(inp_one, *p_one)
    
    return jax.vmap(predict_one, in_axes=(0, 0))(inputs_sel, params_sel)

def plot_model_fit(
    model,
    params,
    per_cell_losses,      # jax.Array: (n_cells,) - pre-computed losses for all cells
    inputs,               # jax.Array: (n_cells, 2, n_trials)
    response,             # jax.Array: (n_cells, n_trials)
    sample_selection,     # list/np/jax array of indices
    smoothing_sigma=0.0,  # only use for real rates, never for model predictions (which are noiseless)    
    ):
    """
    Plot model fit for selected cells.
    
    Args:
        model: the model function
        params: (n_cells, n_params) array of parameters
        per_cell_losses: (n_cells,) pre-computed losses for all cells (from compute_total_loss)
        inputs: (n_cells, 2, n_trials) input data
        response: (n_cells, n_trials) true firing rates
        sample_selection: indices of cells to plot
        smoothing_sigma: smoothing for actual rate maps
    
    Returns:
        png_bytes: PNG image as bytes
    """
    sample_selection = jnp.asarray(sample_selection, dtype=jnp.int32)

    n_samples = int(sample_selection.shape[0])
    n_cols = 4
    n_rows = n_samples

    def normalise_map(m):
        m = m.astype(float)
        m = m - np.nanmin(m)
        maxv = np.nanmax(m)
        if maxv > 0:
            m = m / maxv
        return m

    # --- Gather selected cells ---
    inputs_sel = inputs[sample_selection]        # (n_samples, 2, n_trials)
    response_sel = response[sample_selection]    # (n_samples, n_trials)
    params_sel = params[sample_selection]        # (n_samples, ...)

    # --- Compute predictions ONLY for selected cells ---
    pred_firing_rates_sel = compute_predictions_for_selection(model, inputs_sel, params_sel)
    
    # --- Get losses for selected cells ---
    loss_sel = per_cell_losses[sample_selection]     # (n_samples,) 

    # --- Convert to NumPy for plotting ---
    inputs_np = np.asarray(inputs_sel)
    true_np = np.asarray(response_sel)
    pred_np = np.asarray(pred_firing_rates_sel)
    loss_np = np.asarray(loss_sel)
    sel_np = np.asarray(sample_selection)

    # --- ONE figure, ONE axes grid ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # If n_rows == 1, axes has shape (2,), so normalise it
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        cell_idx = int(sel_np[i])

        x = inputs_np[i, 0, :]
        y = inputs_np[i, 1, :]

        unsmoothed_actual_rate_map = _bin_to_rate_map(x, y, true_np[i], smoothing_sigma=0.0)
        smoothed_actual_rate_map = _bin_to_rate_map(x, y, true_np[i], smoothing_sigma=smoothing_sigma)
        pred_rate_map   = _bin_to_rate_map(x, y, pred_np[i], smoothing_sigma=0.0)

        # Actual - no smoothing
        im0 = axes[i, 0].imshow(
            unsmoothed_actual_rate_map.T,
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="viridis",
        )
        axes[i, 0].set_title(f"Unsmoothed Actual (cell {cell_idx})")
        axes[i, 0].set_xlabel("X Position (normalized)")
        axes[i, 0].set_ylabel("Y Position (normalized)")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Actual - smoothed
        im0 = axes[i, 1].imshow(
            smoothed_actual_rate_map.T,
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="viridis",
        )
        axes[i, 1].set_title(f"Smoothed Actual (sigma={smoothing_sigma}, cell {cell_idx})")
        axes[i, 1].set_xlabel("X Position (normalized)")
        axes[i, 1].set_ylabel("Y Position (normalized)")
        fig.colorbar(im0, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Pred
        im1 = axes[i, 2].imshow(
            pred_rate_map.T ,
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="viridis",
        )
        axes[i, 2].set_title(f"Predicted | Loss: {loss_np[i]:.4f}")
        axes[i, 2].set_xlabel("X Position (normalized)")
        axes[i, 2].set_ylabel("Y Position (normalized)")
        fig.colorbar(im1, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # ---------- column 2: overlay ----------
        smoothed_actual_norm = normalise_map(smoothed_actual_rate_map)
        pred_norm = normalise_map(pred_rate_map)

        # Choose a shared scale per cell (keeps relative amplitude meaningful)
        vmin = 0.0  # often sensible for firing rates
        vmax = max(float(np.nanmax(smoothed_actual_rate_map)), float(np.nanmax(pred_rate_map)))
        vmax = max(vmax, 1e-8)  # avoid divide-by-zero

        def scale_shared(m):
            m = np.asarray(m, dtype=float)
            m = (m - vmin) / (vmax - vmin)
            return np.clip(m, 0.0, 1.0)

        actual_s = scale_shared(smoothed_actual_rate_map).T
        pred_s   = scale_shared(pred_rate_map).T

        overlay = np.zeros((*actual_s.shape, 3))
        overlay[..., 0] = actual_s   # red
        overlay[..., 1] = pred_s     # green

        axes[i, 3].imshow(
            overlay,
            origin="lower",
            extent=[-1, 1, -1, 1],
        )
        axes[i, 3].set_title(f"Overlay (red=smoothed actual, green=pred), vmax={vmax:.2f}")
        axes[i, 3].set_xlabel("X Position (normalized)")
        axes[i, 3].set_ylabel("Y Position (normalized)")
        fig.colorbar(im1, ax=axes[i, 3], fraction=0.046, pad=0.04)

    # pad the suptitle so that it doesn't overlap with the top row of plots
    # fig.suptitle(f"Model fit vs data (smoothing sigma={smoothing_sigma})", fontsize=14, fontweight="bold")
    fig.tight_layout()

    # --- render once ---
    fig.canvas.draw()

    # --- save to PNG bytes ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()

    plt.show()

    plt.close(fig)
    return png_bytes

seed_model = grid_seed_programs.grid_model_2
seed_model_jax = grid_seed_programs.grid_model_2_jax
parameter_estimator = grid_seed_programs.parameter_estimator_2

seed_model_code_string = utils.format_function_source(seed_model, new_name="grid_model_seed_2")
parameter_estimator_code_string = utils.format_function_source(parameter_estimator, new_name="parameter_estimator_seed_2")

function_signature_prompt = lambda next_version: f"""
**Function Signature:**

Your parameter estimator must have this signature:
```
def parameter_estimator_seed_2_v{next_version}(X, spike_counts):
```
where:
* `X` is a 2D array with shape `(n_features, n_trials)`. Each row `X[i]` is a different input variable.
* For orientation tuning, `X[0]` is the stimulus angle (theta) in radians.
* Access inputs by index: `theta = X[0]`, `contrast = X[1]`, etc.
* `spike_counts` has shape `(n_trials,)`.
* Return: a 1D array of estimated parameter values matching the neuron model's free parameters.
"""

docstring_guidelines_prompt = lambda next_version: f"""
**Docstring Guidelines:**
* Begin by listing the parent models and give them a name that describes their key features, e.g., `parent_model_1: simple_exponential_decay-model`, `parent_model_2: double_exponential_decay_model`. Never refer to the models as `neuron_model_v1`, `neuron_model_v2`, etc. Instead, refer to them as `parent_models` or their descriptive names (e.g. `simple_exponential_decay_model`).
* Do not refer to the current model as `neuron_model_v{next_version}`. Instead, refer to it as "this model".
* Provide a simple description of each parameter
"""

# BEGIN

# initialise chat
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chat = client.aio.chats.create(
    model="gemini-2.5-pro", 
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
        include_thoughts=True
        )
    ),
    history=[]
)

import experiments.grid_cells.seed_programs as grid_seed_programs

async def main(n_iter=4, output_dir=None):
    result = load_and_process_data('/home/dabin/data/Toroidal_topology_grid_cell_data/rat_q_grid_modules_1_2.npz', filter_grid_cells=True)
    inputs = result['inputs']
    outputs = result['outputs']
    rate_maps = result['rate_maps']
    position_data = result['position_data']
    n_spatial_bins = position_data['n_spatial_bins']

    x_raw = inputs['x']
    y_raw = inputs['y']
    n_cells = outputs.shape[0]

    # x_inputs = ensure_inputs(x_raw)
    x_data = inputs.to_tensor()  # shape: (n_samples, n_features, n_trials)
    y = outputs['firing_rate']  # shape: (n_cells, n_trials)
    n_samples, n_features, n_trials = x_data.shape

    # train/test split over trials (axis 2)
    # split the trials into 10 equal length chunks and allocate all odd chunks to train and even chunks to test 
    key = jax.random.PRNGKey(42)        
    n_trial_splits = 10 
    trials_per_split = n_trials // n_trial_splits
    split_indices = [jnp.arange(i * trials_per_split, (i + 1) * trials_per_split) for i in range(n_trial_splits)]
    training_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 1])
    test_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 0])

    # Split inputs and response: x has shape (n_samples, n_features, n_trials)
    x_train = x_data[:, :, training_trials_idx]  # (n_samples, n_features, training_size)
    y_train = y[:, training_trials_idx]           # (n_samples, training_size)
    x_test = x_data[:, :, test_trials_idx]        # (n_samples, n_features, test_size)
    y_test = y[:, test_trials_idx]                # (n_samples, test_size)

    # Create output directory with timestamp if not provided
    if output_dir is None:
        timestamp = datetime.now()
        output_dir = Path(f"program_databases/{timestamp.strftime('%m-%d')}/{timestamp.strftime('%H-%M-%S')}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open log file for writing
    log_file_path = output_dir / "param_estimator.txt"
    log_file = open(log_file_path, "w")
    
    def log_only(text):
        """Write to log file only."""
        log_file.write(text + "\n")
    
    def log_and_print(text):
        """Write to log file and print to console."""
        print(text)
        log_file.write(text + "\n")
    
    seed_model = grid_seed_programs.grid_model_2
    seed_model_jax = grid_seed_programs.grid_model_2_jax
    seed_parameter_estimator = grid_seed_programs.parameter_estimator_2

    current_param_estimator = seed_parameter_estimator
    for i_iter in range(n_iter):    
        if i_iter == 0:
            prompt = lambda next_version : f"""You are an AI scientist, and your task is to analyze and improve parameter estimation for a grid cell firing model.
            Grid cells fire in a hexagonal pattern as an animal navigates through space. Grid cell firing rates are bounded, typically 0-50 Hz. 

            Here is seed_2 grid_model and its corresponding parameter_estimator. Can you try to come up with a better parameter estimator? 
            Here is also a corresponding image diagnostic of the model. Note this image contains the evaluation of both seed_1 and seed_2, but I have only given you the code for seed_2. Ignore seed_1 for now.
            The parameters should be estimated directly using statistical principles and knowledge of what the parameters represent (receptive field positions, amplitudes, sheer, etc). 
            Analyze the progression of the estimators, generalize improvements, and create a new estimator better than all previous estimators. 

            Return the new parameter estimator as a Python function, and also provide an explanation of your reasoning and the improvements you made.
            Any function necessary to run the new estimator should also be included in the code block as we will only select the code inside.

            {function_signature_prompt(next_version)}
            {docstring_guidelines_prompt(next_version)}

            The new parameter_estimator will be called `parameter_estimator_seed_2_v{next_version}` and it should have the same function signature as the previous estimator.
            Make sure to wrap the code in a single qute block with the tag "python" after the triple backticks, so that I can extract it easily using this function : 
                code_blocks = re.findall(r"```python\n(.*?)\n```", part.text, re.DOTALL)

            seed grid_model: 
            {seed_model_code_string}

            parameter_estimator :
            {parameter_estimator_code_string}
            """

            init_image_dir = Path("program_databases/02-02/11-18-26/image_feedback/initial_programs.png")
            img_bytes = init_image_dir.read_bytes()
            
            # Save initial image
            (output_dir / f"iter_{i_iter + 1}_image_input.png").write_bytes(img_bytes)

        else:
            params = compute_initial_params(current_param_estimator, seed_model_jax, np.asarray(x_train), np.asarray(y_train))

            # Compute losses separately (memory-efficient)
            train_mean_loss, train_per_cell_losses = compute_total_loss(
                seed_model_jax, quadratic_loss, x_train, params, y_train
            )

            test_mean_loss, test_per_cell_losses = compute_total_loss(
                seed_model_jax, quadratic_loss, x_test, params, y_test
            )

            sample_selection = np.random.choice(np.asarray(y_train).shape[0], size=6, replace=False)
            mean_loss_sel = jnp.mean(train_per_cell_losses[sample_selection])
            test_mean_loss_sel = jnp.mean(test_per_cell_losses[sample_selection])
            
            img_bytes = plot_model_fit(
                model=seed_model_jax,
                params=params,
                per_cell_losses=train_per_cell_losses,
                smoothing_sigma=1.5,
                inputs=x_train,
                response=y_train,
                sample_selection=sample_selection,
            )
            
            # Save image for this iteration
            (output_dir / f"iter_{i_iter + 1}_image_input.png").write_bytes(img_bytes)

            prompt = lambda next_version : f"""Here is an image diagnostic of the model fit on random grid cells to the data using the new estimator.
            The total mean training loss across ALL {len(y_train)} cells is: {float(train_mean_loss):.4f}. The mean train loss for the 6 displayed cells is: {float(mean_loss_sel):.4f}.       
            The total mean test loss across ALL {len(y_test)} cells is: {float(test_mean_loss):.4f}. The mean test loss for the 6 displayed cells is: {float(test_mean_loss_sel):.4f}.

            The first column displays the mean firing rate of the data with no smoothing. The second column displays the mean firing rate of the data with smoothing (see smoothing sigma value in plot). 
            The third column displays the model fit using the current parameter estimator (no smoothing). 
            The fourth column compares the model fit (green) to the smoothed data (red). We did not normalise the firing rates across these plots, so the absolute firing rates are comparable across columns.
            Each row is a different random grid cell from the data.

            Analyze the fit, identify specific strengths and weaknesses, and suggest targeted improvements to the estimator based on this and your previous analyses.

            The new parameter_estimator will be called `parameter_estimator_seed_2_v{next_version}` and it should have the same function signature as the previous estimator.

            Return the new parameter estimator as a Python function, and also provide an explanation of your reasoning and the improvements you made.
            Any function necessary to run the new estimator should also be included in the code block as we will only select the code inside.

            The new parameter_estimator will be called `parameter_estimator_seed_2_v{next_version}` and it should have the same function signature as the previous estimator.

            Return the new parameter estimator as a Python function, and also provide an explanation of your reasoning and the improvements you made.
            Any function necessary to run the new estimator should also be included in the code block as we will only select the code inside.
            """

        prompt_text = prompt(i_iter + 1)
        log_and_print(f"iter {i_iter + 1} Prompt ")
        log_only(prompt_text)

        message_parts = [types.Part.from_text(text=prompt_text), 
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                        ]

        log_and_print(f"---------- Sending Message : Iteration {i_iter + 1} ----------")
        response = await chat.send_message(message_parts)
        log_and_print(f"---------- Received Message : Iteration {i_iter + 1} ----------")

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                log_and_print(f"iter {i_iter + 1} Thought tokens:")
                log_only(part.text)
                log_only("")
            else:
                log_only(f"iter {i_iter + 1} Answer:")
                log_only(part.text)

                # try to extract the python code which is inside a single quote block with the tag "python" after the triple backticks
                code_block = utils.extract_code_block(part.text)

                if not code_block:
                    raise "No code block found in the response for iteration {i_iter}"
                else:
                    # convert the code into a function and assign it to current_param_estimator
                    current_param_estimator = utils.str_to_func(code_block, f"parameter_estimator_seed_2_v{i_iter + 1}") 
                log_and_print("")
        
    # if last iteration, display the final diagnostic image 
    if i_iter == n_iter - 1:
        params = compute_initial_params(current_param_estimator, seed_model_jax, np.asarray(x_train), np.asarray(y_train))

        # Compute train and test losses separately
        train_mean_loss, train_per_cell_losses = compute_total_loss(
            seed_model_jax, quadratic_loss, x_train, params, y_train
        )
        test_mean_loss, test_per_cell_losses = compute_total_loss(
            seed_model_jax, quadratic_loss, x_test, params, y_test
        )
        
        sample_selection = np.random.choice(np.asarray(y_train).shape[0], size=6, replace=False)
        
        img_bytes = plot_model_fit(
            model=seed_model_jax,
            params=params,
            per_cell_losses=train_per_cell_losses,
            smoothing_sigma=1.5,
            inputs=x_train,
            response=y_train,
            sample_selection=sample_selection,
        )
        
        # Save final output image
        (output_dir / "final_output.png").write_bytes(img_bytes)

    log_and_print(f"Final train mean loss across all cells: {float(train_mean_loss):.4f}")
    log_and_print(f"Final test mean loss across all cells: {float(test_mean_loss):.4f}")
    
    # Close log file
    log_file.close()
    print(f"\nLogs saved to: {log_file_path}")
    print(f"Images saved to: {output_dir}")

import argparse

if __name__ == "__main__":
    import asyncio
    argparser = argparse.ArgumentParser(description="Test LLM calls for grid cell parameter estimator improvement.")
    argparser.add_argument("--n_iter", type=int, default=4, help="Number of iterations for LLM improvement loop.")
    argparser.add_argument("--output_dir", type=str, default=None, help="Output directory for logs and images. Defaults to program_databases/mm-DD/HH-MM-SS/")
    args = argparser.parse_args()
    asyncio.run(main(n_iter=args.n_iter, output_dir=args.output_dir))