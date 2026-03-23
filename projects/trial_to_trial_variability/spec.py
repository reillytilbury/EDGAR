"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, params) and param_est_v1(X, Y)
- model_v2(X, params) and param_est_v2(X, Y)
- params is a dict of named arrays/scalars (same keys for model + estimator)

LOSS FUNCTION:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
"""
import numpy as np
from typing import Tuple
from src.data_structures import Inputs, Outputs


# ========================
# 1. DATA
# ========================

def extract_stimulus_related_response(data: dict, n_pcs: int = 8, z_score: bool = False, spont_mean_removal: bool = False) -> np.ndarray:
    """
    Extracts the stimulus-related response from the data. Copy pasted with small modifications from https://github.com/MouseLand/stringer-et-al-2019/blob/master/utils.py#L98
    Args:
        data (dict): The data dictionary containing the stimulus-related response and other information. Values expected to be convertible to JAX arrays.
        n_pcs (int): The number of spointaneous PCs to remove from the response.
        z_score (bool): Whether to z-score the response.
    Returns:
        stim_related_response (np.ndarray): The stimulus-related response matrix.
    """
    # Convert relevant data parts to JAX arrays explicitly if needed, JAX often handles np arrays implicitly
    sresp = np.asarray(data['sresp'])

    if spont_mean_removal:
        mean_spont = np.asarray(data['mean_spont'])
        sresp = sresp - mean_spont[:, np.newaxis]

    if n_pcs > 0:
        u_spont = np.asarray(data['u_spont'])
        sresp = sresp - u_spont[:, :n_pcs] @ (u_spont[:, :n_pcs].T @ sresp)

    if z_score:
        sresp = (sresp - np.mean(sresp, axis=1, keepdims=True)) / np.std(sresp, axis=1, keepdims=True)    

    return sresp

def zscore_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    z-score each row of a 2D array
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)    
    return (X - mean) / (std + eps)

def load_and_process_data(
    data_path: str, 
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
    train_to_test_split_ratio: float = 0.5,
    # zscore: bool = True,
    conc_threshold: float = 0.55,
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess neural data and return data in the form of 
    Inputs and Outputs objects. 
    
    Parameters
    ----------
    data_path : str
        Path to the .npy file containing neural data.
    random_seed : int
        Random seed for reproducibility of source / target cell split 
    train_to_test_split_ratio : float
        Ratio of source cells to target cells (e.g. 0.7 means 70% of cells are in the training pouplation)     

    Returns
    -------
    X: Inputs object - has shape (2, n_source_cells, n_time) where sample 0 is a training population 
        and sample 1 is the held-out population for testing 
    Y: Outputs object - has shape (2, n_target_cells, n_time) where sample 0 is the training population 
        and sample 1 is the held-out population for testing
    """
    # load and preprocess data
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    # Use passed data extraction function or fall back to default
    response = extract_stimulus_related_response(neural_data, n_pcs=0)

    angles = neural_data['istim'] # shape (n_trials)
    assert max(angles) <= 2 * np.pi, "Expected angles to be in radians and between 0 and 2pi"
    n_trials = response.shape[1]

    if conc_threshold is not None:
        conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
        good_cells = np.where(conc > conc_threshold)[0]

        print(f"Filtering for orientation selectivity. Kept {len(good_cells)} out of {response.shape[0]} cells.")
        response = response[good_cells]

    rng = np.random.default_rng(random_seed)

    # Source/Target split for trial to trial variability 
    cell_idx = rng.permutation(response.shape[0])
    n_source_cells = int(train_to_test_split_ratio * response.shape[0])
    source_cells = cell_idx[:n_source_cells]
    target_cells = cell_idx[n_source_cells:]

    if source_cells.size == 0 or target_cells.size == 0:
        raise ValueError("Train to test split ratio results in empty source or target cell population. Please adjust the ratio.")    

    half_source = source_cells.size // 2 
    half_target = target_cells.size // 2 
    train_source = source_cells[:half_source]
    # make sure we handle odd number of cells by putting the extra cell in the training set
    test_source = source_cells[-half_source:]
    train_target = target_cells[:half_target]
    test_target = target_cells[-half_target:]

    # TODO : think about whether z_scoring should happen after the cell and trial split or before? Currently it's after 
    X_train = response[train_source]
    X_test = response[test_source]
    Y_train = response[train_target]
    Y_test = response[test_target]

    # rather than zscore, just divide by the std 
    eps = 1e-12
    X_train = X_train / (X_train.std(axis=1, keepdims=True) + eps)
    X_test = X_test / (X_test.std(axis=1, keepdims=True) + eps)
    Y_train = Y_train / (Y_train.std(axis=1, keepdims=True) + eps)
    Y_test = Y_test / (Y_test.std(axis=1, keepdims=True) + eps)
    # if zscore:
    #     X_train = zscore_rows(X_train)
    #     X_test = zscore_rows(X_test)
    #     Y_train = zscore_rows(Y_train)
    #     Y_test = zscore_rows(Y_test)

    # Create Inputs object with the angles as the first (and currently only) input
    # Shape: (n_cells, 1, n_trials)
    X_dict = {}
    # create a stimulus array by tiling it by n_source number of times 
    X_dict['stimulus'] = np.tile(angles, (half_source, 1))
    X_dict['train'] = X_train
    X_dict['test'] = X_test 

    Y_dict = {}
    Y_dict['stimulus'] = np.tile(angles, (half_target, 1))
    Y_dict['train'] = Y_train
    Y_dict['test'] = Y_test
    # X = np.stack([X_train, X_test], axis=0) # (2, n_source, n_time)
    # Y = np.stack([Y_train, Y_test], axis=0) # (2, n_target, n_time)
    X = Inputs.from_dict(X_dict)
    Y = Outputs.from_dict(Y_dict)
    return X, Y

def train_test_split(
    X: Inputs,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample indices and train trial indices.
    """
    raise NotImplementedError


# ========================
# 2. SEED MODELS
# ========================

def model_v1(X, params):
    raise NotImplementedError


def param_est_v1(X, Y):
    raise NotImplementedError


def model_v2(X, params):
    raise NotImplementedError


def param_est_v2(X, Y):
    raise NotImplementedError


# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true):
    raise NotImplementedError


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(X, Y, programs_list, save_path=""):
    raise NotImplementedError


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================
