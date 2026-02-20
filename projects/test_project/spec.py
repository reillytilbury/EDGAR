"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, *params) and param_est_v1(X, Y)
- model_v2(X, *params) and param_est_v2(X, Y)

Loss:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, programs_list, X_eval, save_path, labels)
"""
import numpy as np
from typing import Tuple
from src.data_structures import Inputs, Outputs


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess data and return canonical Inputs/Outputs.
    """
    raise NotImplementedError


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

def model_v1(X):
    raise NotImplementedError


def param_est_v1(X, Y):
    raise NotImplementedError


def model_v2(X):
    raise NotImplementedError


def param_est_v2(X, Y):
    raise NotImplementedError


# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true):
    return (Y_true - Y_pred) ** 2


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(X, Y, programs_list, X_eval, save_path="", labels=("model_v1", "model_v2")):
    raise NotImplementedError


# ========================
# 5. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================
