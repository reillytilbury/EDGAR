import numpy as np
# import utils

def load_data(data_path: str | None = None,
              conc_thresh: float = 0.55,
              activity_thresh: float = 0.4,
              n_pcs: int = 0):
    """
    Load orientation dataset and return (X, Y).

    X: (n_stim_dim, n_trials) with n_stim_dim=1 (angles in radians)
    Y: (n_cells, n_trials) spike counts
    """
    if data_path is None:
        raise ValueError("Orientation data_path is not set. Provide data_path or set DATA_PATH.")
    neural_data = np.load(data_path, allow_pickle=True).item()
    response = utils.extract_stimulus_related_response(neural_data, n_pcs=n_pcs)
    angles = np.asarray(neural_data["istim"])

    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    response_sum = np.sum(response, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / (response_sum + 1e-8))
    good_cells = np.where((firing_probs > activity_thresh) & (conc > conc_thresh))[0]
    response = response[good_cells, :]

    X = angles[None, :]
    Y = response
    return X, Y
