import math
from typing import Any


def _safe_loss(val: Any) -> float:
    """Returns a float representation of a loss value, mapping None, nan,
    and non-finite values to float("inf"), while letting invalid types raise.
    """
    if val is None:
        return float("inf")
    val_float = float(val)
    if not math.isfinite(val_float):
        return float("inf")
    return val_float
