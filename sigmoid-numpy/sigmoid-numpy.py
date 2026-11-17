import numpy as np


def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    exp_neg_abs = np.exp(-np.abs(x_arr))
    ans = np.where(
        x_arr >= 0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )

    if x_arr.ndim == 0:
        return float(ans.item())

    return ans