import numpy as np
import numba as nb


@nb.njit()
def soft_threshold(value: float, threshold: float):
    r"""The soft thresholding function.

    For value \(x\) and threshold \(\\lambda\), the soft thresholding function \(S(x, \\lambda)\) is
    defined as:

    $$S(x, \\lambda) = sign(x)(|x| - \\lambda)$$

    Args:
        value (float): The value
        threshold (float): The threshold

    Returns:
        out (float): The thresholded value
    """
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0)


@nb.njit()
def get_start_beta(beta_path, beta_path_new, i, which_start_value):
    r"""
    Get the starting value for the coordinate descent at the next lambda.
    """
    if which_start_value == "average":
        return (beta_path_new[max(i - 1, 0), :] + beta_path[max(i, 0), :]) / 2
    elif which_start_value == "previous_lambda":
        return beta_path_new[max(i - 1, 0), :]
    else:
        return beta_path[max(i, 0), :]
