import numpy as np


def add_intercept(X: np.ndarray):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def make_intercept(n_observations: int) -> np.ndarray:
    r"""Make the intercept series as N x 1 array.

    Args:
        y (np.ndarray): Response variable $Y$

    Returns:
        np.ndarray: Intercept array.
    """
    return np.ones((n_observations, 1))
