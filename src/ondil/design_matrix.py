import numpy as np


def add_intercept(X: np.ndarray):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def make_intercept(n_observations: int) -> np.ndarray:
    """Make the intercept series as N x 1 array.

    Args:
        y (np.ndarray): Response variable $Y$

    Returns:
        np.ndarray: Intercept array.
    """
    return np.ones((n_observations, 1))


def make_lags(
    y: np.ndarray,
    lags: np.ndarray | list[int] | int,
) -> np.ndarray:
    """Make lagged versions of y.

    Args:
        y (np.ndarray): Response variable $Y$
        lags (np.ndarray | list[int] | int): Lags to create.

    Returns:
        np.ndarray: Lagged array.
    """

    if isinstance(lags, int):
        lags = np.linspace(1, lags, lags, dtype=int).tolist()
    n_lags = len(lags)

    X = np.hstack([np.roll(y[:, None], i) for i in lags])
    X[np.triu_indices(n_lags, k=1)] = np.mean(y)

    return X


def subset_array(X, features):
    """Subset array X by features.

    Args:
        X (np.ndarray): Input array.
        features (list[int] | np.ndarray): List of feature indices.

    Returns:
        np.ndarray: Subsetted array.
    """
    if isinstance(features, list):
        return X[:, features]
    elif isinstance(features, np.ndarray):
        return X[:, features.astype(int)]
    elif isinstance(features, str):
        if features == "all":
            return X
        else:
            raise ValueError(f"String feature specifier '{features}' not recognized.")
