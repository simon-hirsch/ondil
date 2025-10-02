import numba as nb
import numpy as np


@nb.njit()
def init_forget_vector(forget: float, size: int) -> np.ndarray:
    """Initialise an exponentially discounted vector of weights.

    Recursively initialise a vector of exponentially discounted weights of `size` N.

    The weight for $n$-th observation is defined as $(1 - \\text{forget})^{(N - n)}$

    Note that this functions assumes that the first observation is the oldest observation
    and the last observation is the newest observation. This is in line with the standard
    `pandas` way of sorting `pd.DataFrame`s with `Datetime`-indices.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        forget (float): Forget factor.
        size (int): Length of the output vector.

    Returns:
        np.ndarray: Vector of exponentially discounted weights.
    """
    gamma = 1 - forget
    if gamma == 1:
        out = np.ones(size)
    else:
        out = np.empty(size)
        out[-1] = 1
        for i in range(size - 1, 0, -1):
            out[i - 1] = out[i] * gamma
    return out


@nb.njit()
def init_gram(X: np.ndarray, w: np.ndarray, forget: float = 0) -> np.ndarray:
    """Initialise the Gramian Matrix.

    The Gramian Matrix is defined as
    $$
    G = X^T \Gamma WX
    $$
    where $X$ is the design matrix, $W$ is a diagonal, user-defined weight matrix, $\\Gamma$ is a
    diagonal matrix of exponentially discounting weights.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        X (np.ndarray): Design matrix $X$
        w (np.ndarray): Weights vector
        forget (float, optional): Forget factor. Defaults to 0.

    Returns:
        np.ndarray: Gramian Matrix.
    """
    f = init_forget_vector(forget, X.shape[0])
    gram = (X * np.expand_dims((w * f) ** 0.5, -1)).T @ (
        X * np.expand_dims((w * f) ** 0.5, -1)
    )
    return gram


@nb.njit()
def init_y_gram(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, forget: float = 0
) -> np.ndarray:
    """Initialise the y-Gramian Matrix.

    The Gramian Matrix is defined as $$
    H = X^T \\Gamma WY
    $$ where $X$ is the design matrix, $Y$ is the response variable, $W$ is a diagonal,
    user-defined weight matrix, $\\Gamma$ is a diagonal matrix of exponentially discounting weights.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        X (np.ndarray): Design matrix $X$
        y (np.ndarray): Response variable $Y$
        w (np.ndarray): Weights vector
        forget (float, optional): Forget factor. Defaults to 0.

    Returns:
        np.ndarray: y-Gramian Matrix.
    """
    f = init_forget_vector(forget, X.shape[0])
    gram = np.expand_dims(
        (X * np.expand_dims((w * f) ** 0.5, -1)).T @ (y * (w * f) ** 0.5), axis=-1
    )
    return gram


@nb.njit()
def init_inverted_gram(X: np.ndarray, w: np.ndarray, forget: float = 0) -> np.ndarray:
    """Initialise the inverted Gramian Matrix.

    The inverted Gramian Matrix is defined as $$
    G = (X^T \Gamma WX)^{-1}
    $$ where $X$ is the design matrix, $W$ is a diagonal, user-defined weight matrix,
    $\Gamma$ is a diagonal matrix of exponentially discounting weights.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        X (np.ndarray): Design matrix $X$
        w (np.ndarray): Weights vector
        forget (float, optional): Forget factor. Defaults to 0.

    Returns:
        np.ndarray: Gramian Matrix.

    Raises:
        ValueError: If the matrix is not invertible (if rank(X.T @ X) < X.shape[0]).

    """
    gram = init_gram(X=X, w=w, forget=forget)
    rank = np.linalg.matrix_rank(gram)
    if rank == gram.shape[0]:
        inv_gram = np.linalg.inv(gram)
        return inv_gram
    else:
        raise ValueError("Matrix is not invertible.")


# TODO (SH): For some reason we switched the syntax in C++ here
# We should change that


@nb.njit()
def update_gram(
    gram: np.ndarray, X: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    """Update the Gramian Matrix.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        gram (np.ndarray): Gramian Matrix
        X (np.ndarray): New observations for $X$
        forget (float, optional): Forget factor. Defaults to 0.
        w (float, optional): Weights for the new observations. Defaults to 1.

    Returns:
        np.ndarray: Updated Gramian Matrix.
    """
    if X.shape[0] == 1:
        # Single Step Update
        new_gram = (1 - forget) * gram + w * np.outer(X, X)
    else:
        # Batch Update
        batch_size = X.shape[0]
        f = init_forget_vector(size=batch_size, forget=forget)
        weights = np.expand_dims((w * f) ** 0.5, axis=-1)
        new_gram = gram * (1 - forget) ** batch_size + (X * weights).T @ (X * weights)
    return new_gram


@nb.njit()
def update_y_gram(
    gram: np.ndarray, X: np.ndarray, y: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    """Update the Y-Gramian Matrix.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        gram (np.ndarray): Gramian Matrix
        X (np.ndarray): New Observations for $X$
        y (np.ndarray): New Observations for $Y$
        forget (float, optional): Forget Factor. Defaults to 0.
        w (float, optional): Weights for the new observations. Defaults to 1.

    Returns:
        np.ndarray: Updated Gramian Matrix.
    """
    if X.shape[0] == 1:
        # Single Update
        new_gram = (1 - forget) * gram + w * np.outer(X, y)
    else:
        # Batch update
        batch_size = X.shape[0]
        f = init_forget_vector(size=batch_size, forget=forget)
        new_gram = gram * (1 - forget) ** batch_size + np.expand_dims(
            ((X * np.expand_dims((w * f) ** 0.5, axis=-1)).T @ (y * (w * f) ** 0.5)), -1
        )
    return new_gram


@nb.jit()
def _update_inverted_gram(
    gram: np.ndarray, X: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    """Update the inverted Gramian for one step"""
    gamma = 1 - forget
    new_gram = (1 / gamma) * (
        gram - ((w * gram @ np.outer(X, X) @ gram) / (gamma + w * X @ gram @ X.T))
    )
    return new_gram


def update_inverted_gram(
    gram: np.ndarray, X: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    """Update the inverted Gramian Matrix.

    !!! numba "Numba"
        This function uses `numba` just-in-time-compilation.

    Args:
        gram (np.ndarray): Inverted Gramian Matrix.
        X (np.ndarray): New observations for $X$.
        forget (float, optional): Forget Factor. Defaults to 0.
        w (float, optional): Weights for new observations. Defaults to 1.

    Returns:
        np.ndarray: Updated inverted Gramian matrix.
    """
    if X.shape[0] == 1:
        new_gram = _update_inverted_gram(gram, X, forget=forget, w=w)
    else:
        new_gram = _update_inverted_gram(
            gram, X=np.expand_dims(X[0, :], 0), forget=forget, w=w[0]
        )
        for i in range(1, X.shape[0]):
            new_gram = _update_inverted_gram(
                gram=new_gram, X=np.expand_dims(X[i, :], 0), forget=forget, w=w[i]
            )
    return new_gram
