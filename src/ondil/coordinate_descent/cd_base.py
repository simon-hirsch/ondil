from typing import Literal, Tuple

import numba as nb
import numpy as np

from .utils import get_start_beta, soft_threshold


@nb.njit()
def beta_update(x_gram, y_gram, beta_now, j):
    return y_gram[j] - (x_gram[j, :] @ beta_now) + x_gram[j, j] * beta_now[j]


@nb.njit()
def online_coordinate_descent(
    x_gram: np.ndarray,
    y_gram: np.ndarray,
    beta: np.ndarray,
    regularization: float,
    regularization_weights: np.ndarray | None,
    is_regularized: np.ndarray,
    alpha: float,
    beta_lower_bound: np.ndarray | None,
    beta_upper_bound: np.ndarray | None,
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, int]:
    r"""The parameter update cycle of the online coordinate descent.

    Args:
        x_gram (np.ndarray): X-Gramian $$X^TX$$
        y_gram (np.ndarray): Y-Gramian $$X^TY$$
        beta (np.ndarray): Current beta vector
        regularization (float): Regularization parameter lambda
        is_regularized (bool): Vector of bools indicating whether the coefficient is regularized
        beta_lower_bound (np.ndarray): Lower bounds for beta
        beta_upper_bound (np.ndarray): Upper bounds for beta
        selection (Literal['cyclic', 'random'], optional): Apply cyclic or random coordinate descent. Defaults to "cyclic".
        tolerance (float, optional): Tolerance for the beta update. Defaults to 1e-4.
        max_iterations (int, optional): Maximum iterations. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, int]: Converged $$ \\beta $$
    """
    i = 0
    J = beta.shape[0]
    JJ = np.arange(J)
    beta_now = np.copy(beta)
    beta_star = np.copy(beta)

    if regularization_weights is None:
        regularization_weights = np.ones(J)

    while True:
        i += 1
        beta_star = np.copy(beta_now)
        if (selection == "random") and (i >= 2):
            JJ = np.random.permutation(J)
        for j in JJ:
            if (i < 2) | (beta_now[j] != 0):
                update = beta_update(x_gram, y_gram, beta_now, j)

                if is_regularized[j]:
                    update = soft_threshold(
                        update, alpha * regularization * regularization_weights[j]
                    )
                    denom = x_gram[j, j] + regularization * regularization_weights[
                        j
                    ] * (1 - alpha)
                else:
                    denom = x_gram[j, j]

                beta_now[j] = update / denom
                # Bounds
                if beta_lower_bound is not None:
                    beta_now[j] = max(beta_now[j], beta_lower_bound[j])
                if beta_upper_bound is not None:
                    beta_now[j] = min(beta_now[j], beta_upper_bound[j])

        if np.max(np.abs(beta_now - beta_star)) <= tolerance * np.max(np.abs(beta_now)):
            break
        if i > max_iterations:
            break
    return beta_now, i


@nb.njit()
def online_coordinate_descent_quadratic(
    x_gram: np.ndarray,
    y_gram: np.ndarray,
    beta: np.ndarray,
    penalty_matrix: np.ndarray,
    quadratic_regularization: float,
    regularization: float,
    regularization_weights: np.ndarray | None,
    is_regularized: np.ndarray,
    alpha: float,
    beta_lower_bound: np.ndarray | None,
    beta_upper_bound: np.ndarray | None,
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, int]:
    r"""Coordinate descent with an additional quadratic penalty matrix.

    Solves the penalized weighted least squares problem in Gram form

    $$\min_\beta \tfrac12 \beta^\top (G + \lambda_S S)\beta - h^\top \beta
    + \text{L1/elastic-net regularization}$$

    by running the standard online coordinate descent on the augmented
    Gramian $G + \lambda_S S$. With a zero penalty matrix this reduces
    exactly to `online_coordinate_descent`.

    Args:
        x_gram (np.ndarray): X-Gramian $$X^TX$$
        y_gram (np.ndarray): Y-Gramian $$X^TY$$
        beta (np.ndarray): Current beta vector
        penalty_matrix (np.ndarray): Quadratic penalty matrix $S$
        quadratic_regularization (float): Regularization strength $\lambda_S$ for the quadratic penalty
        regularization (float): L1/elastic-net regularization parameter lambda
        regularization_weights (np.ndarray | None): Weights for the L1 regularization
        is_regularized (np.ndarray): Vector of bools indicating whether the coefficient is L1-regularized
        alpha (float): The elastic net mixing parameter
        beta_lower_bound (np.ndarray | None): Lower bounds for beta
        beta_upper_bound (np.ndarray | None): Upper bounds for beta
        selection (Literal['cyclic', 'random'], optional): Apply cyclic or random coordinate descent. Defaults to "cyclic".
        tolerance (float, optional): Tolerance for the beta update. Defaults to 1e-4.
        max_iterations (int, optional): Maximum iterations. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, int]: Converged $$ \\beta $$ and the iteration count.
    """
    a_gram = x_gram + quadratic_regularization * penalty_matrix
    return online_coordinate_descent(
        x_gram=a_gram,
        y_gram=y_gram,
        beta=beta,
        regularization=regularization,
        regularization_weights=regularization_weights,
        is_regularized=is_regularized,
        alpha=alpha,
        beta_lower_bound=beta_lower_bound,
        beta_upper_bound=beta_upper_bound,
        selection=selection,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


@nb.njit()
def online_coordinate_descent_quadratic_path(
    x_gram: np.ndarray,
    y_gram: np.ndarray,
    beta_path: np.ndarray,
    lambda_path: np.ndarray,
    penalty_matrix: np.ndarray,
    is_regularized: np.ndarray,
    alpha: float,
    regularization: float,
    regularization_weights: np.ndarray | None,
    beta_lower_bound: np.ndarray | None,
    beta_upper_bound: np.ndarray | None,
    which_start_value: Literal[
        "previous_lambda", "previous_fit", "average"
    ] = "previous_lambda",
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Run quadratic-penalty coordinate descent on a grid of penalty strengths.

    For each $\lambda$ in `lambda_path`, solves the Gram-form problem with
    augmented Gramian $G + \lambda S$, warm-starting via `get_start_beta`.
    The L1 part (`regularization`, typically 0 for pure quadratic penalties)
    is held fixed along the path.

    Args:
        x_gram (np.ndarray): X-Gramian $$X^TX$$
        y_gram (np.ndarray): Y-Gramian $$X^TY$$
        beta_path (np.ndarray): The current coefficient path
        lambda_path (np.ndarray): The grid of quadratic penalty strengths
        penalty_matrix (np.ndarray): Quadratic penalty matrix $S$
        is_regularized (np.ndarray): Vector of bools indicating whether the coefficient is L1-regularized
        alpha (float): The elastic net mixing parameter
        regularization (float): Fixed L1 regularization parameter (0 disables soft-thresholding)
        regularization_weights (np.ndarray | None): Weights for the L1 regularization
        beta_lower_bound (np.ndarray | None): Lower bounds for beta
        beta_upper_bound (np.ndarray | None): Upper bounds for beta
        which_start_value (Literal['previous_lambda', 'previous_fit', 'average'], optional): Values to warm-start the coordinate descent. Defaults to "previous_lambda".
        selection (Literal['cyclic', 'random'], optional): Apply cyclic or random coordinate descent. Defaults to "cyclic".
        tolerance (float, optional): Tolerance for the beta update. Defaults to 1e-4.
        max_iterations (int, optional): Maximum iterations. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple with the updated coefficient path and the iteration counts.
    """
    beta_path_new = np.zeros_like(beta_path)
    iterations = np.zeros_like(lambda_path)

    if regularization_weights is None:
        regularization_weights = np.ones(beta_path.shape[1])

    for i, quadratic_regularization in enumerate(lambda_path):
        beta = get_start_beta(beta_path, beta_path_new, i, which_start_value)
        beta_path_new[i, :], iterations[i] = online_coordinate_descent_quadratic(
            x_gram=x_gram,
            y_gram=y_gram,
            beta=beta,
            penalty_matrix=penalty_matrix,
            quadratic_regularization=quadratic_regularization,
            regularization=regularization,
            regularization_weights=regularization_weights,
            is_regularized=is_regularized,
            alpha=alpha,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            selection=selection,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    return beta_path_new, iterations


@nb.njit()
def online_coordinate_descent_path(
    x_gram: np.ndarray,
    y_gram: np.ndarray,
    beta_path: np.ndarray,
    lambda_path: np.ndarray,
    is_regularized: np.ndarray,
    alpha: float,
    early_stop: int,
    regularization_weights: np.ndarray | None,
    beta_lower_bound: np.ndarray | None,
    beta_upper_bound: np.ndarray | None,
    which_start_value: Literal[
        "previous_lambda", "previous_fit", "average"
    ] = "previous_lambda",
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Run coordinate descent on a grid of regularization values.

    Args:
        x_gram (np.ndarray): X-Gramian $$X^TX$$
        y_gram (np.ndarray): Y-Gramian $$X^TY$$
        beta_path (np.ndarray): The current coefficent path
        lambda_path (np.ndarray): The lambda grid
        is_regularized (bool): Vector of bools indicating whether the coefficient is regularized
        alpha (float): The elastic net mixing parameter
        early_stop (int, optional): Early stopping criterion. 0 implies no early stopping. Defaults to 0.
        beta_lower_bound (np.ndarray): Lower bounds for beta
        beta_upper_bound (np.ndarray): Upper bounds for beta.
        constraint_matrix (np.ndarray): The constraint matrix A
        constraint_bounds (np.ndarray): The constraint bounds b
        which_start_value (Literal['previous_lambda', 'previous_fit', 'average'], optional): Values to warm-start the coordinate descent. Defaults to "previous_lambda".
        selection (Literal['cyclic', 'random'], optional): Apply cyclic or random coordinate descent. Defaults to "cyclic".
        tolerance (float, optional): Tolerance for the beta update. Will be passed through to the parameter update. Defaults to 1e-4.
        max_iterations (int, optional): Maximum iterations. Will be passed through to the parameter update. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple with the updated coefficient path and the iteration count.
    """

    beta_path_new = np.zeros_like(beta_path)
    iterations = np.zeros_like(lambda_path)

    if regularization_weights is None:
        regularization_weights = np.ones(beta_path.shape[1])

    for i, regularization in enumerate(lambda_path):
        beta = get_start_beta(beta_path, beta_path_new, i, which_start_value)
        if (early_stop > 0) and np.count_nonzero(beta) >= early_stop:
            beta_path_new[i, :] = beta
            iterations[i] = 0
        else:
            beta_path_new[i, :], iterations[i] = online_coordinate_descent(
                x_gram=x_gram,
                y_gram=y_gram,
                beta=beta,
                regularization=regularization,
                regularization_weights=regularization_weights,
                is_regularized=is_regularized,
                alpha=alpha,
                beta_lower_bound=beta_lower_bound,
                beta_upper_bound=beta_upper_bound,
                selection=selection,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

    return beta_path_new, iterations
