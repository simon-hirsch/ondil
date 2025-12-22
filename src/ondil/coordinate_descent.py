from typing import Literal, Tuple

import numba as nb
import numpy as np


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


# @nb.njit([
#     "(float64[:,:], float64[:], float64[:], float64, bool[:], str, float64, int64)",
#    "(float32[:,:], float32[:], float32[:], float32, bool[:], str, float32, int32)",
# ])
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
    dual_penalty: np.ndarray | None = None,
    constraint_matrix: np.ndarray | None = None,
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

    if dual_penalty is not None:
        constraint_participation = np.where(constraint_matrix != 0, 1, 0)

    while True:
        i += 1
        beta_star = np.copy(beta_now)
        if (selection == "random") and (i >= 2):
            JJ = np.random.permutation(J)
        for j in JJ:
            if (i < 2) | (beta_now[j] != 0):
                update = (
                    y_gram[j] - (x_gram[j, :] @ beta_now) + x_gram[j, j] * beta_now[j]
                )
                if dual_penalty is not None:
                    update -= np.sum(
                        dual_penalty
                        * constraint_participation[:, j]
                        * constraint_matrix[:, j]
                    )

                if is_regularized[j]:
                    update = soft_threshold(
                        update, alpha * regularization * regularization_weights[j]
                    )
                    denom = x_gram[j, j] + regularization * regularization_weights[
                        j
                    ] * (1 - alpha)
                else:
                    denom = x_gram[j, j]

                if dual_penalty is not None:
                    denom += np.sum(
                        (constraint_participation[:, j] * constraint_matrix[:, j]) ** 2
                    )

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
def linear_constrained_coordinate_descent(
    x_gram: np.ndarray,
    y_gram: np.ndarray,
    beta: np.ndarray,
    regularization: float,
    regularization_weights: np.ndarray | None,
    is_regularized: np.ndarray,
    alpha: float,
    constraint_matrix: np.ndarray,
    constraint_bounds: np.ndarray,
    beta_lower_bound: np.ndarray | None = None,
    beta_upper_bound: np.ndarray | None = None,
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    dual_tolerance: float = 1e-4,
    dual_stepsize: float = 1.0,
    adaptive_dual_stepsize: bool = False,
    max_iterations: int = 1000,
    max_dual_iterations: int = 100,
) -> Tuple[np.ndarray, int]:
    r"""The parameter update cycle of the online coordinate descent with linear constraints."""

    # We solve the dual problem via projected subgradient descent
    constraint_residuals = constraint_matrix @ beta - constraint_bounds
    dual_penalty = np.zeros(constraint_matrix.shape[0])
    dual_gap = np.sum(np.fmax(constraint_residuals, 0) ** 2)
    dual_gap_old = dual_gap + 1e-3

    for i in range(max_dual_iterations):
        beta, k = online_coordinate_descent(
            x_gram=x_gram,
            y_gram=y_gram,
            beta=beta,
            regularization=regularization,
            regularization_weights=regularization_weights,
            is_regularized=is_regularized,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            alpha=alpha,
            selection=selection,
            dual_penalty=dual_penalty,
            constraint_matrix=constraint_matrix,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        # print(dual_penalty, constraint_residuals, dual_stepsize)

        constraint_residuals = constraint_matrix @ beta - constraint_bounds
        dual_penalty = np.fmax(0, dual_penalty + dual_stepsize * constraint_residuals)

        if np.max(constraint_residuals) <= dual_tolerance:
            break

        if adaptive_dual_stepsize:
            dual_gap = np.sum(np.fmax(constraint_residuals, 0) ** 2)
            dual_stepsize = dual_stepsize * dual_gap_old / (dual_gap + 1e-3)
            dual_gap_old = dual_gap

    return beta, i


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
        # Select the according start values for the next CD update
        if which_start_value == "average":
            beta = (beta_path_new[max(i - 1, 0), :] + beta_path[max(i, 0), :]) / 2
        if which_start_value == "previous_lambda":
            beta = beta_path_new[max(i - 1, 0), :]
        else:
            beta = beta_path[max(i, 0), :]

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


@nb.njit()
def online_linear_constrained_coordinate_descent_path(
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
    constraint_matrix: np.ndarray | None,
    constraint_bounds: np.ndarray | None,
    which_start_value: Literal[
        "previous_lambda", "previous_fit", "average"
    ] = "previous_lambda",
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
    dual_tolerance: float = 1e-4,
    dual_stepsize: float = 1.0,
    adaptive_dual_stepsize: bool = False,
    max_dual_iterations: int = 100,
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

    if (constraint_matrix is None) ^ (constraint_bounds is None):
        raise ValueError(
            "Both constraint_matrix and constraint_bounds must be provided for constrained coordinate descent."
        )

    for i, regularization in enumerate(lambda_path):
        # Select the according start values for the next CD update
        if which_start_value == "average":
            beta = (beta_path_new[max(i - 1, 0), :] + beta_path[max(i, 0), :]) / 2
        if which_start_value == "previous_lambda":
            beta = beta_path_new[max(i - 1, 0), :]
        else:
            beta = beta_path[max(i, 0), :]

        if (early_stop > 0) and np.count_nonzero(beta) >= early_stop:
            beta_path_new[i, :] = beta
            iterations[i] = 0
        else:
            beta_path_new[i, :], iterations[i] = linear_constrained_coordinate_descent(
                x_gram=x_gram,
                y_gram=y_gram,
                beta=beta,
                regularization=regularization,
                regularization_weights=regularization_weights,
                is_regularized=is_regularized,
                alpha=alpha,
                constraint_matrix=constraint_matrix,
                constraint_bounds=constraint_bounds,
                beta_lower_bound=beta_lower_bound,
                beta_upper_bound=beta_upper_bound,
                selection=selection,
                tolerance=tolerance,
                dual_tolerance=dual_tolerance,
                dual_stepsize=dual_stepsize,
                adaptive_dual_stepsize=adaptive_dual_stepsize,
                max_iterations=max_iterations,
                max_dual_iterations=max_dual_iterations,
            )

    return beta_path_new, iterations
