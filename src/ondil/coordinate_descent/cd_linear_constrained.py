from typing import Literal, Tuple

import numpy as np
import numba as nb

from .utils import soft_threshold, get_start_beta


@nb.njit()
def online_coordinate_descent_lagrange_relaxed(
    x_gram: np.ndarray,
    y_gram: np.ndarray,
    beta: np.ndarray,
    regularization: float,
    regularization_weights: np.ndarray | None,
    is_regularized: np.ndarray,
    alpha: float,
    beta_lower_bound: np.ndarray | None,
    beta_upper_bound: np.ndarray | None,
    dual_penalty: np.ndarray | None,
    constraint_matrix: np.ndarray | None,
    constraint_bounds: np.ndarray | None,
    dual_stepsize: float = 1.0,
    relaxation_method: Literal["alm", "pdga"] = "alm",
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
        constraint_matrix (np.ndarray): Constraint matrix for linear constraints
        constraint_bounds (np.ndarray): Constraint bounds for linear constraints
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
                update = (
                    y_gram[j] - (x_gram[j, :] @ beta_now) + x_gram[j, j] * beta_now[j]
                )
                if relaxation_method == "pdga":
                    update -= np.sum(dual_penalty * constraint_matrix[:, j])
                elif relaxation_method == "alm":
                    tilde_mu = dual_penalty + dual_stepsize * (
                        constraint_matrix @ beta_now - constraint_bounds
                    )
                    update -= np.sum(tilde_mu * constraint_matrix[:, j])
                    # mask = np.arange(J) != j
                    # update += -np.sum(
                    #     dual_penalty * constraint_matrix[:, j]
                    # ) + dual_stepsize * (
                    #     +(constraint_matrix[:, j] @ constraint_bounds)
                    #     - constraint_gram[mask, j] @ beta_now[mask]
                    # )
                else:
                    raise ValueError(
                        f"Relaxation method {relaxation_method} not recognized."
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

                if relaxation_method == "alm":
                    denom += dual_stepsize * np.sum(constraint_matrix[:, j] ** 2)

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
def online_linear_constrained_coordinate_descent(
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
    relaxation_method: Literal["alm", "pdga"] = "alm",
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    dual_tolerance: float = 1e-4,
    dual_stepsize: float = 1.0,
    max_iterations: int = 1000,
    max_dual_iterations: int = 100,
) -> Tuple[np.ndarray, int]:
    r"""The parameter update cycle of the online coordinate descent with linear constraints."""

    # We solve the dual problem via projected subgradient descent
    constraint_residuals = constraint_matrix @ beta - constraint_bounds
    dual_penalty = np.zeros(constraint_matrix.shape[0])

    stepsize = dual_stepsize

    for i in range(max_dual_iterations):
        beta, k = online_coordinate_descent_lagrange_relaxed(
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
            relaxation_method=relaxation_method,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            dual_stepsize=stepsize,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        # print(dual_penalty, constraint_residuals, dual_stepsize)
        constraint_residuals = constraint_matrix @ beta - constraint_bounds
        dual_penalty = np.fmax(0, dual_penalty + stepsize * constraint_residuals)

        # for ALM we can have adaptive step sizes
        # to ensure that we don't have the bias in the quadratic penalty
        # if all constraints are satisfied
        if relaxation_method == "alm":
            if np.all(constraint_residuals <= 0):
                stepsize = min(max(stepsize / 2, 0.0001), dual_stepsize)
            else:
                stepsize = min(max(stepsize * 2, 0.0001), dual_stepsize)

        if np.max(constraint_residuals) <= dual_tolerance:
            break

    return beta, i


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
    relaxation_method: Literal["alm", "dpga"] = "alm",
    which_start_value: Literal[
        "previous_lambda", "previous_fit", "average"
    ] = "previous_lambda",
    selection: Literal["cyclic", "random"] = "cyclic",
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
    dual_tolerance: float = 1e-4,
    dual_stepsize: float = 1.0,
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
        beta = get_start_beta(beta_path, beta_path_new, i, which_start_value)

        if (early_stop > 0) and np.count_nonzero(beta) >= early_stop:
            beta_path_new[i, :] = beta
            iterations[i] = 0
        else:
            beta_path_new[i, :], iterations[i] = (
                online_linear_constrained_coordinate_descent(
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
                    relaxation_method=relaxation_method,
                    selection=selection,
                    tolerance=tolerance,
                    dual_tolerance=dual_tolerance,
                    dual_stepsize=dual_stepsize,
                    max_iterations=max_iterations,
                    max_dual_iterations=max_dual_iterations,
                )
            )

    return beta_path_new, iterations
