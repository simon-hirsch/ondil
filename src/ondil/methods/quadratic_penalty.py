from typing import Literal

import numpy as np

from ..base import EstimationMethod
from ..coordinate_descent import online_coordinate_descent_quadratic_path
from ..gram import init_gram, init_y_gram, update_gram, update_y_gram
from ..logging import logger


class QuadraticPenaltyPath(EstimationMethod):
    r"""
    Path-based estimation with a general quadratic penalty matrix.

    Solves the penalized weighted least squares problem in Gram form

    $$\min_\beta \tfrac12 \beta^\top (G + \lambda S)\beta - h^\top \beta$$

    along a (geometric) decreasing grid of penalty strengths $\lambda$, where $S$ is a
    user-supplied positive semi-definite penalty matrix (e.g. a P-spline difference
    penalty $S = D_q^\top D_q$). Coordinate descent is warm-started along the path.

    If a fixed `lambda_` is given, the path collapses to a single value.

    The maximum penalty strength is scaled relative to the data via
    $\lambda_\max = c \cdot \mathrm{tr}(G) / \mathrm{tr}(S)$ so that the largest
    penalty effectively enforces the penalty null space, and the lower end of the
    grid is $\lambda_\min = \lambda_\max \varepsilon_\lambda$.

    We use `numba` to speed up the coordinate descent algorithm.
    """

    def __init__(
        self,
        penalty_matrix: np.ndarray | None = None,
        lambda_n: int = 50,
        lambda_eps: float = 1e-12,
        lambda_max_scale: float = 1e6,
        lambda_: float | None = None,
        start_value_initial: Literal[
            "previous_lambda", "previous_fit", "average"
        ] = "previous_lambda",
        start_value_update: Literal[
            "previous_lambda", "previous_fit", "average"
        ] = "previous_fit",
        selection: Literal["cyclic", "random"] = "cyclic",
        beta_lower_bound: np.ndarray | None = None,
        beta_upper_bound: np.ndarray | None = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ):
        """
        Initializes the quadratic penalty path method.

        Args:
            penalty_matrix (np.ndarray | None): The quadratic penalty matrix $S$. Can also be passed
                at fit time via `fit_beta_path(..., penalty_matrix=...)`.
            lambda_n (int): Number of lambda values to use in the path. Default is 50.
            lambda_eps (float): Minimum lambda value as a fraction of the maximum lambda. Default is 1e-12.
            lambda_max_scale (float): Scale factor $c$ for the data-driven maximum lambda
                $\\lambda_\\max = c \\cdot \\mathrm{tr}(G) / \\mathrm{tr}(S)$. Default is 1e6.
            lambda_ (float | None): Fixed penalty strength. If given, no grid is used. Default is None.
            start_value_initial (Literal["previous_lambda", "previous_fit", "average"]): Method to initialize the start value for the first fit. Default is "previous_lambda".
            start_value_update (Literal["previous_lambda", "previous_fit", "average"]): Method to choose the start value on updates. Default is "previous_fit".
            selection (Literal["cyclic", "random"]): Coordinate selection scheme. Default is "cyclic".
            beta_lower_bound (np.ndarray | None): Lower bound for the coefficients. Default is None.
            beta_upper_bound (np.ndarray | None): Upper bound for the coefficients. Default is None.
            tolerance (float): Tolerance for the optimization. Default is 1e-6.
            max_iterations (int): Maximum number of iterations for the optimization. Default is 1000.
        """
        super().__init__(
            _path_based_method=True,
            _accepts_bounds=True,
            _accepts_selection=True,
        )
        self.penalty_matrix = penalty_matrix
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.lambda_max_scale = lambda_max_scale
        self.lambda_ = lambda_
        self.start_value_initial = start_value_initial
        self.start_value_update = start_value_update
        self.selection = selection
        self.beta_lower_bound = beta_lower_bound
        self.beta_upper_bound = beta_upper_bound
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self._path_length = 1 if lambda_ is not None else lambda_n

    def make_lambda_path(
        self,
        x_gram: np.ndarray,
        penalty_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct the (geometrically decreasing) grid of penalty strengths."""
        if self.lambda_ is not None:
            return np.array([float(self.lambda_)])
        penalty_matrix = (
            penalty_matrix if penalty_matrix is not None else self.penalty_matrix
        )
        if penalty_matrix is None:
            raise ValueError("No penalty matrix provided.")
        trace_penalty = np.trace(penalty_matrix)
        if np.isclose(trace_penalty, 0.0):
            raise ValueError("Penalty matrix has zero trace.")
        lambda_max = self.lambda_max_scale * np.trace(x_gram) / trace_penalty
        if not np.isfinite(lambda_max) or lambda_max <= 0:
            lambda_max = 1.0
        return np.geomspace(lambda_max, lambda_max * self.lambda_eps, self.lambda_n)

    @staticmethod
    def effective_degrees_of_freedom(
        x_gram: np.ndarray,
        penalty_matrix: np.ndarray,
        lambda_path: np.ndarray,
    ) -> np.ndarray:
        r"""Compute the effective degrees of freedom for each penalty strength.

        $$\mathrm{edf}(\lambda) = \mathrm{tr}\big((G + \lambda S)^{-1} G\big)$$
        """
        edf = np.empty(lambda_path.shape[0])
        for i, lam in enumerate(lambda_path):
            edf[i] = np.trace(np.linalg.solve(x_gram + lam * penalty_matrix, x_gram))
        return edf

    @staticmethod
    def init_x_gram(X, weights, forget):
        return init_gram(X=X, w=weights, forget=forget)

    @staticmethod
    def init_y_gram(X, y, weights, forget):
        return init_y_gram(X, y, w=weights, forget=forget)

    @staticmethod
    def update_x_gram(gram, X, weights, forget):
        return update_gram(gram, X, w=weights, forget=forget)

    @staticmethod
    def update_y_gram(gram, X, y, weights, forget):
        return update_y_gram(gram, X, y, forget=forget, w=weights)

    def _fit_path(
        self,
        x_gram: np.ndarray,
        y_gram: np.ndarray,
        beta_path: np.ndarray,
        is_regularized: np.ndarray,
        which_start_value: str,
        **kwargs,
    ) -> np.ndarray:
        logger.debug(f"Got following kwargs: {[*kwargs.keys()]}")
        penalty_matrix = kwargs.get("penalty_matrix", self.penalty_matrix)
        if penalty_matrix is None:
            raise ValueError("No penalty matrix provided.")

        beta_lower_bound = kwargs.get("beta_lower_bound", self.beta_lower_bound)
        beta_upper_bound = kwargs.get("beta_upper_bound", self.beta_upper_bound)

        lambda_path = self.make_lambda_path(
            x_gram=x_gram, penalty_matrix=penalty_matrix
        )
        self.lambda_path_ = lambda_path

        beta_path, _ = online_coordinate_descent_quadratic_path(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta_path=beta_path,
            lambda_path=lambda_path,
            penalty_matrix=penalty_matrix,
            is_regularized=is_regularized,
            alpha=1.0,
            regularization=0.0,
            regularization_weights=None,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            which_start_value=which_start_value,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        return beta_path

    def fit_beta_path(self, x_gram, y_gram, is_regularized, **kwargs):
        beta_path = np.zeros((self._path_length, x_gram.shape[0]))
        return self._fit_path(
            x_gram=x_gram,
            y_gram=y_gram,
            beta_path=beta_path,
            is_regularized=is_regularized,
            which_start_value=self.start_value_initial,
            **kwargs,
        )

    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized, **kwargs):
        return self._fit_path(
            x_gram=x_gram,
            y_gram=y_gram,
            beta_path=beta_path,
            is_regularized=is_regularized,
            which_start_value=self.start_value_update,
            **kwargs,
        )

    def fit_beta(self, x_gram, y_gram, **kwargs):
        raise NotImplementedError(
            "QuadraticPenaltyPath is a path-based method. Use fit_beta_path."
        )

    def update_beta(self, x_gram, y_gram, beta, **kwargs):
        raise NotImplementedError(
            "QuadraticPenaltyPath is a path-based method. Use update_beta_path."
        )
