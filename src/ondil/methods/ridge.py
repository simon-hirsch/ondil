from typing import Literal, Optional

import numpy as np

from ..base import EstimationMethod
from ..coordinate_descent import online_coordinate_descent
from ..gram import init_gram, init_y_gram, update_gram, update_y_gram


class Ridge(EstimationMethod):
    """Single-lambda Ridge Estimation.

    The ridge method runs coordinate descent for a single lambda.

    We allow to pass user-defined lower and upper bounds for the coefficients.
    The coefficient bounds must be an `numpy` array of the length of `X` respectively of the number of variables in the
    equation _plus the intercept, if you fit one_. This allows to box-constrain the coefficients to a certain range.

    Lastly, we have some rather technical parameters like the number of coordinate descent iterations,
    whether you want to cycle randomly and for which tolerance you want to break. We use active set iterations, i.e.
    after the first coordinate-wise update for each regularization strength, only non-zero coefficients are updated.

    We use `numba` to speed up the coordinate descent algorithm.
    """

    def __init__(
        self,
        lambda_reg: float | None = None,
        start_beta: np.ndarray | None = None,
        selection: Literal["cyclic", "random"] = "cyclic",
        beta_lower_bound: np.ndarray | None = None,
        beta_upper_bound: np.ndarray | None = None,
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
    ):
        """
        Initializes the Ridge method with the specified parameters.

        Args:
            lambda_reg (float): Regularization parameter. Must be greater than 0. Higher values lead to more regularization. If not set, the average variance of the features is used as the default.
            selection (Literal["cyclic", "random"]): Method to select features during the path. Default is "cyclic".
            beta_lower_bound (np.ndarray | None): Lower bound for the coefficients. Default is None.
            beta_upper_bound (np.ndarray | None): Upper bound for the coefficients. Default is None.
            tolerance (float): Tolerance for the optimization. Default is 1e-4.
            max_iterations (int): Maximum number of iterations for the optimization. Default is 1000.
        """
        super().__init__(
            _path_based_method=False,
            _accepts_bounds=True,
            _accepts_selection=False,
        )
        self.beta_lower_bound = beta_lower_bound
        self.beta_upper_bound = beta_upper_bound
        self.lambda_reg = lambda_reg
        self.selection = selection
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.start_beta = start_beta

    def _set_and_validate_bounds(self, x_gram: np.ndarray) -> None:
        J = x_gram.shape[1]
        if self.beta_lower_bound is None:
            self.beta_lower_bound = np.repeat(-np.inf, J)
        if self.beta_upper_bound is None:
            self.beta_upper_bound = np.repeat(np.inf, J)

        if len(self.beta_lower_bound) != J:
            raise ValueError("Lower bound does not have correct length")
        if len(self.beta_upper_bound) != J:
            raise ValueError("Upper bound does not have correct length")

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

    def fit_beta(self, x_gram, y_gram, is_regularized):
        self._set_and_validate_bounds(x_gram=x_gram)

        # beta = (x_gram @ y_gram).squeeze(-1)

        if self.start_beta is not None:
            # Use user-defined start value
            beta = self.start_beta
        else:
            # Use OLS solution for initialization if x_gram is invertible
            if np.linalg.matrix_rank(x_gram) == x_gram.shape[0]:
                beta = (np.linalg.inv(x_gram) @ y_gram).squeeze(-1)
            else:
                # If x_gram is not invertible, initialize with zeros
                beta = np.zeros(x_gram.shape[1])

        if self.lambda_reg is None:
            self.lambda_reg = np.linalg.trace(x_gram) / x_gram.shape[0]

        # Ensure that beta is a column vector
        if beta.ndim > 1:
            raise ValueError(
                "The beta vector must be a column vector. Please check the input data."
            )

        beta, _ = online_coordinate_descent(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta=beta,
            regularization=self.lambda_reg,
            is_regularized=is_regularized,
            alpha=0.0,
            beta_lower_bound=self.beta_lower_bound,
            beta_upper_bound=self.beta_upper_bound,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        return beta

    def update_beta(self, x_gram, y_gram, beta, is_regularized):
        beta, _ = online_coordinate_descent(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta=beta,
            regularization=self.lambda_reg,
            alpha=0.0,
            is_regularized=is_regularized,
            beta_lower_bound=self.beta_lower_bound,
            beta_upper_bound=self.beta_upper_bound,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        return beta

    def fit_beta_path(self, x_gram, y_gram, is_regularized):
        return super().fit_beta_path(x_gram, y_gram, is_regularized)

    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized):
        return super().update_beta_path(x_gram, y_gram, beta_path, is_regularized)
