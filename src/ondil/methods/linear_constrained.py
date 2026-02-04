from typing import Literal

import numpy as np

from ..base import EstimationMethod
from ..coordinate_descent import (
    online_linear_constrained_coordinate_descent,
    online_linear_constrained_coordinate_descent_path,
)
from ..gram import init_gram, init_y_gram, update_gram, update_y_gram
from ..logging import logger
from .elasticnet import ElasticNetPath


class LinearConstrainedCoordinateDescent(EstimationMethod):
    r"""Linear Constrained (unconstrained) Estimation."""

    def __init__(
        self,
        constraint_matrix: np.ndarray | None = None,
        constraint_bounds: np.ndarray | None = None,
        relaxation_method: Literal["alm", "pgda"] = "alm",
        beta_lower_bound: np.ndarray | None = None,
        beta_upper_bound: np.ndarray | None = None,
        selection: Literal["cyclic", "random"] = "cyclic",
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
        max_dual_iterations: int = 100,
    ):
        """Linear constrained coordinate descent method for linear models.

        We solve the following optimization problem:
        $$ \min_{\beta} \frac{1}{2} ||y - X \beta||_2^2 \\
        \text{s.t. } C \beta \leq d 
        L \leq \beta \leq U
        $$

        Where $C$ is the constraint matrix and $d$ are the constraint bounds and
        $L$ and $U$ are the lower and upper bounds on the coefficients.
        This allows to impose linear constraints on the coefficients and box constraints.

        Estimation runs using coordinate descent with either Augmented Lagrangian Method (ALM) or
        Projected Gradient Descent Algorithm (PGDA) to handle the constraints. This implies the
        CD solves a relaxed version of the constrained problem at each iteration.
        
        ALM is generally more stable and accurate, while PGDA is faster but may be less accurate.

        The constraint matrix and bounds can be provided during initialization or
        during the fit/update methods as kwargs to allow for dynamic constraints.

        Args:
            constraint_matrix (np.ndarray | None): Constraint matrix C. Default is None.
            constraint_bounds (np.ndarray | None): Constraint bounds d. Default is None.
            relaxation_method (Literal["alm", "pgda"]): Method to handle constraints. Default
            beta_lower_bound (np.ndarray | None): Lower bound for the coefficients. Default is None.
            beta_upper_bound (np.ndarray | None): Upper bound for the coefficients. Default is None.
            selection (Literal["cyclic", "random"]): Method to select features during the path. Default is "cyclic".
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
        self.constraint_matrix = constraint_matrix
        self.constraint_bounds = constraint_bounds
        self.relaxation_method = relaxation_method
        self.max_dual_iterations = max_dual_iterations

        # CD parameters
        self.selection = selection
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    @staticmethod
    def _validate_bounds(
        beta_lower_bound: np.ndarray,
        beta_upper_bound: np.ndarray,
        x_gram: np.ndarray,
    ) -> None:
        J = x_gram.shape[1]
        if beta_lower_bound is not None:
            if len(beta_lower_bound) != J:
                raise ValueError("Lower bound does not have correct length")
        if beta_upper_bound is not None:
            if len(beta_upper_bound) != J:
                raise ValueError("Upper bound does not have correct length")

    @staticmethod
    def _raise_missing_constraint(constraint_matrix, constraint_bounds):
        if (constraint_matrix is None) or (constraint_bounds is None):
            raise ValueError(
                "Constraint matrix and constraint bounds must be provided for linear constrained coordinate descent."
            )

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

    def fit_beta(
        self,
        x_gram,
        y_gram,
        is_regularized,
        **kwargs,
    ):
        beta = np.zeros(x_gram.shape[1])

        logger.debug(f"Got following kwargs: {[*kwargs.keys()]}")
        constraint_matrix = kwargs.get("constraint_matrix", None)
        constraint_bounds = kwargs.get("constraint_bounds", None)
        beta_lower_bound = kwargs.get("beta_lower_bound", self.beta_lower_bound)
        beta_upper_bound = kwargs.get("beta_upper_bound", self.beta_upper_bound)
        self._validate_bounds(
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            x_gram=x_gram,
        )

        if constraint_matrix is None:
            constraint_matrix = self.constraint_matrix
            logger.debug("Using constraint matrix from class attribute.")
        if constraint_bounds is None:
            constraint_bounds = self.constraint_bounds
            logger.debug("Using constraint bounds from class attribute.")

        # Ensure that beta is a column vector
        if beta.ndim > 1:
            raise ValueError(
                "The beta vector must be a column vector. Please check the input data."
            )

        self._raise_missing_constraint(constraint_matrix, constraint_bounds)

        beta, _ = online_linear_constrained_coordinate_descent(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta=beta,
            regularization=0.0,
            regularization_weights=None,
            is_regularized=is_regularized,
            alpha=0.0,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            relaxation_method=self.relaxation_method,
            max_dual_iterations=self.max_dual_iterations,
        )
        return beta

    def update_beta(self, x_gram, y_gram, beta, is_regularized, **kwargs):
        logger.debug(f"Got following kwargs: {[*kwargs.keys()]}")
        constraint_matrix = kwargs.get("constraint_matrix", None)
        constraint_bounds = kwargs.get("constraint_bounds", None)
        beta_lower_bound = kwargs.get("beta_lower_bound", self.beta_lower_bound)
        beta_upper_bound = kwargs.get("beta_upper_bound", self.beta_upper_bound)
        self._validate_bounds(
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            x_gram=x_gram,
        )

        if constraint_matrix is None:
            constraint_matrix = self.constraint_matrix
            logger.debug("Using constraint matrix from class attribute.")
        if constraint_bounds is None:
            constraint_bounds = self.constraint_bounds
            logger.debug("Using constraint bounds from class attribute.")

        self._raise_missing_constraint(constraint_matrix, constraint_bounds)

        beta, _ = online_linear_constrained_coordinate_descent(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta=beta,
            regularization=0.0,
            regularization_weights=None,
            is_regularized=is_regularized,
            alpha=0.0,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            relaxation_method=self.relaxation_method,
            max_dual_iterations=self.max_dual_iterations,
        )
        return beta

    def fit_beta_path(self, x_gram, y_gram, is_regularized):
        return super().fit_beta_path(x_gram, y_gram, is_regularized)

    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized):
        return super().update_beta_path(x_gram, y_gram, beta_path, is_regularized)


class LinearConstrainedElasticNetPath(ElasticNetPath):
    r"""Linear Constrained Elastic Net Path Estimation.

    The elastic net method runs coordinate descent along a (geometric) decreasing grid of regularization strengths (lambdas).
    We automatically calculate the maximum regularization strength for which all (not-regularized) coefficients are 0.
    The lower end of the lambda grid is defined as $$\\lambda_\min = \\lambda_\max * \\varepsilon_\\lambda.$$

    The elastic net method is a combination of LASSO and Ridge regression. Parameter $\alpha$ controls the balance
    between LASSO and Ridge. Thereby, $\alpha=0$ corresponds to Ridge regression and $\alpha=1$ corresponds to LASSO
    regression.

    We allow to pass user-defined lower and upper bounds for the coefficients.
    The coefficient bounds must be an `numpy` array of the length of `X` respectively of the number of variables in the
    equation _plus the intercept, if you fit one_. This allows to box-constrain the coefficients to a certain range.

    Furthermore, we allow to choose the start value, i.e. whether you want an update to be warm-started on the previous fit's path
    or on the previous reguarlization strength or an average of both. If your data generating process is rather stable,
    the `"previous_fit"` should give considerable speed gains, since warm starting on the previous strength is effectively batch-fitting.

    Lastly, we have some rather technical parameters like the number of coordinate descent iterations,
    whether you want to cycle randomly and for which tolerance you want to break. We use active set iterations, i.e.
    after the first coordinate-wise update for each regularization strength, only non-zero coefficients are updated.

    !!! numba
        We use `numba` to speed up the coordinate descent algorithm.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        lambda_n: int = 100,
        lambda_eps: float = 1e-4,
        early_stop: int = 0,
        constraint_matrix: np.ndarray | None = None,
        constraint_bounds: np.ndarray | None = None,
        relaxation_method: Literal["alm", "pgda"] = "alm",
        beta_lower_bound: np.ndarray | None = None,
        beta_upper_bound: np.ndarray | None = None,
        auto_regularization_weights: bool = False,
        start_value_initial: Literal[
            "previous_lambda", "previous_fit", "average"
        ] = "previous_lambda",
        start_value_update: Literal[
            "previous_lambda", "previous_fit", "average"
        ] = "previous_fit",
        selection: Literal["cyclic", "random"] = "cyclic",
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
        max_dual_iterations: int = 100,
    ):
        r"""
        Initializes the LinearConstrainedElasticNetPath method with the specified parameters.

        Args:
            alpha (float): Mixing parameter between the L1 and L2 loss. Alpha = 0 corresponds to Ridge, Alpha = 1 corresponds to LASSO.
            lambda_n (int): Number of lambda values to use in the path. Default is 100.
            lambda_eps (float): Minimum lambda value as a fraction of the maximum lambda. Default is 1e-4.
            early_stop (int): Early stopping criterion. Will stop if the number of non-zero parameters is reached. Default is 0 (no early stopping).
            constraint_matrix (np.ndarray | None): Constraint matrix C for linear constraints. Default is None.
            constraint_bounds (np.ndarray | None): Constraint bounds d for linear constraints. Default is None.
            relaxation_method (Literal["alm", "pgda"]): Method to handle constraints. Default is "alm".
            beta_lower_bound (np.ndarray | None): Lower bound for the coefficients. Default is None.
            beta_upper_bound (np.ndarray | None): Upper bound for the coefficients. Default is None.
            auto_regularization_weights (bool): Whether to automatically compute regularization weights based on the data. Default is False.
            start_value_initial (Literal["previous_lambda", "previous_fit", "average"]): Method to initialize the start value for the first lambda. Default is "previous_lambda".
            start_value_update (Literal["previous_lambda", "previous_fit", "average"]): Method to update the start value for subsequent lambdas. Default is "previous_fit".
            selection (Literal["cyclic", "random"]): Method to select features during the path. Default is "cyclic".
            tolerance (float): Tolerance for the optimization. Default is 1e-4.
            max_iterations (int): Maximum number of iterations for the optimization. Default is 1000.
        """

        super().__init__(
            alpha=alpha,
            lambda_n=lambda_n,
            lambda_eps=lambda_eps,
            selection=selection,
            tolerance=tolerance,
            max_iterations=max_iterations,
            auto_regularization_weights=auto_regularization_weights,
            start_value_initial=start_value_initial,
            start_value_update=start_value_update,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            early_stop=early_stop,
        )

        self.constraint_matrix = constraint_matrix
        self.constraint_bounds = constraint_bounds
        self.relaxation_method = relaxation_method
        self.max_dual_iterations = max_dual_iterations

    @staticmethod
    def _raise_missing_constraint(constraint_matrix, constraint_bounds):
        if (constraint_matrix is None) or (constraint_bounds is None):
            raise ValueError(
                "Constraint matrix and constraint bounds must be provided for linear constrained coordinate descent."
            )

    def fit_beta_path(self, x_gram, y_gram, is_regularized, **kwargs):
        logger.debug(f"Got following kwargs: {[*kwargs.keys()]}")
        regularization_weights = kwargs.get("regularization_weights", None)
        constraint_matrix = kwargs.get("constraint_matrix", None)
        constraint_bounds = kwargs.get("constraint_bounds", None)
        beta_lower_bound = kwargs.get("beta_lower_bound", self.beta_lower_bound)
        beta_upper_bound = kwargs.get("beta_upper_bound", self.beta_upper_bound)

        self._validate_bounds(
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            x_gram=x_gram,
        )

        if self.auto_regularization_weights:
            if regularization_weights is not None:
                logger.warning(
                    "Both automatic regularization weights and user-defined regularization weights are provided. "
                    "Using automatic regularization weights."
                )
            regularization_weights = self._calculate_regularization_weights(
                x_gram=x_gram
            )

        lambda_max = self._get_lambda_max(
            x_gram=x_gram,
            y_gram=y_gram,
            is_regularized=is_regularized,
            regularization_weights=regularization_weights,
        )
        lambda_path = np.geomspace(
            lambda_max, lambda_max * self.lambda_eps, self.lambda_n
        )

        if constraint_matrix is None:
            constraint_matrix = self.constraint_matrix
            logger.debug("Using constraint matrix from class attribute.")
        if constraint_bounds is None:
            constraint_bounds = self.constraint_bounds
            logger.debug("Using constraint bounds from class attribute.")

        self._raise_missing_constraint(constraint_matrix, constraint_bounds)

        beta_path = np.zeros((self.lambda_n, x_gram.shape[0]))
        beta_path, _ = online_linear_constrained_coordinate_descent_path(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            early_stop=self.early_stop,
            beta_path=beta_path,
            lambda_path=lambda_path,
            alpha=self.alpha,
            is_regularized=is_regularized,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            which_start_value=self.start_value_initial,
            regularization_weights=regularization_weights,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            relaxation_method=self.relaxation_method,
            max_dual_iterations=self.max_dual_iterations,
        )
        return beta_path

    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized, **kwargs):
        logger.debug(f"Got following kwargs: {[*kwargs.keys()]}")
        regularization_weights = kwargs.get("regularization_weights", None)
        constraint_matrix = kwargs.get("constraint_matrix", None)
        constraint_bounds = kwargs.get("constraint_bounds", None)

        beta_lower_bound = kwargs.get("beta_lower_bound", self.beta_lower_bound)
        beta_upper_bound = kwargs.get("beta_upper_bound", self.beta_upper_bound)
        self._validate_bounds(
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            x_gram=x_gram,
        )

        if self.auto_regularization_weights:
            if regularization_weights is not None:
                logger.warning(
                    "Both automatic regularization weights and user-defined regularization weights are provided. "
                    "Using automatic regularization weights."
                )
            regularization_weights = self._calculate_regularization_weights(
                x_gram=x_gram
            )

        lambda_max = self._get_lambda_max(
            x_gram=x_gram,
            y_gram=y_gram,
            is_regularized=is_regularized,
            regularization_weights=regularization_weights,
        )
        lambda_path = np.geomspace(
            lambda_max, lambda_max * self.lambda_eps, self.lambda_n
        )

        if constraint_matrix is None:
            constraint_matrix = self.constraint_matrix
            logger.debug("Using constraint matrix from class attribute.")
        if constraint_bounds is None:
            constraint_bounds = self.constraint_bounds
            logger.debug("Using constraint bounds from class attribute.")

        self._raise_missing_constraint(constraint_matrix, constraint_bounds)

        beta_path, _ = online_linear_constrained_coordinate_descent_path(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta_path=beta_path,
            lambda_path=lambda_path,
            alpha=self.alpha,
            early_stop=self.early_stop,
            is_regularized=is_regularized,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
            which_start_value=self.start_value_update,
            regularization_weights=regularization_weights,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            relaxation_method=self.relaxation_method,
            max_dual_iterations=self.max_dual_iterations,
        )
        return beta_path
