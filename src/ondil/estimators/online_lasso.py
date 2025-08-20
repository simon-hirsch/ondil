from typing import Literal

import numpy as np

from ..estimators.online_linear_model import OnlineLinearModel
from ..methods import LassoPath


class OnlineLasso(OnlineLinearModel):
    def __init__(
        self,
        forget: float = 0,
        scale_inputs: bool = True,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        ic: Literal["aic", "bic", "hqc", "max"] = "bic",
        early_stop: int = 0,
        beta_lower_bound: np.ndarray | None = None,
        beta_upper_bound: np.ndarray | None = None,
        lambda_n: int = 100,
        lambda_eps: float = 1e-4,
        start_value: str = "previous_fit",
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        """Online LASSO estimator class.

        This class initializes the online linear regression fitted using LASSO. The estimator object provides three main methods,
        ``estimator.fit(X, y)``, ``estimator.update(X, y)`` and ``estimator.predict(X)``.

        Args:
            forget (float, optional): Exponential discounting of old observations. Defaults to 0.
            scale_inputs (bool, optional): Whether to scale the $X$ matrix. Defaults to True.
            fit_intercept (bool, optional): Whether to add an intercept in the estimation. Defaults to True.
            regularize_intercept (bool, optional): Whether to regularize the intercept. Defaults to False.
            ic (Literal["aic", "bic", "hqc", "max"], optional): The information criteria for model selection. Defaults to "bic".
            early_stop (int, optional): Early stopping criterion. If we reach `early_stop` non-zero coefficients, we stop. Defaults to 0 (no early stopping).
            beta_lower_bound (np.ndarray | None, optional): Lower bounds for beta. Keep in mind the size of X and whether you want to fit an intercept. None corresponds to unconstrained estimation.Defaults to None.
            beta_upper_bound (np.ndarray | None, optional): Lower bounds for beta. Keep in mind the size of X and whether you want to fit an intercept. None corresponds to unconstrained estimation. Defaults to None.
            lambda_n (int, optional): Length of the regularization path. Defaults to 100.
            lambda_eps (float, optional): The largest regularization is determined automatically such that the solution is fully regularized. The smallest regularization is taken as $\\varepsilon  \\lambda^\max$ and we will use an exponential grid. Defaults to 1e-4.
            start_value (str, optional): Whether to choose the previous fit or the previous regularization as start value. Defaults to 100.
            tolerance (float, optional): Tolerance for breaking the CD. Defaults to 1e-4.
            max_iterations (int, optional): Max number of CD iterations. Defaults to 1000.
            selection (Literal["cyclic", "random"], optional): Whether to cycle through all coordinates in order or random. For large problems, random might increase convergence. Defaults to 100.
        """

        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.start_value = start_value
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.selection = selection
        self.beta_lower_bound = beta_lower_bound
        self.beta_upper_bound = beta_upper_bound
        self.early_stop = early_stop

        concrete_method = LassoPath(
            lambda_eps=lambda_eps,
            lambda_n=lambda_n,
            start_value_initial="previous_lambda",
            early_stop=early_stop,
            start_value_update=start_value,
            tolerance=tolerance,
            max_iterations=max_iterations,
            selection=selection,
            beta_lower_bound=beta_lower_bound,
            beta_upper_bound=beta_upper_bound,
        )
        super().__init__(
            forget=forget,
            scale_inputs=scale_inputs,
            fit_intercept=fit_intercept,
            regularize_intercept=regularize_intercept,
            method=concrete_method,
            ic=ic,
        )
