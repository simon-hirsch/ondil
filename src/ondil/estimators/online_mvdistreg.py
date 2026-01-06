import copy
import numbers
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
import warnings

import numba as nb
import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from ..base import Distribution, OndilEstimatorMixin
from ..design_matrix import make_intercept
from ..distributions import MultivariateNormalInverseCholesky
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..scaler import OnlineScaler
from ..types import ParameterShapes
from ..utils import calculate_effective_training_length
from ..validation import ensure_atleast_bivariate, validate_response_support

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def indices_along_diagonal(D: int) -> List:
    """This functions generates a list of indices that will go along
    the diagonal first and then go along the off-diagonals for a upper/lower
    triangular matrix.

    Args:
        D (int): Dimension of the matrix.

    Returns:
        List: List of indices
    """
    K = []
    for i in range(D):
        K.append(i)
        for j in range(D, i + 1, -1):
            K.append((K[-1] + j))
    return K


@nb.jit(
    [
        "float32[:, :](float32[:], float32[:], float32[:, :], boolean)",
        "float64[:, :](float64[:], float64[:], float64[:, :], boolean)",
    ],
    parallel=False,
)
def fast_vectorized_interpolate(x, xp, fp, ascending=True):
    dim = fp.shape[1]
    out = np.zeros((x.shape[0], dim), dtype=fp.dtype)
    if not ascending:
        x_asc = x[::-1]
        xp_asc = xp[::-1]

    for i in range(dim):
        if ascending:
            out[:, i] = np.interp(
                x=x,
                xp=xp,
                fp=fp[:, i],
            )
        else:
            out[:, i] = np.interp(
                x=x_asc,
                xp=xp_asc,
                fp=fp[:, i][::-1],
            )[::-1]
    return out


def make_model_array(X, eq, fit_intercept):
    n = X.shape[0]

    if isinstance(eq, str) and (eq == "intercept"):
        if not fit_intercept:
            raise ValueError(
                "fit_intercept[param] is false, but equation says intercept."
            )
        out = make_intercept(n_observations=n)
    else:
        if isinstance(eq, str) and (eq == "all"):
            if isinstance(X, np.ndarray):
                out = X
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                out = X.to_numpy()
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                out = X.to_numpy()
        elif isinstance(eq, np.ndarray) or isinstance(eq, list):
            if isinstance(X, np.ndarray):
                out = X[:, eq]
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                out = X.loc[:, eq]
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                out = X.select(eq).to_numpy()
        else:
            raise ValueError("Did not understand equation. Please check.")

        if fit_intercept:
            out = np.hstack((make_intercept(n), out))

    return out


class MultivariateOnlineDistributionalRegressionPath(
    OndilEstimatorMixin, RegressorMixin, MultiOutputMixin, BaseEstimator
):
    _parameter_constraints = {
        "distribution": [callable],
        "equation": [dict, type(None)],
        "forget": [Interval(numbers.Real, 0.0, 1.0, closed="left"), dict],
        "learning_rate": [Interval(numbers.Real, 0.0, 1.0, closed="left")],
        "fit_intercept": [bool, dict],
        "regularize_intercept": [bool, dict],
        "scale_inputs": [bool, np.ndarray],
        "verbose": [Interval(numbers.Integral, 0, None, closed="left")],
        "method": [StrOptions({"ols", "lasso"}), dict],
        "ic": [StrOptions({"ll", "aic", "bic", "hqc", "max"})],
        "iteration_along_diagonal": [bool],
        "approx_fast_model_selection": [bool],
        "max_regularisation_size": [
            Interval(numbers.Integral, 1, None, closed="left"),
            type(None),
        ],
        "early_stopping": [bool],
        "early_stopping_criteria": [StrOptions({"ll", "aic", "bic", "hqc", "max"})],
        "early_stopping_abs_tol": [Interval(numbers.Real, 0.0, None, closed="left")],
        "early_stopping_rel_tol": [Interval(numbers.Real, 0.0, None, closed="left")],
        "weight_delta": [numbers.Real, dict],
        "max_iterations_inner": [Interval(numbers.Integral, 1, None, closed="left")],
        "max_iterations_outer": [Interval(numbers.Integral, 1, None, closed="left")],
        "overshoot_correction": [dict, type(None)],
        "dampen_estimation": [
            bool,
            Interval(numbers.Integral, 0, None, closed="left"),
            dict,
        ],
        "debug": [bool],
        "rel_tol_inner": [Interval(numbers.Real, 0.0, None, closed="left")],
        "abs_tol_inner": [Interval(numbers.Real, 0.0, None, closed="left")],
        "rel_tol_outer": [Interval(numbers.Real, 0.0, None, closed="left")],
        "abs_tol_outer": [Interval(numbers.Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        distribution: Distribution = MultivariateNormalInverseCholesky(),
        equation: Dict | None = None,
        forget: float | Dict = 0.0,
        learning_rate: float = 0.0,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        scale_inputs: bool = True,
        verbose: int = 1,
        method: Literal["ols", "lasso"] | Dict[int, Literal["ols", "lasso"]] = "ols",
        ic: Literal["ll", "aic", "bic", "hqc", "max"] = "aic",
        iteration_along_diagonal: bool = False,
        approx_fast_model_selection: bool = True,
        max_regularisation_size: int | None = None,
        early_stopping: bool = True,
        early_stopping_criteria: Literal["ll", "aic", "bic", "hqc", "max"] = "aic",
        early_stopping_abs_tol: float = 0.001,
        early_stopping_rel_tol: float = 0.001,
        weight_delta: float | Dict[int, float] = 1.0,
        max_iterations_inner: int = 10,
        max_iterations_outer: int = 10,
        overshoot_correction: Optional[Dict[int, float]] = None,
        dampen_estimation: bool | int = False,
        debug: bool = False,
        rel_tol_inner: float = 1e-3,
        abs_tol_inner: float = 1e-3,
        rel_tol_outer: float = 1e-3,
        abs_tol_outer: float = 1e-3,
    ):
        """Initialize the Online Multivariate Distributional Regression Path Estimator.

        This estimator fits a multivariate distribution to the data using an online approach.
        It regularizes the scale matrix by estimating it sequentially from an "independence" configuration to a full covariance matrix,
        using the AD-R regularization scheme. Supports various regression methods, model selection criteria, early stopping, and input scaling.

        Args:
            distribution (Distribution, optional): The multivariate distribution to fit. Defaults to MultivariateNormalInverseCholesky().
            equation (Dict | None, optional): Dictionary specifying the regression equation. Defaults to None.
            forget (float | Dict, optional): Forgetting factor for online updates, can be a float or per-dimension dict. Defaults to 0.0.
            learning_rate (float, optional): Learning rate for online updates. Defaults to 0.0.
            fit_intercept (bool, optional): Whether to fit an intercept term. Defaults to True.
            regularize_intercept (bool, optional): Whether to regularize the intercept term. Defaults to False.
            scale_inputs (bool, optional): Whether to scale input features. Defaults to True.
            verbose (int, optional): Verbosity level for logging. Defaults to 1.
            method (Literal["ols", "lasso"] | Dict[int, Literal["ols", "lasso"]], optional): Regression method(s) to use, either "ols", "lasso", or per-dimension dict. Defaults to "ols".
            ic (Literal["ll", "aic", "bic", "hqc", "max"], optional): Information criterion for model selection. Defaults to "aic".
            iteration_along_diagonal (bool, optional): If True, iterates regularization along the diagonal of the scale matrix. Defaults to False.
            approx_fast_model_selection (bool, optional): If True, uses approximate fast model selection. Defaults to True.
            max_regularisation_size (int | None, optional): Maximum size for regularization path. Defaults to None.
            early_stopping (bool, optional): Enables early stopping during regularization. Defaults to True.
            early_stopping_criteria (Literal["ll", "aic", "bic", "hqc", "max"], optional): Criterion for early stopping. Defaults to "aic".
            early_stopping_abs_tol (float, optional): Absolute tolerance for early stopping. Defaults to 0.001.
            early_stopping_rel_tol (float, optional): Relative tolerance for early stopping. Defaults to 0.001.
            weight_delta (float | Dict[int, float], optional): Step size for weight updates, can be float or per-dimension dict. Defaults to 1.0.
            max_iterations_inner (int, optional): Maximum number of inner iterations. Defaults to 10.
            max_iterations_outer (int, optional): Maximum number of outer iterations. Defaults to 10.
            overshoot_correction (Optional[Dict[int, float]], optional): Correction factors for overshooting during updates. Defaults to None.
            dampen_estimation (bool | int, optional): If True or int, dampens estimation updates. Defaults to False.
            debug (bool, optional): Enables debug mode for additional logging. Defaults to False.
            rel_tol_inner (float, optional): Relative tolerance for convergence in inner loop. Defaults to 1e-3.
            abs_tol_inner (float, optional): Absolute tolerance for convergence in inner loop. Defaults to 1e-3.
            rel_tol_outer (float, optional): Relative tolerance for convergence in outer loop. Defaults to 1e-3.
            abs_tol_outer (float, optional): Absolute tolerance for convergence in outer loop. Defaults to 1e-3.
        """

        self.distribution = distribution
        self.forget = forget
        self.fit_intercept = fit_intercept
        self.regularize_intercept = regularize_intercept
        self.dampen_estimation = dampen_estimation
        self.weight_delta = weight_delta

        self.method = method
        self.learning_rate = learning_rate
        self.equation = equation
        self.iteration_along_diagonal = iteration_along_diagonal
        self.overshoot_correction = overshoot_correction

        # Early stopping
        self.max_regularisation_size = max_regularisation_size
        self.early_stopping = early_stopping
        self.early_stopping_criteria = early_stopping_criteria
        self.early_stopping_abs_tol = early_stopping_abs_tol
        self.early_stopping_rel_tol = early_stopping_rel_tol

        # Scaler
        self.scale_inputs = scale_inputs

        # For LASSO
        self.ic = ic
        self.approx_fast_model_selection = approx_fast_model_selection

        # Iterations and other internal stuff
        self.max_iterations_outer = max_iterations_outer
        self.max_iterations_inner = max_iterations_inner
        self.rel_tol_inner = rel_tol_inner
        self.abs_tol_inner = abs_tol_inner
        self.rel_tol_outer = rel_tol_outer
        self.abs_tol_outer = abs_tol_outer

        # For pretty printing.
        self.verbose = verbose
        self.debug = debug

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.two_d_labels = True
        tags.target_tags.single_output = False
        tags.target_tags.multi_output = True
        tags.input_tags.sparse = False
        tags.regressor_tags.poor_score = True
        return tags

    def _process_parameter(self, attribute: Any, default: Any, name: str) -> None:
        if isinstance(attribute, dict):
            for p in range(self.distribution.n_params):
                if p not in attribute.keys():
                    warnings.warn(
                        f"[{self.__class__.__name__}] "
                        f"No value given for parameter {name} for distribution "
                        f"parameter {p}. Setting default value {default}.",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    if isinstance(default, dict):
                        attribute[p] = default[p]
                    else:
                        attribute[p] = default
        else:
            # No warning since we expect that floats/strings/ints are either the defaults
            # Or given on purpose for all params the ame
            attribute = {p: attribute for p in range(self.distribution.n_params)}

        return attribute

    def _prepare_estimator(self):
        self._scaler = OnlineScaler(
            forget=self.learning_rate,
            to_scale=self.scale_inputs,
        )
        # For simplicity
        self._forget = self._process_parameter(
            self.forget,
            default=0.0,
            name="forget",
        )
        self._fit_intercept = self._process_parameter(
            self.fit_intercept,
            default=True,
            name="fit_intercept",
        )
        self._regularize_intercept = self._process_parameter(
            self.regularize_intercept,
            default=False,
            name="regularize_intercept",
        )
        self._dampen_estimation = self._process_parameter(
            self.dampen_estimation,
            default=False,
            name="dampen_estimation",
        )
        self._weight_delta = self._process_parameter(
            self.weight_delta,
            default=1.0,
            name="weight_delta",
        )
        self._overshoot_correction = self._process_parameter(
            self.overshoot_correction,
            default=None,
            name="overshoot_correction",
        )

    def _make_iteration_indices(self, param: int):
        if (
            self.distribution.parameter_shape
            in [
                ParameterShapes.SQUARE_MATRIX,
                ParameterShapes.LOWER_TRIANGULAR_MATRIX,
                ParameterShapes.UPPER_TRIANGULAR_MATRIX,
            ]
        ) & self.iteration_along_diagonal:
            index = np.arange(indices_along_diagonal(self.dim_[param]))
        else:
            index = np.arange(self.n_dist_elements_[param])

        return index

    def _make_matrix_or_intercept(
        self, n_observations: int, x: np.ndarray, add_intercept: bool, param: int
    ):
        """
        Make the covariate matrix or an intercept array based on the input.

        Args:
            y (np.ndarray): Response variable.
            x (np.ndarray): Covariate matrix.
            add_intercept (bool): Flag indicating whether to add an intercept.
            param (int): Parameter index.

        Returns:
            np.ndarray: Matrix or intercept array.
        """
        if x is None:
            return make_intercept(n_observations=n_observations)
        elif add_intercept[param]:
            return self._add_intercept(x=x, param=param)
        else:
            return x

    # Different UV - MV
    def _make_initial_theta(self, y: np.ndarray):
        theta = {
            a: {
                p: self.distribution.initial_values(y, p)
                for p in range(self.distribution.n_params)
            }
            for a in range(self.adr_steps_)
        }
        # Handle AD-R Regularization
        for a in range(self.adr_steps_):
            for p in range(self.distribution.n_params):
                theta = self._handle_path_regularization(theta=theta, p=p, a=a)

        return theta

    def _handle_path_regularization(self, theta, p: int, a: int):
        if self.distribution._regularization_allowed[p]:
            mask = (
                self._adr_distance[p] >= self._adr_mapping_index_to_max_distance[p][a]
            )
            regularized = self.distribution.cube_to_flat(theta[a][p], p)
            regularized[:, mask] = regularized[:, mask] = 0
            theta[a][p] = self.distribution.flat_to_cube(regularized, p)
            return theta
        else:
            # If no regularization is allowed, we just return the theta
            return theta

    # Only MV
    def _is_element_adr_regularized(self, p: int, k: int, a: int):
        if not self.distribution._regularization_allowed[p]:
            return False
        else:
            return (
                self._adr_distance[p][k]
                >= self._adr_mapping_index_to_max_distance[p][a]
            )

    # Only MV
    # This should be using the distribution
    def _prepare_adr_regularization(self) -> None:
        self._adr_distance = {}
        self._adr_mapping_index_to_max_distance = {
            p: np.arange(1, self.adr_steps_ + 1)
            for p in range(self.distribution.n_params)
            if self.distribution._regularization_allowed[p]
        }
        for p in range(self.distribution.n_params):
            if self.distribution._regularization_allowed[p]:
                self._adr_distance[p] = (
                    self.distribution.get_adr_regularization_distance(
                        dim=self.dim_, param=p
                    )
                )

    # Different UV-MV
    def _get_number_of_covariates(self, X: np.ndarray):
        J = {}
        for p in range(self.distribution.n_params):
            J[p] = {}
            for k in range(self.n_dist_elements_[p]):
                if isinstance(self._equation[p][k], str):
                    if self._equation[p][k] == "all":
                        J[p][k] = X.shape[1] + int(self._fit_intercept[p])
                    if self._equation[p][k] == "intercept":
                        J[p][k] = 1
                elif isinstance(self._equation[p][k], np.ndarray) or isinstance(
                    self._equation[p][k], list
                ):
                    J[p][k] = len(self._equation[p][k]) + int(self._fit_intercept[p])
                else:
                    raise ValueError("Something unexpected happened")
        return J

    def _prepare_estimation_method(self, y):
        dim = y.shape[1]
        method = {
            p: {k: get_estimation_method(m) for k in range(K)}
            for p, (m, K) in enumerate(
                zip(
                    self._process_parameter(
                        self.method, default="ols", name="method"
                    ).values(),
                    self.distribution.fitted_elements(dim).values(),
                )
            )
        }
        return method

    def _check_path_based_estimation_methods(self):
        self._lambda_n = {}

        for p in range(self.distribution.n_params):
            lmbdas = []
            for k in range(self.n_dist_elements_[p]):
                if self._method[p][k]._path_based_method:
                    lmbdas.append(self._method[p][k]._path_length)
            if len(set(lmbdas)) > 1:
                raise ValueError(
                    f"Distribution parameter {p} has different path lengths for different elements. "
                    "This is not supported by the current implementation. "
                    "Please use a single path length for all elements of the distribution parameter."
                )
            if len(set(lmbdas)) == 1:
                self._lambda_n[p] = lmbdas[0]
            else:
                # If no path length is given, we assume that the method is not path-based
                # and we set the path length to 1.
                self._lambda_n[p] = 1

    def _get_next_adr_start_values(self, theta, p, a):
        mask = (
            self._adr_distance[p] >= self._adr_mapping_index_to_max_distance[p][a - 1]
        )
        prev_est = self.distribution.cube_to_flat(theta[a - 1][p], p)
        regularized = self.distribution.cube_to_flat(theta[a][p], p)
        regularized[:, ~mask] = prev_est[:, ~mask]
        start_value = self.distribution.flat_to_cube(regularized, p)
        return start_value

    # Different UV-MV
    def validate_equation(self, equation):
        if equation is None:
            warnings.warn(
                f"[{self.__class__.__name__}] "
                "Equation is not specified. "
                "Per default, will estimate the first distribution parameter by all covariates found in X. "
                "All other distribution parameters will be estimated by an intercept."
            )
            equation = {
                p: (
                    {k: "all" for k in range(self.n_dist_elements_[p])}
                    if p == 0
                    else {k: "intercept" for k in range(self.n_dist_elements_[p])}
                )
                for p in range(self.distribution.n_params)
            }
        else:
            for p in range(self.distribution.n_params):
                # Check that all distribution parameters are in the equation.
                # If not, add intercept.
                if p not in equation.keys():
                    print(
                        f"{self.__class__.__name__}",
                        f"Distribution parameter {p} is not in equation.",
                        "All elements of the parameter will be estimated by an intercept.",
                    )
                    equation[p] = {
                        k: "intercept" for k in range(self.n_dist_elements_[p])
                    }

                else:
                    for k in range(self.n_dist_elements_[p]):
                        if k not in equation[p].keys():
                            message = (
                                f"Distribution parameter {p}, element {k} is not in equation.",
                                "Element of the parameter will be estimated by an intercept.",
                            )
                            self._print_message(level=1, message=message)
                            equation[p][k] = "intercept"

                    if not (
                        isinstance(equation[p][k], np.ndarray)
                        or (equation[p][k] in ["all", "intercept"])
                    ):
                        if not (
                            isinstance(equation[p], list) and (HAS_PANDAS | HAS_POLARS)
                        ):
                            raise ValueError(
                                "The equation should contain of either: \n"
                                " - a numpy array of dtype int, \n"
                                " - a list of string column names \n"
                                " - or the strings 'all' or 'intercept' \n"
                                f"you have passed {equation[p]} for the distribution parameter {p}."
                            )

        return equation

    # Different UV - MV
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "MultivariateOnlineDistributionalRegressionPath":
        """Fit the estimator to the data.

        !!! Note
            The response $y$ must be a multivariate response
            variable, i.e., a 2D array where each column represents
            a different response variable and we expect at least a
            bivariate response. Hence, passing an `(n, 1)` array
            for y does not work!

        Args:
            X (np.ndarray): Covariate or feature matrix.
            y (np.ndarray): Multivariate response variable.

        Returns:
            estimator: Returns the fitted estimator.
        """

        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            dtype=[np.float64, np.float32],
            multi_output=True,
            ensure_min_samples=2,
        )
        ensure_atleast_bivariate(y=y)
        validate_response_support(self, y=y)
        _ = type_of_target(y, raise_unknown=True)

        # Prepare the estimator
        self._prepare_estimator()

        # Set fixed values
        self.dim_ = y.shape[1]
        self.n_observations_ = y.shape[0]
        self.n_training_ = calculate_effective_training_length(
            forget=self.learning_rate, n_obs=self.n_observations_
        )

        self.n_dist_elements_ = self.distribution.fitted_elements(self.dim_)
        # Validate the equation and set the method
        # Ensure that we have the same number of lambda_n for all methods per distr param.
        self._method = self._prepare_estimation_method(y=y)
        self._check_path_based_estimation_methods()
        self._equation = self.validate_equation(self.equation)
        self.n_features_ = self._get_number_of_covariates(X)

        # Without prior information, the optimal ADR regularization is
        # The largest model
        self.adr_steps_ = self.distribution.get_regularization_size(self.dim_)
        if self.max_regularisation_size is not None:
            self.adr_steps_ = np.fmin(self.adr_steps_, self.max_regularisation_size)

        self.optimal_adr_ = self.adr_steps_

        # Empty coefficient dictionaries
        self.coef_ = {}
        self.coef_path_ = {}

        # Handle scaling
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X=X)

        # Create the regularization mask
        # For the "lower estimation" i.e. the CD path.
        self.is_regularized_ = {
            p: {
                k: np.repeat(True, self.n_features_[p][k])
                for k in range(self.n_dist_elements_[p])
            }
            for p in range(self.distribution.n_params)
        }
        for p in range(self.distribution.n_params):
            for k in range(self.n_dist_elements_[p]):
                if not self._regularize_intercept[p]:
                    if self._fit_intercept[p]:
                        self.is_regularized_[p][k][0] = False

        # Iteration index for the elements of each distribution parameter
        self._iter_index = {
            p: self._make_iteration_indices(p)
            for p in range(self.distribution.n_params)
        }

        self._prepare_adr_regularization()
        theta = self._make_initial_theta(y)

        # Current information
        self._current_likelihood = np.array([
            np.sum(self.distribution.logpdf(y=y, theta=theta[a]))
            for a in range(self.adr_steps_)
        ])
        self._model_selection = {
            p: {a: {} for a in range(self.adr_steps_)}
            for p in range(self.distribution.n_params)
        }
        self._x_gram = {
            p: {
                k: np.empty((
                    self.adr_steps_,
                    self.n_features_[p][k],
                    self.n_features_[p][k],
                ))
                for k in range(self.n_dist_elements_[p])
            }
            for p in range(self.distribution.n_params)
        }
        self._y_gram = {
            p: {
                k: np.empty((self.adr_steps_, self.n_features_[p][k]))
                for k in range(self.n_dist_elements_[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.coef_path_ = {
            p: {
                k: np.zeros((
                    self.adr_steps_,
                    self._lambda_n[p],
                    self.n_features_[p][k],
                ))
                for k in range(self.n_dist_elements_[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.coef_ = {
            p: {
                k: np.zeros((self.adr_steps_, self.n_features_[p][k]))
                for k in range(self.n_dist_elements_[p])
            }
            for p in range(self.distribution.n_params)
        }

        # Some information about the different iterations
        self.iteration_count_ = np.zeros(
            (self.max_iterations_outer, self.distribution.n_params, self.adr_steps_),
            dtype=int,
        )
        self.iteration_likelihood_ = np.zeros((
            self.max_iterations_outer,
            self.max_iterations_inner,
            self.distribution.n_params,
            self.adr_steps_,
        ))
        # Call the fit
        self._outer_fit(X=X_scaled, y=y, theta=theta)
        self._print_message(message="Finished fitting distribution parameters.")

        return self

    def _outer_fit(self, X, y, theta):
        adr_start = time.time()

        for a in range(self.adr_steps_):
            adr_it_start = time.time()
            outer_start = time.time()

            global_old_likelihood = 0
            converged = False
            decreasing = False

            for outer_iteration in range(self.max_iterations_outer):
                outer_it_start = time.time()
                # Main outer loop
                for p in range(self.distribution.n_params):
                    message = (
                        f"Outer Iteration: {outer_iteration}, "
                        f"fitting Distribution parameter {p}, AD-r step {a}"
                    )
                    self._print_message(level=2, message=message)
                    theta = self._inner_fit(
                        X=X,
                        y=y,
                        theta=theta,
                        outer_iteration=outer_iteration,
                        p=p,
                        a=a,
                    )
                # Get the global likelihood
                global_likelihood = self._current_likelihood[a]

                converged, _ = self._check_outer_convergence(
                    global_old_likelihood, global_likelihood
                )

                # start value for the next AD-R step
                if (
                    converged
                    | decreasing
                    | (outer_iteration == self.max_iterations_outer - 1)
                ):
                    if a < (self.adr_steps_ - 1):
                        for p in range(self.distribution.n_params):
                            if not self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = copy.deepcopy(theta[a][p])
                            if self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = self._get_next_adr_start_values(
                                    theta, p, a + 1
                                )

                # Timings
                outer_it_end = time.time()
                outer_it_time = outer_it_end - outer_it_start
                outer_it_avg = (outer_it_end - outer_start) / (outer_iteration + 1)
                outer_pred_time = outer_it_time * (
                    self.max_iterations_outer - outer_iteration - 1
                )
                message = (
                    f"Last outer iteration {outer_iteration} took {round(outer_it_time, 1)} sec. "
                    f"Average outer iteration took {round(outer_it_avg, 1)} sec. "
                    f"Expected to be finished in max {round(outer_pred_time, 1)} sec. "
                )
                self._print_message(level=2, message=message)

                # End timing for the iteration
                adr_it_end = time.time()

                if converged:
                    break
                else:
                    global_old_likelihood = global_likelihood

            # Timings
            adr_it_last = adr_it_end - adr_it_start
            adr_it_avg = (adr_it_end - adr_start) / (a + 1)
            message = (
                f"Last ADR iteration {a} took {round(adr_it_last, 1)} sec. "
                f"Average ADR iteration took {round(adr_it_avg, 1)} sec. "
            )
            self._print_message(level=1, message=message)

            # Calculate the improvement
            if self.early_stopping_criteria == "ll":
                # Check the lilelihood for early stopping
                self.improvement_abs_ = -np.diff(self._current_likelihood)
                self.improvement_abs_scaled_ = -self.improvement_abs_ / self.n_training_
                self.improvement_rel_ = (
                    self.improvement_abs_ / self._current_likelihood[-1:]
                )

            elif self.early_stopping_criteria in ["aic", "bic", "hqc", "max"]:
                self._early_stopping_n_params = np.array([
                    self.count_nonzero_coef(self.coef_, a)
                    for a in range(self.adr_steps_)
                ])
                self.early_stopping_ic_ = InformationCriterion(
                    n_observations=self.n_training_,
                    n_parameters=self._early_stopping_n_params,
                    criterion=self.early_stopping_criteria,
                ).from_ll(self._current_likelihood)

                self.improvement_abs_ = -np.diff(self.early_stopping_ic_)
                self.improvement_abs_scaled_ = self.improvement_abs_
                self.improvement_rel_ = (
                    self.improvement_abs_ / self.early_stopping_ic_[-1:]
                )
            else:
                raise ValueError(
                    "Did not recognize criteria AD-R regularization stopping criteria."
                )

            if self.early_stopping and (a > 0) and (a < self.adr_steps_ - 1):
                # In the last step, it does not make sense to "early stop"

                if (
                    self.improvement_abs_scaled_[a - 1] < self.early_stopping_abs_tol
                ) or (self.improvement_rel_[a - 1] < self.early_stopping_rel_tol):
                    message = (
                        f"Early stopping due to AD-r-regression. "
                        f"Last increase in r lead to relative improvement: {self.improvement_rel_[a - 1]}, scaled absolute improvement {self.improvement_abs_scaled_[a - 1]}",
                    )
                    self._print_message(level=1, message=message)

                    if self.improvement_rel_[a - 1] > 0:
                        self.optimal_adr_ = a
                    elif self.improvement_rel_[a - 1] < 0:
                        self.optimal_adr_ = a - 1

                    self.improvement_abs_[(a + 1) :] = 0
                    self.improvement_rel_[(a + 1) :] = 0
                    self.improvement_abs_scaled_[(a + 1) :] = 0
                    break
            else:
                # The largest theta is the optimal one
                # But might be overfit
                self.optimal_adr_ = a

        self.theta_ = theta
        self.optimal_theta_ = self.theta_[self.optimal_adr_]
        self.last_fit_adr_max_ = self.optimal_adr_

    @staticmethod
    def count_nonzero_coef(beta, adr):
        non_zero = 0
        for p, coef in beta.items():
            non_zero += int(np.sum([np.sum(c[adr, :] != 0) for k, c in coef.items()]))
        return non_zero

    def _count_coef_to_be_fitted_param(self, adr: int, param: int, k: int):
        """Count the coefficients that should be fitted for this param"""
        idx_k = self._iter_index[param].tolist().index(k)
        if self.distribution._regularization_allowed[param]:
            count = 0
            for kk in self._iter_index[param][idx_k:]:
                count += int(not self._is_element_adr_regularized(param, kk, adr))
        else:
            count = len(self._iter_index[param][idx_k:])
        return count

    def count_coef_to_be_fitted(
        self, outer_iteration: int, inner_iteration: int, adr: int, param: int, k: int
    ):
        """Count all coefficients that should be fitted."""
        if (outer_iteration > 0) or (inner_iteration > 0):
            count = 0
        else:
            count = 0
            for p in range(param, self.distribution.n_params):
                count += self._count_coef_to_be_fitted_param(
                    adr=adr, param=p, k=k if p == param else 0
                )
        return count

    @staticmethod
    def _check_convergence(old_value, new_value, rel_tol, abs_tol) -> Tuple[bool, bool]:
        # Account for floating point acc
        if np.abs(new_value - old_value) / np.abs(new_value) < rel_tol:
            converged = True
            decreasing = False
        elif np.abs(new_value - old_value) < abs_tol:
            converged = True
            decreasing = False
        elif (new_value < old_value) & (np.abs(new_value - old_value) > 1e-10):
            converged = False
            decreasing = True
        else:
            converged = False
            decreasing = False
        return converged, decreasing

    def _check_inner_convergence(self, old_value, new_value) -> Tuple[bool, bool]:
        return self._check_convergence(
            old_value=old_value,
            new_value=new_value,
            rel_tol=self.rel_tol_inner,
            abs_tol=self.abs_tol_inner,
        )

    def _check_outer_convergence(self, old_value, new_value) -> Tuple[bool, bool]:
        return self._check_convergence(
            old_value=old_value,
            new_value=new_value,
            rel_tol=self.rel_tol_outer,
            abs_tol=self.abs_tol_outer,
        )

    def get_dampened_prediction(
        self,
        prediction: np.ndarray,
        eta: np.ndarray,
        outer_iteration: int,
        inner_iteration: int,
        param: int,
    ):
        if (outer_iteration == 0) & (inner_iteration < self._dampen_estimation[param]):
            out = (prediction * self._dampen_estimation[param] + eta) / (
                self._dampen_estimation[param] + 1
            )
        else:
            out = prediction
        return out

    def _inner_fit(self, y, X, theta, outer_iteration, a, p):
        converged = False
        decreasing = False
        old_likelihood = self._current_likelihood[a]

        weights_forget = init_forget_vector(
            forget=self.learning_rate,
            size=self.n_observations_,
        )

        for inner_iteration in range(self.max_iterations_inner):
            # If the likelihood is at some point decreasing, we're breaking
            # Hence we need to store previous iteration values:
            if (inner_iteration > 0) | (outer_iteration > 0):
                prev_theta = copy.copy(theta)
                prev_x_gram = copy.copy(self._x_gram[p])
                prev_y_gram = copy.copy(self._y_gram[p])
                prev_model_selection = copy.copy(self._model_selection)
                prev_beta = copy.copy(self.coef_)
                prev_beta_path = copy.copy(self.coef_path_)

            # This will check if we
            if (inner_iteration == 0) and (outer_iteration == 0) & (a == 0):
                theta[a] = self.distribution.set_initial_guess(y, theta[a], p)
                theta = self._handle_path_regularization(theta=theta, p=p, a=a)

            # Iterate through all elements of the distribution parameter
            for k in self._iter_index[p]:
                if self.debug:
                    print("Fitting", outer_iteration, inner_iteration, p, k, a)

                if self._is_element_adr_regularized(p=p, k=k, a=a):
                    self.coef_[p][k][a] = np.zeros(self.n_features_[p][k])
                    self.coef_path_[p][k][a] = np.zeros((
                        self._lambda_n[p],
                        self.n_features_[p][k],
                    ))
                else:
                    eta = self.distribution.link_function(theta[a][p], p)
                    eta = self.distribution.cube_to_flat(eta, param=p)

                    # Derivatives wrt to the parameter
                    dl1dp1 = self.distribution.element_dl1_dp1(
                        y, theta=theta[a], param=p, k=k
                    )
                    dl2dp2 = self.distribution.element_dl2_dp2(
                        y, theta=theta[a], param=p, k=k, clip=False
                    )

                    dl1_link = self.distribution.link_function_derivative(
                        theta[a][p], p
                    )
                    dl2_link = self.distribution.link_function_second_derivative(
                        theta[a][p],
                        p,
                    )
                    dl1_link = self.distribution.cube_to_flat(dl1_link, param=p)
                    dl1_link = dl1_link[:, k]
                    dl2_link = self.distribution.cube_to_flat(dl2_link, param=p)
                    dl2_link = dl2_link[:, k]

                    dl1_deta1 = dl1dp1 * (1 / dl1_link)
                    dl2_deta2 = (dl2dp2 * dl1_link - dl1dp1 * dl2_link) / dl1_link**3

                    wt = np.fmax(-dl2_deta2, 1e-10)
                    wv = eta[:, k] + dl1_deta1 / wt

                    # Create the more arrays
                    x = make_model_array(
                        X=X,
                        eq=self._equation[p][k],
                        fit_intercept=self._fit_intercept[p],
                    )
                    if self.debug:
                        print(
                            "Rank of X",
                            outer_iteration,
                            inner_iteration,
                            p,
                            k,
                            a,
                            np.linalg.matrix_rank(x),
                        )

                    self._x_gram[p][k][a] = self._method[p][k].init_x_gram(
                        X=x,
                        weights=wt ** self._weight_delta[p],
                        forget=self._forget[p],
                    )
                    self._y_gram[p][k][a] = (
                        self
                        ._method[p][k]
                        .init_y_gram(
                            X=x,
                            y=wv,
                            weights=wt ** self._weight_delta[p],
                            forget=self._forget[p],
                        )
                        .squeeze()
                    )

                    if self._method[p][k]._path_based_method:
                        self.coef_path_[p][k][a] = self._method[p][k].fit_beta_path(
                            x_gram=self._x_gram[p][k][a],
                            y_gram=self._y_gram[p][k][a][:, None],
                            is_regularized=self.is_regularized_[p][k],
                        )
                        eta_elem = x @ self.coef_path_[p][k][a].T
                        theta_elem = self.distribution.element_link_inverse(
                            eta_elem, param=p, k=k, d=self.dim_
                        )

                        opt_ic = self._fit_model_selection(
                            y=y,
                            theta_fit=theta_elem,
                            theta=theta,
                            outer_iteration=outer_iteration,
                            inner_iteration=inner_iteration,
                            a=a,
                            k=k,
                            param=p,
                        )
                        # select optimal beta and theta
                        self.coef_[p][k][a] = self.coef_path_[p][k][a][opt_ic, :]
                        theta[a] = self.distribution.set_theta_element(
                            theta[a], theta_elem[:, opt_ic], param=p, k=k
                        )
                    else:
                        self.coef_path_[p][k][a] = None
                        self.coef_[p][k][a] = self._method[p][k].fit_beta(
                            x_gram=self._x_gram[p][k][a],
                            y_gram=self._y_gram[p][k][a][:, None],
                            is_regularized=self.is_regularized_[p][k],
                        )

                    eta[:, k] = self.get_dampened_prediction(
                        prediction=np.squeeze(x @ self.coef_[p][k][a]),
                        eta=eta[:, k],
                        inner_iteration=inner_iteration,
                        outer_iteration=outer_iteration,
                        param=p,
                    )
                    theta[a][p] = self.distribution.link_inverse(
                        self.distribution.flat_to_cube(eta, param=p), param=p
                    )
                    if (self._overshoot_correction[p] is not None) and (
                        inner_iteration + outer_iteration < 1
                    ):
                        theta[a] = self.distribution.set_theta_element(
                            theta[a],
                            theta[a][p][:, k] + self.overshoot_correction[p][k],
                            param=p,
                            k=k,
                        )

                self._current_likelihood[a] = (
                    self.distribution.logpdf(y, theta=theta[a]) * weights_forget
                ).sum()

            ## Check the most important convergence measures here now
            if inner_iteration == (self.max_iterations_inner - 1):
                warnings.warn(
                    "Reached max inner iterations. Algorithm may or may not be converged."
                )
            self.iteration_count_[outer_iteration, p, a] = inner_iteration
            self.iteration_likelihood_[outer_iteration, inner_iteration, p, a] = (
                self._current_likelihood[a]
            )
            message = (
                f"Outer iteration: {outer_iteration}, inner iteration {inner_iteration}, parameter {p}, AD-R {a}:"
                f"current likelihood: {self._current_likelihood[a]},"
                f"previous iteration likelihood {self.iteration_likelihood_[outer_iteration, inner_iteration - 1, p, a] if inner_iteration > 0 else self._current_likelihood[a]}"
            )
            self._print_message(level=2, message=message)

            if (inner_iteration > 0) & (
                (inner_iteration > self._dampen_estimation[p]) | (outer_iteration > 0)
            ):
                converged, decreasing = self._check_inner_convergence(
                    old_value=old_likelihood, new_value=self._current_likelihood[a]
                )

                if converged:
                    break
                else:
                    # For the next iteration
                    old_likelihood = self._current_likelihood[a]

            # If the LL is decreasing, we're resetting to the previous iteration
            if ((outer_iteration > 0) | (inner_iteration > 1)) & decreasing:
                warnings.warn("Likelihood is decreasing. Breaking.")
                # Reset to values from the previous iteration
                theta = prev_theta
                self._model_selection = prev_model_selection
                self._x_gram[p] = prev_x_gram
                self._y_gram[p] = prev_y_gram
                self.coef_ = prev_beta
                self.coef_path_ = prev_beta_path
                self._current_likelihood[a] = old_likelihood
                break

        return theta

    def _fit_model_selection(
        self,
        y,
        theta_fit,
        theta,
        outer_iteration,
        inner_iteration,
        a: int,
        k: int,
        param: int,
    ):
        weights_forget = init_forget_vector(
            self.learning_rate,
            self.n_observations_,
        )
        # Model selection
        if self.ic == "max":
            opt_ic = self._lambda_n[param] - 1
        else:
            theta_ll = copy.deepcopy(theta[a])
            theta_elem_delta = np.diff(theta_fit, axis=1)
            theta_ms = self.distribution.set_theta_element(
                theta_ll, theta_fit[:, 0], param=param, k=k
            )
            approx_ll = np.sum(self.distribution.logpdf(y, theta_ll) * weights_forget)
            approx_ll = np.repeat(approx_ll, 100)
            for l_idx in range(1, self._lambda_n[param]):
                theta_ms = self.distribution.set_theta_element(
                    theta_ll, theta_fit[:, l_idx], param=param, k=k
                )
                if self.approx_fast_model_selection:
                    approx_ll[l_idx] = approx_ll[l_idx - 1] + np.sum(
                        self.distribution.element_dl1_dp1(y, theta_ms, param=param, k=k)
                        * theta_elem_delta[:, l_idx - 1]
                        * weights_forget
                    )
                else:
                    approx_ll[l_idx] = np.sum(
                        self.distribution.logpdf(
                            y,
                            theta=theta_ms,
                        )
                        * weights_forget
                    )

            # Count number of nonzero coefficients
            # Subtract current beta if already fitted
            # If in the first iteration, add intercept
            # for all to-be-fitted parameters
            # that are not AD-R regularized
            nonzero = self.count_nonzero_coef(self.coef_, adr=a)
            nonzero = nonzero + int(np.sum(self.coef_[param][k][a, :] != 0))
            nonzero = nonzero + self.count_coef_to_be_fitted(
                outer_iteration, inner_iteration, adr=a, param=param, k=k
            )
            ic = InformationCriterion(
                n_observations=self.n_observations_,
                n_parameters=nonzero,
                criterion=self.ic,
            ).from_ll(log_likelihood=approx_ll)
            self._model_selection[param][a][k] = {
                "ll": approx_ll,
                "non_zero": nonzero,
                "ic": ic,
            }
            opt_ic = np.argmin(ic)

        return opt_ic

    def _update_model_selection(
        self,
        y,
        theta_fit: np.ndarray,
        theta: Dict,
        outer_iteration: int,
        inner_iteration: int,
        a: int,
        k: int,
        param: int,
    ) -> int:
        """
        Update the model selection.

        Returns the optimal IC's index.
        """
        weights_forget = init_forget_vector(
            self.learning_rate,
            self.n_observations_step_,
        )

        if self.ic == "max":
            opt_ic = self._lambda_n[param] - 1
        else:
            # Model selection
            theta_ll = copy.deepcopy(theta[a])

            theta_elem_delta = np.diff(theta_fit, axis=1)
            theta_ms = self.distribution.set_theta_element(
                theta_ll, theta_fit[:, 0], param=param, k=k
            )
            approx_ll = np.sum(self.distribution.logpdf(y, theta_ll) * weights_forget)
            approx_ll = np.repeat(approx_ll, 100)
            for l_idx in range(1, self._lambda_n[param]):
                theta_ms = self.distribution.set_theta_element(
                    theta_ll, theta_fit[:, l_idx], param=param, k=k
                )
                if self.approx_fast_model_selection:
                    approx_ll[l_idx] = approx_ll[l_idx - 1] + np.sum(
                        (
                            self.distribution.element_dl1_dp1(
                                y, theta_ms, param=param, k=k
                            )
                            * theta_elem_delta[:, l_idx - 1]
                        )
                        * weights_forget
                    )
                else:
                    approx_ll[l_idx] = np.sum(
                        self.distribution.logpdf(
                            y,
                            theta=theta_ms,
                        )
                        * weights_forget
                    )
            approx_ll = approx_ll + (
                self._model_selection_old[param][a][k]["ll"]
                * (1 - self.learning_rate) ** self.n_observations_step_
            )
            # Count number of nonzero coefficients
            # Subtract current beta if already fitted
            # If in the first iteration, add intercept
            # for all to-be-fitted parameters
            # that are not AD-R regularized
            nonzero = self.count_nonzero_coef(self.coef_, adr=a)
            nonzero = nonzero + int(np.sum(self.coef_[param][k][a, :] != 0))
            nonzero = nonzero + self.count_coef_to_be_fitted(
                outer_iteration, inner_iteration, adr=a, param=param, k=k
            )
            ic = InformationCriterion(
                n_observations=self.n_observations_,
                n_parameters=nonzero,
                criterion=self.ic,
            ).from_ll(log_likelihood=approx_ll)
            opt_ic = np.argmin(ic)
            self._model_selection[param][a][k] = {
                "ll": approx_ll,
                "non_zero": nonzero,
                "ic": ic,
            }

        return opt_ic

    # Different UV - MV
    def predict(
        self,
        X: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Predict the location parameter.

        Parameters
        ----------
        X : np.ndarray, optional
            The input data. If None, will use a default value of ones.

        """

        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, dtype=[np.float64, np.float32])

        if X is None:
            X_scaled = np.ones((1, 1))
            N = 1
            self._print_message("X is None. Prediction will have length 1.")
        else:
            X_scaled = self._scaler.transform(X=X)
            N = X.shape[0]

        array = np.zeros((N, self.n_dist_elements_[0]))
        for k in range(self.n_dist_elements_[0]):
            array[:, k] = (
                make_model_array(
                    X=X_scaled,
                    eq=self._equation[0][k],
                    fit_intercept=self._fit_intercept[0],
                )
                @ self.coef_[0][k][self.optimal_adr_, :]
            ).squeeze()
        out = self.distribution.flat_to_cube(array, 0)
        out = self.distribution.link_inverse(out, 0)
        return out

    # Different UV - MV
    def predict_distribution_parameters(
        self,
        X: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:
        """Predict the distribution parameters.

        Args:
            X (Optional[np.ndarray], optional): Covariate or feature matrix. Defaults to None.

        Returns:
            Dict[int, np.ndarray]: Return the predicted distribution parameters as a dictionary.
        """

        check_is_fitted(self)
        X = validate_data(
            self,
            X=X,
            reset=False,
            dtype=[np.float64, np.float32],
        )

        if X is None:
            X_scaled = np.ones((1, 1))
            N = 1
            self._print_message("X is None. Prediction will have length 1.")
        else:
            X_scaled = self._scaler.transform(X=X)
            N = X.shape[0]
        out = {}

        for p in range(self.distribution.n_params):
            array = np.zeros((N, self.n_dist_elements_[p]))
            for k in range(self.n_dist_elements_[p]):
                array[:, k] = (
                    make_model_array(
                        X=X_scaled,
                        eq=self._equation[p][k],
                        fit_intercept=self._fit_intercept[p],
                    )
                    @ self.coef_[p][k][self.optimal_adr_, :]
                ).squeeze()
            out[p] = self.distribution.flat_to_cube(array, p)
            out[p] = self.distribution.link_inverse(out[p], p)
        return out

    def predict_all_adr(
        self,
        X: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:
        """Predict all regualarized fits in one go.

        Args:
            X (Optional[np.ndarray], optional): Covariate or feature matrix. Defaults to None.

        Returns:
            Dict[int, np.ndarray]: Return the predicted distribution parameters for all AD-R steps as a dictionary. The structure is
            {adr_step: {param: np.ndarray}} where `adr_step` is the AD-R step index and `param` is the distribution parameter index.
        """

        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, dtype=[np.float64, np.float32])

        if X is None:
            X_scaled = np.ones((1, 1))
            N = 1
            self._print_message("X is None. Prediction will have length 1.")
        else:
            X_scaled = self._scaler.transform(X=X)
            N = X.shape[0]
        out = {}
        for a in range(self.adr_steps_):
            out[a] = {}
            if a <= self.optimal_adr_:
                for p in range(self.distribution.n_params):
                    array = np.zeros((N, self.n_dist_elements_[p]))
                    for k in range(self.n_dist_elements_[p]):
                        array[:, k] = (
                            make_model_array(
                                X=X_scaled,
                                eq=self._equation[p][k],
                                fit_intercept=self._fit_intercept[p],
                            )
                            @ self.coef_[p][k][a, :]
                        ).squeeze()
                    out[a][p] = self.distribution.flat_to_cube(array, p)
                    out[a][p] = self.distribution.link_inverse(out[a][p], p)
            else:
                out[a] = copy.deepcopy(out[self.optimal_adr_])

        return out

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Align ondil with the scikit-learn API for partial fitting.

        The first partial fit will call `fit`, and subsequent calls will call `update`.
        Allows furthermore to use the sklearn testing framework.

        Overwrites the base class method to avoid sample_weights. This estimator does not support sample weights.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target values.
        Returns
        -------
        self : Estimator
            The fitted estimator.

        """
        if self.is_fitted:
            self.update(X=X, y=y)
        else:
            self.fit(X=X, y=y)
        return self

    # Different UV - MV
    @_fit_context(prefer_skip_nested_validation=True)
    def update(
        self, X: np.ndarray, y: np.ndarray
    ) -> "MultivariateOnlineDistributionalRegressionPath":
        """
        Updates the estimator with new observations.

        This method validates the input data, updates internal counters for the number of observations,
        recalculates the effective training length based on the learning rate, and updates the model's
        likelihood. It also scales the input features, stores previous states for model selection and
        likelihood, and performs an outer update step with the scaled data.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples, n_outputs).
        Returns
        -------
        self : object
            Returns the updated estimator instance.
        """

        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=False,
            dtype=[np.float64, np.float32],
            multi_output=True,
        )
        ensure_atleast_bivariate(y=y)
        validate_response_support(self, y=y)

        self.n_observations_ += y.shape[0]
        self.n_observations_step_ = y.shape[0]
        self.n_training_ = calculate_effective_training_length(
            forget=self.learning_rate, n_obs=self.n_observations_
        )
        theta = self.predict_all_adr(X)
        self._scaler.update(X=X)
        X_scaled = self._scaler.transform(X=X)

        self._x_gram_old = copy.deepcopy(self._x_gram)
        self._y_gram_old = copy.deepcopy(self._y_gram)
        self._model_selection_old = copy.deepcopy(self._model_selection)
        self._old_likelihood = self._current_likelihood + 0
        self._old_likelihood_discounted = (
            1 - self.learning_rate
        ) ** self.n_observations_step_ * self._old_likelihood
        self._current_likelihood = self._old_likelihood_discounted + np.array([
            np.sum(
                self.distribution.logpdf(y=y, theta=theta[a])
                * init_forget_vector(self.learning_rate, y.shape[0])
            )
            for a in range(self.adr_steps_)
        ])
        self._outer_update(X=X_scaled, y=y, theta=theta)

        return self

    # Different UV - MV
    def _inner_update(self, X, y, theta, outer_iteration, a, p):
        converged = False
        decreasing = False
        old_likelihood = self._current_likelihood[a]
        weights_forget = init_forget_vector(
            self.learning_rate, self.n_observations_step_
        )

        for inner_iteration in range(self.max_iterations_inner):
            # If the likelihood is at some point decreasing, we're breaking
            # Hence we need to store previous iteration values:
            if (inner_iteration > 0) | (outer_iteration > 0):
                prev_theta = copy.copy(theta)
                prev_x_gram = copy.copy(self._x_gram[p])
                prev_y_gram = copy.copy(self._y_gram[p])
                prev_model_selection = copy.copy(self._model_selection)
                prev_beta = copy.copy(self.coef_)
                prev_beta_path = copy.copy(self.coef_path_)

            for k in self._iter_index[p]:
                # Handle AD-R Regularization
                if self._is_element_adr_regularized(p=p, k=k, a=a):
                    self.coef_[p][k][a] = np.zeros(self.n_features_[p][k])
                    self.coef_path_[p][k][a] = np.zeros((
                        self._lambda_n[p],
                        self.n_features_[p][k],
                    ))

                else:
                    eta = self.distribution.link_function(theta[a][p], p)
                    eta = self.distribution.cube_to_flat(eta, param=p)

                    # Derivatives wrt to the parameter
                    dl1dp1 = self.distribution.element_dl1_dp1(
                        y, theta=theta[a], param=p, k=k
                    )
                    dl2dp2 = self.distribution.element_dl2_dp2(
                        y, theta=theta[a], param=p, k=k
                    )

                    dl1_link = self.distribution.link_function_derivative(
                        theta[a][p], p
                    )
                    dl2_link = self.distribution.link_function_second_derivative(
                        theta[a][p], p
                    )
                    dl1_link = self.distribution.cube_to_flat(dl1_link, param=p)
                    dl1_link = dl1_link[:, k]
                    dl2_link = self.distribution.cube_to_flat(dl2_link, param=p)
                    dl2_link = dl2_link[:, k]

                    dl1_deta1 = dl1dp1 * (1 / dl1_link)
                    dl2_deta2 = (dl2dp2 * dl1_link - dl1dp1 * dl2_link) / dl1_link**3

                    wt = np.fmax(-dl2_deta2, 1e-10)
                    wv = eta[:, k] + dl1_deta1 / wt

                    # Make model arrays
                    x = make_model_array(
                        X=X,
                        eq=self._equation[p][k],
                        fit_intercept=self._fit_intercept[p],
                    )
                    self._x_gram[p][k][a] = self._method[p][k].update_x_gram(
                        gram=self._x_gram_old[p][k][a],
                        X=x,
                        weights=wt ** self._weight_delta[p],
                        forget=self._forget[p],
                    )
                    self._y_gram[p][k][a] = (
                        self
                        ._method[p][k]
                        .update_y_gram(
                            gram=np.expand_dims(self._y_gram_old[p][k][a], -1),
                            X=x,
                            y=wv,
                            weights=wt ** self._weight_delta[p],
                            forget=self._forget[p],
                        )
                        .squeeze()
                    )
                    if self._method[p][k]._path_based_method:
                        self.coef_path_[p][k][a] = self._method[p][k].update_beta_path(
                            x_gram=self._x_gram[p][k][a],
                            y_gram=self._y_gram[p][k][a][:, None],
                            beta_path=self.coef_path_[p][k][a],
                            is_regularized=self.is_regularized_[p][k],
                        )
                        eta_elem = x @ self.coef_path_[p][k][a].T
                        theta_elem = self.distribution.element_link_inverse(
                            eta_elem, param=p, k=k, d=self.dim_
                        )

                        opt_ic = self._update_model_selection(
                            y=y,
                            theta=theta,
                            theta_fit=theta_elem,
                            outer_iteration=outer_iteration,
                            inner_iteration=inner_iteration,
                            param=p,
                            k=k,
                            a=a,
                        )
                        # Select the optimal beta
                        self.coef_[p][k][a] = self.coef_path_[p][k][a][opt_ic, :]
                        theta[a] = self.distribution.set_theta_element(
                            theta[a], theta_elem[:, opt_ic], param=p, k=k
                        )
                    else:
                        self.coef_path_[p][k][a] = None
                        self.coef_[p][k][a] = self._method[p][k].update_beta(
                            x_gram=self._x_gram[p][k][a],
                            y_gram=self._y_gram[p][k][a][:, None],
                            beta=self.coef_[p][k][a],
                            is_regularized=self.is_regularized_[p][k],
                        )
                        self.coef_[p][k][a] = (
                            self._x_gram[p][k][a] @ self._y_gram[p][k][a]
                        )

                    # Calculate the other stuff
                    eta[:, k] = np.squeeze(x @ self.coef_[p][k][a])
                    theta[a][p] = self.distribution.link_inverse(
                        self.distribution.flat_to_cube(eta, param=p), param=p
                    )

                self._current_likelihood[a] = (
                    np.sum(self.distribution.logpdf(y, theta=theta[a]) * weights_forget)
                    + self._old_likelihood_discounted[a]
                )

            self.iteration_count_[outer_iteration, p] = inner_iteration
            self.iteration_likelihood_[outer_iteration, inner_iteration, p, a] = (
                self._current_likelihood[a]
            )

            # Are we in the last iteration
            if inner_iteration == (self.max_iterations_inner - 1):
                warnings.warn(
                    "Reached max inner iterations. Algorithm may or may not be converged."
                )

            # Are we converged
            if inner_iteration > 0:
                converged, decreasing = self._check_inner_convergence(
                    old_value=old_likelihood, new_value=self._current_likelihood[a]
                )

                if converged:
                    break
                else:
                    # For the next iteration
                    old_likelihood = self._current_likelihood[a]

            # Are we diverging?
            if ((outer_iteration > 0) | (inner_iteration > 1)) & decreasing:
                warnings.warn("Likelihood is decreasing. Breaking.")
                # Reset to values from the previous iteration
                theta = prev_theta
                self._model_selection = prev_model_selection
                self._x_gram[p] = prev_x_gram
                self._y_gram[p] = prev_y_gram
                self.coef_ = prev_beta
                self.coef_path_ = prev_beta_path
                self._current_likelihood[a] = old_likelihood
                break

        return theta

    # Different UV - MV
    def _outer_update(self, X, y, theta):
        adr_start = time.time()
        for a in range(min(self.adr_steps_, self.last_fit_adr_max_ + 1)):
            adr_it_start = time.time()
            outer_start = time.time()

            global_old_likelihood = 0
            converged = False

            for outer_iteration in range(self.max_iterations_outer):
                outer_it_start = time.time()
                # Main Loop
                for p in range(self.distribution.n_params):
                    message = (
                        f"Outer Iteration: {outer_iteration}, "
                        f"fitting Distribution parameter {p}, AD-r step {a}"
                    )
                    self._print_message(level=2, message=message)
                    theta = self._inner_update(
                        X=X, y=y, theta=theta, outer_iteration=outer_iteration, p=p, a=a
                    )

                # Get global LL
                global_likelihood = self._current_likelihood[a]
                converged, _ = self._check_outer_convergence(
                    global_old_likelihood, global_likelihood
                )

                # start value for the next AD-R step
                if converged | (outer_iteration == self.max_iterations_outer - 1):
                    if a < (self.adr_steps_ - 1):
                        for p in range(self.distribution.n_params):
                            if not self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = copy.deepcopy(theta[a][p])
                            if self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = self._get_next_adr_start_values(
                                    theta, p, a + 1
                                )

                # Timings
                outer_it_end = time.time()
                outer_it_time = outer_it_end - outer_it_start
                outer_it_avg = (outer_it_end - outer_start) / (outer_iteration + 1)
                outer_pred_time = outer_it_time * (
                    self.max_iterations_outer - outer_iteration - 1
                )
                message = (
                    f"Last outer iteration {outer_iteration} took {round(outer_it_time, 1)} sec. "
                    f"Average outer iteration took {round(outer_it_avg, 1)} sec. "
                    f"Expected to be finished in max {round(outer_pred_time, 1)} sec. "
                )
                self._print_message(level=2, message=message)

                # End timing for the iteration
                adr_it_end = time.time()

                if converged:
                    break
                else:
                    global_old_likelihood = global_likelihood

            # Next AD-R iteration
            adr_it_last = adr_it_end - adr_it_start
            adr_it_avg = (adr_it_end - adr_start) / (a + 1)

            message = (
                f"Last ADR iteration {a} took {round(adr_it_last, 1)} sec. "
                f"Average ADR iteration took {round(adr_it_avg, 1)} sec. ",
            )
            self._print_message(level=1, message=message)

            # Calculate the improvement
            if self.early_stopping_criteria == "ll":
                # Check the likelihood for early stopping
                self.improvement_abs_ = -np.diff(self._current_likelihood)
                self.improvement_abs_scaled_ = -self.improvement_abs_ / self.n_training_
                self.improvement_rel_ = (
                    self.improvement_abs_ / self._current_likelihood[-1:]
                )

            elif self.early_stopping_criteria in ["aic", "bic", "hqc", "max"]:
                self._early_stopping_n_params = np.array([
                    self.count_nonzero_coef(self.coef_, a)
                    for a in range(self.adr_steps_)
                ])
                self.early_stopping_ic_ = InformationCriterion(
                    n_observations=self.n_training_,
                    n_parameters=self._early_stopping_n_params,
                    criterion=self.early_stopping_criteria,
                ).from_ll(self._current_likelihood)

                self.improvement_abs_ = -np.diff(self.early_stopping_ic_)
                self.improvement_abs_scaled_ = self.improvement_abs_
                self.improvement_rel_ = (
                    self.improvement_abs_ / self.early_stopping_ic_[-1:]
                )
            else:
                raise ValueError(
                    "Did not recognize criteria AD-R regularization stopping criteria."
                )

            if (
                self.early_stopping
                and (a > 0)
                and (
                    a < min(self.adr_steps_ - 1, self.last_fit_adr_max_)
                )  # In the last step, it does not make sense to "early stop"
            ):
                if (
                    self.improvement_abs_scaled_[a - 1] < self.early_stopping_abs_tol
                ) or (self.improvement_rel_[a - 1] < self.early_stopping_rel_tol):
                    message = (
                        f"Early stopping due to AD-r-regression. "
                        f"Last increase in r lead to relative improvement: {self.improvement_rel_[a - 1]}, scaled absolute improvement {self.improvement_abs_scaled_[a - 1]}",
                    )

                    self._print_message(level=1, message=message)
                    if self.improvement_rel_[a - 1] > 0:
                        self.optimal_adr_ = a
                    elif self.improvement_rel_[a - 1] < 0:
                        self.optimal_adr_ = a - 1

                    self.improvement_abs_[(a + 1) :] = 0
                    self.improvement_rel_[(a + 1) :] = 0
                    self.improvement_abs_scaled_[(a + 1) :] = 0
                    break
            else:
                # The largest theta is the optimal one
                # But might be overfit
                self.optimal_adr_ = a

        self.theta_ = theta
        self.optimal_theta_ = self.theta_[self.optimal_adr_]
        self.last_fit_adr_max_ = self.optimal_adr_
