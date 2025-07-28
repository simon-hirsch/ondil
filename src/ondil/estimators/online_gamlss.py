import copy
import numbers
import warnings
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from .. import HAS_PANDAS, HAS_POLARS
from ..base import Distribution, EstimationMethod, OndilEstimatorMixin
from ..distributions import Normal
from ..error import OutOfSupportError
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..scaler import OnlineScaler
from ..utils import calculate_effective_training_length, online_mean_update

if HAS_PANDAS:
    import pandas as pd
if HAS_POLARS:
    import polars as pl  # noqa


class OnlineDistributionalRegression(
    OndilEstimatorMixin, RegressorMixin, BaseEstimator
):
    """The online/incremental GAMLSS class."""

    _parameter_constraints = {
        "forget": [Interval(numbers.Real, 0.0, 1.0, closed="left")],
        "fit_intercept": [bool],
        "scale_inputs": [bool, np.ndarray],
        "regularize_intercept": [bool],
        "method": [EstimationMethod, str],
        "ic": [StrOptions({"aic", "bic", "hqc", "max"})],
        "distribution": [callable],
        "equation": [dict, type(None)],
        "model_selection": [StrOptions({"local_rss", "global_ll"})],
        "prefit_initial": [Interval(numbers.Integral, 0, None, closed="left")],
        "prefit_update": [Interval(numbers.Integral, 0, None, closed="left")],
        "cautious_updates": [bool],
        "max_it_outer": [Interval(numbers.Integral, 1, None, closed="left")],
        "max_it_inner": [Interval(numbers.Integral, 1, None, closed="left")],
        "abs_tol_outer": [Interval(numbers.Real, 0.0, None, closed="left")],
        "abs_tol_inner": [Interval(numbers.Real, 0.0, None, closed="left")],
        "rel_tol_outer": [Interval(numbers.Real, 0.0, None, closed="left")],
        "rel_tol_inner": [Interval(numbers.Real, 0.0, None, closed="left")],
        "step_size": [numbers.Real, dict],
        "verbose": [Interval(numbers.Integral, 0, None, closed="left")],
        "debug": [bool],
        "param_order": [np.ndarray, type(None)],
    }

    def __init__(
        self,
        distribution: Distribution = Normal(),
        equation: Dict[int, Union[str, np.ndarray, list]] = None,
        forget: float | Dict[int, float] = 0.0,
        method: Union[
            str, EstimationMethod, Dict[int, str], Dict[int, EstimationMethod]
        ] = "ols",
        scale_inputs: bool | np.ndarray = True,
        fit_intercept: Union[bool, Dict[int, bool]] = True,
        regularize_intercept: Union[bool, Dict[int, bool]] = False,
        ic: Union[str, Dict] = "aic",
        model_selection: Literal["local_rss", "global_ll"] = "local_rss",
        prefit_initial: int = 0,
        prefit_update: int = 0,
        step_size: float | Dict[int, float] = 1.0,
        verbose: int = 0,
        debug: bool = False,
        param_order: np.ndarray | None = None,
        cautious_updates: bool = False,
        cond_start_val: bool = False,
        max_it_outer: int = 30,
        max_it_inner: int = 30,
        abs_tol_outer: float = 1e-3,
        abs_tol_inner: float = 1e-3,
        rel_tol_outer: float = 1e-5,
        rel_tol_inner: float = 1e-5,
        min_it_outer: int = 1,
    ) -> "OnlineDistributionalRegression":
        """The `OnlineDistributionalRegression()` provides the fit, update and predict methods for linear parametric GAMLSS models.

        For a response variable $Y$ which is distributed according to the distribution $\mathcal{F}(\\theta)$
        with the distribution parameters $\\theta$, we model:

        $$g_k(\\theta_k) = \\eta_k = X_k\\beta_k$$

        where $g_k(\cdot)$ is a link function, which ensures that the predicted distribution parameters are in a
        sensible range (we don't want, e.g. negative standard deviations), and $\eta_k$ is the predictor (on the
        space of the link function). The model is fitted using iterative re-weighted least squares (IRLS).


        !!! note "Tips and Tricks"
            If you're facing issues with non-convergence and/or matrix inversion problems, please enable the `debug` mode and increase the
            logging level by increasing `verbose`.
            In debug mode, the estimator will save the weights, working vectors, derivatives each iteration in a
            according dictionary, i.e. self._debug_weights.
            The keys are composed of a tuple of ints of `(parameter, outer_iteration, inner_iteration)`.
            Very small and/or very large weights (implicitly second derivatives) can be a sign that either start values are not chosen appropriately or
            that the distributional assumption does not fit the data well.

        !!! warning "Debug Mode"
            Please don't use debug more for production models since it saves the `X` matrix and its scaled counterpart, so you will get large
            estimator objects.

        !!! warning "Conditional start values `cond_start_val=False`"
            The `cond_start_val` parameter is considered experimental and may not work as expected.

        !!! warning "Cautious updates `cautious_updates=True`"
            The `cautious_updates` parameter is considered experimental and may not work as expected.

        Args:
            distribution (ondil.Distribution): The parametric distribution to use for modeling the response variable.
            equation (Dict[int, Union[str, np.ndarray, list]], optional): The modeling equation for each distribution parameter. The dictionary should map parameter indices to either the strings `'all'`, `'intercept'`, a numpy array of column indices, or a list of column names. Defaults to None, which uses all covariates for the first parameter and intercepts for others.
            forget (float | Dict[int, float], optional): The forget factor for exponential weighting of past observations. Can be a single float for all parameters or a dictionary mapping parameter indices to floats. Defaults to 0.0.
            method (str | EstimationMethod | Dict[int, str] | Dict[int, EstimationMethod], optional): The estimation method for each parameter. Can be a string, EstimationMethod, or a dictionary mapping parameter indices. Defaults to "ols".
            scale_inputs (bool | np.ndarray, optional): Whether to scale the input features. Can be a boolean or a numpy array specifying scaling per feature. Defaults to True.
            fit_intercept (bool | Dict[int, bool], optional): Whether to fit an intercept for each parameter. Can be a boolean or a dictionary mapping parameter indices. Defaults to True.
            regularize_intercept (bool | Dict[int, bool], optional): Whether to regularize the intercept for each parameter. Can be a boolean or a dictionary mapping parameter indices. Defaults to False.
            ic (str | Dict, optional): Information criterion for model selection (e.g., "aic", "bic"). Can be a string or a dictionary mapping parameter indices. Defaults to "aic".
            model_selection (Literal["local_rss", "global_ll"], optional): Model selection strategy. "local_rss" selects based on local residual sum of squares, "global_ll" uses global log-likelihood. Defaults to "local_rss".
            prefit_initial (int, optional): Number of initial outer iterations with only one inner iteration (for stabilization). Defaults to 0.
            prefit_update (int, optional): Number of initial outer iterations with only one inner iteration during updates. Defaults to 0.
            step_size (float | Dict[int, float], optional): Step size for parameter updates. Can be a float or a dictionary mapping parameter indices. Defaults to 1.0.
            verbose (int, optional): Verbosity level for logging. 0 = silent, 1 = high-level, 2 = per-parameter, 3 = per-iteration. Defaults to 0.
            debug (bool, optional): Enable debug mode. Debug mode will save additional data to the estimator object.
                Currently, we save

                    * self._debug_X_dict
                    * self._debug_X_scaled
                    * self._debug_weights
                    * self._debug_working_vectors
                    * self._debug_dl1dlp1
                    * self._debug_dl2dlp2
                    * self._debug_eta
                    * self._debug_fv
                    * self._debug_coef
                    * self._debug_coef_path

                to the the estimator. Debug mode works in batch and online settings. Note that debug mode is not recommended for production use. Defaults to False.
            param_order (np.ndarray | None, optional): Order in which to fit the distribution parameters. Defaults to None (natural order).
            cautious_updates (bool, optional): If True, use smaller step sizes and more iterations when new data are outliers. Defaults to False.
            cond_start_val (bool, optional): If True, use conditional start values for parameters (experimental). Defaults to False.
            max_it_outer (int, optional): Maximum number of outer iterations for the fitting algorithm. Defaults to 30.
            max_it_inner (int, optional): Maximum number of inner iterations for the fitting algorithm. Defaults to 30.
            abs_tol_outer (float, optional): Absolute tolerance for convergence in the outer loop. Defaults to 1e-3.
            abs_tol_inner (float, optional): Absolute tolerance for convergence in the inner loop. Defaults to 1e-3.
            rel_tol_outer (float, optional): Relative tolerance for convergence in the outer loop. Defaults to 1e-5.
            rel_tol_inner (float, optional): Relative tolerance for convergence in the inner loop. Defaults to 1e-5.
            min_it_outer (int, optional): Minimum number of outer iterations before checking for convergence. Defaults to 1.

        Attributes:
            distribution (Distribution): The distribution used for modeling.
            equation (Dict[int, Union[str, np.ndarray, list]]): The modeling equation for each distribution parameter.
            forget (Dict[int, float]): Forget factor for each distribution parameter.
            fit_intercept (Dict[int, bool]): Whether to fit an intercept for each parameter.
            regularize_intercept (Dict[int, bool]): Whether to regularize the intercept for each parameter.
            ic (Dict[int, str]): Information criterion for model selection for each parameter.
            method (Dict[int, EstimationMethod]): Estimation method for each parameter.
            scale_inputs (bool | np.ndarray): Whether to scale the input features.
            param_order (np.ndarray | None): Order in which to fit the distribution parameters.
            n_observations_ (float): Total number of observations used for fitting.
            n_training_ (Dict[int, int]): Effective training length for each distribution parameter.
            n_features_ (Dict[int, int]): Number of features used for each distribution parameter.
            coef_ (np.ndarray): Coefficients for the fitted model, shape (n_params, n_features).
            coef_path_ (np.ndarray): Coefficients path for the fitted model, shape (n_params, n_iterations, n_features). Only available if `method` is a path-based method like LASSO.

        Returns:
            OnlineDistributionalRegression: The OnlineDistributionalRegression instance.

        """
        self.distribution = distribution
        self.equation = equation
        self.forget = forget
        self.fit_intercept = fit_intercept
        self.regularize_intercept = regularize_intercept
        self.ic = ic
        self.method = method
        self.param_order = param_order
        self.model_selection = model_selection

        # Scaling
        self.scale_inputs = scale_inputs

        # These are global for all distribution parameters
        self.max_it_outer = max_it_outer
        self.max_it_inner = max_it_inner
        self.min_it_outer = min_it_outer
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner

        self.step_size = step_size
        self.cautious_updates = cautious_updates
        self.cond_start_val = cond_start_val
        self.debug = debug
        self.verbose = verbose
        self.prefit_initial = prefit_initial
        self.prefit_update = prefit_update

    @property
    def betas(self):
        warnings.warn(
            "OnlineDistributionalRegression.betas is depreciated in favour of OnlineDistributionalRegression.beta to be consistent with beta_path. "
            "Alternatively, use OnlineDistributionalRegression.coef_ as in sklearn.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.coef_

    @property
    def beta(self):
        check_is_fitted(self)
        return self.coef_

    @property
    def beta_path(self):
        check_is_fitted(self)
        return self.coef_path_

    def _prepare_estimator(self):
        self._equation = self._process_equation(self.equation)
        self._method = {
            p: get_estimation_method(m)
            for p, m in self._process_parameter(
                self.method, default="ols", name="method"
            ).items()
        }
        # Parameters that can be set as dicts or single values need to be processed
        self._forget = self._process_parameter(self.forget, default=0.0, name="forget")
        self._ic = self._process_parameter(self.ic, "aic", name="ic")
        self._fit_intercept = self._process_parameter(
            self.fit_intercept, True, "fit_intercept"
        )
        self._regularize_intercept = self._process_parameter(
            self.regularize_intercept, False, "regularize_intercept"
        )
        self._step_size = self._process_parameter(
            self.step_size, default=1.0, name="step_size"
        )

        if self.param_order is None:
            self._param_order = np.arange(self.distribution.n_params)
        else:
            if all(i in self.param_order for i in range(self.distribution.n_params)):
                self._param_order = self.param_order
            else:
                raise ValueError("All parameters should be in the param_order.")

        self._scaler = OnlineScaler(forget=self._forget[0], to_scale=self.scale_inputs)

        if self.cond_start_val:
            warnings.warn(
                "cond_start_val is considered an experimental feature.",
                UserWarning,
                stacklevel=2,
            )

    def _print_message(self, message, level=0):
        if level <= self.verbose:
            print(f"[{self.__class__.__name__}]", message)

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

    def _process_equation(self, equation: Dict):
        """Preprocess the equation object and validate inputs."""
        if equation is None:
            warnings.warn(
                f"[{self.__class__.__name__}] "
                "Equation is not specified. "
                "Per default, will estimate the first distribution parameter by all covariates found in X. "
                "All other distribution parameters will be estimated by an intercept."
            )
            equation = {
                p: "all" if p == 0 else "intercept"
                for p in range(self.distribution.n_params)
            }
        else:
            for p in range(self.distribution.n_params):
                # Check that all distribution parameters are in the equation.
                # If not, add intercept.
                if p not in equation.keys():
                    warnings.warn(
                        f"[{self.__class__.__name__}] "
                        f"Distribution parameter {p} is not in equation. "
                        f"The parameter will be estimated by an intercept."
                    )
                    equation[p] = "intercept"

                if not (
                    isinstance(equation[p], np.ndarray)
                    or (equation[p] in ["all", "intercept"])
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

    @staticmethod
    def _make_intercept(n_observations: int) -> np.ndarray:
        """Make the intercept series as N x 1 array.

        Args:
            y (np.ndarray): Response variable $Y$

        Returns:
            np.ndarray: Intercept array.
        """
        return np.ones((n_observations, 1))

    def _is_intercept_only(self, param: int):
        """Check in the equation whether we model only as intercept"""
        if isinstance(self._equation[param], str):
            return self._equation[param] == "intercept"
        else:
            return False

    def _count_nonzero_coef(self, exclude: int | np.ndarray | None = None) -> int:
        if exclude is None:
            gen = range(self.distribution.n_params)
        else:
            gen = np.delete(np.arange(self.distribution.n_params), exclude)
        return sum(np.count_nonzero(self.coef_[p]) for p in gen)

    def get_J_from_equation(self, X: np.ndarray):
        J = {}
        for p in range(self.distribution.n_params):
            if isinstance(self._equation[p], str):
                if self._equation[p] == "all":
                    J[p] = X.shape[1] + int(self._fit_intercept[p])
                if self._equation[p] == "intercept":
                    J[p] = 1
            elif isinstance(self._equation[p], np.ndarray):
                if np.issubdtype(self._equation[p].dtype, bool):
                    if self._equation[p].shape[0] != X.shape[1]:
                        raise ValueError(f"Shape does not match for param {p}.")
                    J[p] = np.sum(self._equation[p]) + int(self._fit_intercept[p])
                elif np.issubdtype(self._equation[p].dtype, np.integer):
                    if self._equation[p].max() >= X.shape[1]:
                        raise ValueError(f"Shape does not match for param {p}.")
                    J[p] = self._equation[p].shape[0] + int(self._fit_intercept[p])
                else:
                    raise ValueError(
                        "If you pass a np.ndarray in the equation, "
                        "please make sure it is of dtype bool or int."
                    )
            elif isinstance(self._equation[p], list):
                J[p] = len(self._equation[p]) + int(self._fit_intercept[p])
            else:
                raise ValueError("Something unexpected happened")
        return J

    def make_model_array(self, X: Union[np.ndarray], param: int):
        eq = self._equation[param]
        n = X.shape[0]

        # TODO: Check difference between np.array and list more explicitly?
        if isinstance(eq, str) and (eq == "intercept"):
            if not self._fit_intercept[param]:
                raise ValueError(
                    "fit_intercept[param] is false, but equation says intercept."
                )
            out = self._make_intercept(n_observations=n)
        else:
            if isinstance(eq, str) and (eq == "all"):
                if isinstance(X, np.ndarray):
                    out = X
                if HAS_PANDAS and isinstance(X, pd.DataFrame):
                    out = X.to_numpy()
                if HAS_POLARS and isinstance(X, pl.DataFrame):
                    out = X.to_numpy()
            elif isinstance(eq, np.ndarray) | isinstance(eq, list):
                if isinstance(X, np.ndarray):
                    out = X[:, eq]
                if HAS_PANDAS and isinstance(X, pd.DataFrame):
                    out = X.loc[:, eq]
                if HAS_POLARS and isinstance(X, pl.DataFrame):
                    out = X.select(eq).to_numpy()
            else:
                raise ValueError("Did not understand equation. Please check.")

            if self._fit_intercept[param]:
                out = np.hstack((self._make_intercept(n), out))

        return out

    def _fit_select_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wv: np.ndarray,
        wt: np.ndarray,
        w: np.ndarray,
        beta_path: np.ndarray,
        param: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Fit and select a model for a specific distribution parameter.

        Selects the best model along a regularization path by evaluating an information criterion
        (e.g., AIC, BIC) based on either local residual sum of squares (RSS) or global log-likelihood (LL).

        Parameters
        ----------
        X : np.ndarray
            The design matrix of shape (n_samples, n_features).
        y : np.ndarray
            The target variable of shape (n_samples,).
        wv : np.ndarray
            Working response vector of shape (n_samples,).
        wt : np.ndarray
            Working weights vector of shape (n_samples,).
        w : np.ndarray
            Sample weights of shape (n_samples,).
        beta_path : np.ndarray
            Array of shape (n_alphas, n_features) containing the coefficient paths for each regularization parameter.
        param : int
            The parameter index for which the model is being selected.

        Returns
        -------
        beta : np.ndarray
            The selected coefficient vector of shape (n_features,) corresponding to the best model.
        model_selection_data : np.ndarray
            The array of RSS or log-likelihood values used for model selection.
        best_ic : int
            The index of the best model according to the information criterion.

        Notes
        -----
        The selection is performed according to the `model_selection` attribute:
        - If "local_rss", the model is selected based on the weighted residual sum of squares.
        - If "global_ll", the model is selected based on the global log-likelihood.
        The function uses an information criterion (e.g., AIC, BIC) to select the best model.
        """
        f = init_forget_vector(self._forget[param], y.shape[0])
        n_nonzero_coef = np.count_nonzero(beta_path, axis=1)
        n_nonzero_coef_other = self._count_nonzero_coef(exclude=param)

        prediction_path = X @ beta_path.T

        if self.model_selection == "local_rss":
            residuals = wv[:, None] - prediction_path
            rss = np.sum(residuals**2 * w[:, None] * wt[:, None] * f[:, None], axis=0)
            rss = rss / np.mean(wt * w * f)
            ic = InformationCriterion(
                n_observations=self.n_training_[param],
                n_parameters=n_nonzero_coef + n_nonzero_coef_other,
                criterion=self._ic[param],
            ).from_rss(rss=rss)
            best_ic = np.argmin(ic)
            beta = beta_path[best_ic, :]
            model_selection_data = rss

        elif self.model_selection == "global_ll":
            ll = np.zeros(self._method[param]._path_length)
            theta = np.copy(self._fv)
            for i in range(self._method[param]._path_length):
                theta[:, param] = self.distribution.link_inverse(
                    prediction_path[:, i], param=param
                )
                ll[i] = np.sum(f * self.distribution.logpdf(y, theta))

            ic = InformationCriterion(
                n_observations=self.n_training_[param],
                n_parameters=n_nonzero_coef + n_nonzero_coef_other,
                criterion=self._ic[param],
            ).from_ll(log_likelihood=ll)
            best_ic = np.argmin(ic)
            beta = beta_path[best_ic, :]
            model_selection_data = ll

        return beta, model_selection_data, best_ic

    def _update_select_model(
        self,
        X: np.ndarray,
        y: np.ndarray,  # observations / response
        wv: np.ndarray,  # working vector
        wt: np.ndarray,  # working weights
        w: np.ndarray,  # sample weights
        beta_path: np.ndarray,
        model_selection_data: Any,
        param: int,
    ):
        """Update and select a model for a specific distribution parameter.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Observed response values of shape (n_samples,).
        wv : np.ndarray
            Working vector, typically the adjusted response for IRLS or similar algorithms.
        wt : np.ndarray
            Working weights, used for weighted updates.
        w : np.ndarray
            Sample weights for each observation.
        beta_path : np.ndarray
            Array of candidate coefficient vectors along the regularization path, shape (n_candidates, n_features).
        model_selection_data : Any
            Data carried over from previous model selection steps (e.g., previous RSS or log-likelihood values).
        param : int
            Index of the distribution parameter being updated.
        Returns
        -------
        beta : np.ndarray
            Selected coefficient vector for the best model.
        model_selection_data_new : Any
            Updated model selection data (e.g., RSS or log-likelihood values for the selected model).
        best_ic : int
            Index of the best model according to the information criterion.

        """
        f = init_forget_vector(self._forget[param], y.shape[0])
        n_nonzero_coef = np.count_nonzero(beta_path, axis=1)
        prediction_path = X @ beta_path.T
        if self.model_selection == "local_rss":
            # Denominator
            # TODO: This should go in the online mean update function.
            denom = online_mean_update(
                self._mean_of_weights[param],
                wt,
                self._forget[param],
                self.n_observations_,
            )
            residuals = wv[:, None] - prediction_path
            rss = (
                np.sum((residuals**2) * w[:, None] * wt[:, None] * f[:, None], axis=0)
                + (1 - self._forget[param]) ** y.shape[0]
                * (model_selection_data * self._mean_of_weights[param])
            ) / denom
            n_nonzero_coef_other = self._count_nonzero_coef(exclude=param)
            ic = InformationCriterion(
                n_observations=self.n_training_[param],
                n_parameters=n_nonzero_coef + n_nonzero_coef_other,
                criterion=self._ic[param],
            ).from_rss(rss=rss)
            best_ic = np.argmin(ic)
            model_selection_data_new = rss
        elif self.model_selection == "global_ll":
            ll = np.zeros(self._method[param]._path_length)

            theta = np.copy(self._fv)
            for i in range(self._method[param]._path_length):
                theta[:, param] = self.distribution.link_inverse(
                    prediction_path[:, i], param=param
                )
                ll[i] = np.sum(w * f * self.distribution.logpdf(y, theta))
            ll = ll + (1 - self._forget[param]) ** y.shape[0] * model_selection_data

            ic = InformationCriterion(
                n_observations=self.n_training_[param],
                n_parameters=n_nonzero_coef,
                criterion=self._ic[param],
            ).from_ll(log_likelihood=ll)
            best_ic = np.argmin(ic)
            model_selection_data_new = ll

        beta = beta_path[best_ic, :]
        return beta, model_selection_data_new, best_ic

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate the input matrices X and y.

        Args:
            X (np.ndarray): Input matrix $X$
            y (np.ndarray): Response vector $y$.

        Raises:
            OutOfSupportError: If the values of $y$ are below the range of the distribution.
            OutOfSupportError: If the values of $y$ are beyond the range of the distribution.
        """
        if np.any(y < self.distribution.distribution_support[0]):
            raise OutOfSupportError(
                message=(
                    "y contains values below the distribution's support. "
                    f"The smallest value in y is {np.min(y)}. "
                    f"The support of the distribution is {str(self.distribution.distribution_support)}."
                )
            )
        if np.any(y > self.distribution.distribution_support[1]):
            raise OutOfSupportError(
                message=(
                    "Y contains values larger than the distribution's support. "
                    f"The smallest value in y is {np.max(y)}. "
                    f"The support of the distribution is {str(self.distribution.distribution_support)}."
                ),
            )

    def _make_iter_schedule(self):
        return np.repeat(self.max_it_inner, repeats=self.max_it_outer)

    def _make_step_size_schedule(self):
        return np.tile(
            A=list(self._step_size.values()),
            reps=(self.max_it_outer, self.max_it_inner, 1),
        ).astype(float)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "OnlineDistributionalRegression":
        """Fit the online GAMLSS model.

        This method initializes the model with the given covariate data matrix $X$ and response variable $Y$.

        Args:
            X (np.ndarray): Covariate data matrix $X$.
            y (np.ndarray): Response variable $Y$.
            sample_weight (Optional[np.ndarray], optional): User-defined sample weights. Defaults to None.

        Returns:
            OnlineDistributionalRegression: The fitted OnlineDistributionalRegression instance.

        Raises:
            ValueError: If the equation is not specified correctly.
            OutOfSupportError: If the values of $y$ are below or above the distribution's support.
        """

        self._prepare_estimator()

        # Validate inputs
        X, y = validate_data(self, X=X, y=y, reset=True, dtype=[np.float64, np.float32])
        _ = type_of_target(y, raise_unknown=True)
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)
        self._validate_inputs(X, y)

        self.n_observations_ = sample_weight.sum()
        self.n_training_ = {
            p: calculate_effective_training_length(
                self._forget[p], self.n_observations_
            )
            for p in range(self.distribution.n_params)
        }

        self._fv = self.distribution.initial_values(y=y)
        self.n_features_ = self.get_J_from_equation(X=X)

        # Fit scaler and transform
        self._scaler.fit(X, sample_weight=sample_weight)
        X_scaled = self._scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }

        for p, x in X_dict.items():
            cond_num = np.linalg.cond(x)
            if cond_num > 100:
                message = (
                    f"Condition number of X for after scaling (and adding an intercept) for param {p} is {cond_num}. "
                    "This might lead to numerical issues. Consider using a regularized estimation method."
                )
                self._print_message(message=message, level=1)

        if self.debug:
            self._debug_X_dict = X_dict
            self._debug_X_scaled = X_scaled
            self._debug_weights = {}
            self._debug_working_vectors = {}
            self._debug_dl1dlp1 = {}
            self._debug_dl2dlp2 = {}
            self._debug_eta = {}
            self._debug_fv = {}
            self._debug_dv = {}
            self._debug_coef = {}
            self._debug_coef_path = {}

        self._x_gram = {}
        self._y_gram = {}
        self._rss = np.zeros((self.distribution.n_params))
        self._rss_iterations = np.zeros(
            (self.distribution.n_params, self.max_it_outer, self.max_it_inner)
        )

        # For regularized estimation, we have the necessary data to do model selection!
        self._model_selection_data = {}
        self._best_ic = np.zeros(self.distribution.n_params)
        self._best_ic_iterations = np.full(
            (self.distribution.n_params, self.max_it_outer, self.max_it_inner),
            np.nan,
        )

        self._is_regularized = {}
        for p in range(self.distribution.n_params):
            is_regularized = np.repeat(True, self.n_features_[p])
            if self._fit_intercept[p] and not (
                self._regularize_intercept[p] | self._is_intercept_only(p)
            ):
                is_regularized[0] = False
            self._is_regularized[p] = is_regularized

        self.coef_ = {
            p: np.zeros(self.n_features_[p]) for p in range(self.distribution.n_params)
        }
        self.coef_path_ = {p: None for p in range(self.distribution.n_params)}

        # We need to track the sum of weights for each
        # distribution parameter for online model selection
        self._sum_of_weights = {}
        self._mean_of_weights = {}

        self._schedule_iteration = self._make_iter_schedule()
        self._schedule_step_size = self._make_step_size_schedule()

        self._it_inner = np.zeros(self.distribution.n_params)
        if self.prefit_initial > 0:
            message = (
                f"Setting max_it_inner to {self.prefit_initial} for first iteration"
            )
            self._print_message(message=message, level=1)
            self._schedule_iteration[: self.prefit_initial] = 1
            self._current_min_it_outer = int(self.prefit_initial)
        else:
            self._current_min_it_outer = int(self.min_it_outer)

        message = "Starting fit call"
        self._print_message(message=message, level=1)
        (
            self._global_dev,
            self._it_outer,
        ) = self._outer_fit(
            X=X_dict,
            y=y,
            w=sample_weight,
        )
        message = "Finished fit call"
        self._print_message(message=message, level=1)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Update the fit for the online GAMLSS Model.

        Args:
            X (np.ndarray): Covariate data matrix $X$.
            y (np.ndarray): Response variable $Y$.
            sample_weight (Optional[np.ndarray], optional): User-defined sample weights. Defaults to None (all observations have the same weight).
        """
        X, y = validate_data(
            self, X=X, y=y, reset=False, dtype=[np.float64, np.float32]
        )
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)
        _ = type_of_target(y, raise_unknown=True)
        self._validate_inputs(X, y)

        self.n_observations_ += sample_weight.sum()
        self.n_training_ = {
            p: calculate_effective_training_length(
                self._forget[p], self.n_observations_
            )
            for p in range(self.distribution.n_params)
        }

        self._fv = self.predict_distribution_parameters(X, what="response")

        self._scaler.update(X, sample_weight=sample_weight)
        X_scaled = self._scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }

        if self.debug:
            self._debug_X_dict = X_dict
            self._debug_X_scaled = X_scaled
            self._debug_weights = {}
            self._debug_working_vectors = {}
            self._debug_dl1dlp1 = {}
            self._debug_dl2dlp2 = {}
            self._debug_eta = {}
            self._debug_fv = {}
            self._debug_coef = {}
            self._debug_coef_path = {}

        ## Reset rss and ic to avoid confusion
        ## These are only for viewing, not read!!
        self._best_ic[:] = 0
        self._best_ic_iterations[:] = 0
        self._rss_iterations[:] = 0

        ## Copy old values for updating
        self._x_gram_new = copy.copy(self._x_gram)
        self._y_gram_new = copy.copy(self._y_gram)
        self._sum_of_weights_new = copy.copy(self._sum_of_weights)
        self._mean_of_weights_new = copy.copy(self._mean_of_weights)

        self._rss_old = copy.copy(self._rss)
        self._model_selection_data_old = copy.copy(self._model_selection_data)

        # Clean schedules for iterations and step size
        self._schedule_iteration = self._make_iter_schedule()
        self._schedule_step_size = self._make_step_size_schedule()

        if self.prefit_update > 0:
            message = (
                f"Setting max_it_inner to {self.prefit_update} for first iteration"
            )
            self._print_message(message=message, level=1)
            self._schedule_iteration[: self.prefit_update] = 1
            self._current_min_it_outer = int(self.prefit_update)

        if self.cautious_updates:
            lower = self.distribution.quantile(0.005, self._fv).item()
            upper = self.distribution.quantile(0.995, self._fv).item()

            if np.any(y < lower) or np.any(y > upper):
                message = (
                    f"New observations are outliers given current estimates. "
                    f"y: {y}, Lower bound: {lower}, upper bound: {upper}"
                )
                self._print_message(message=message, level=1)
                self._schedule_iteration[:4] = 1
                self._schedule_step_size[:4, :, :] = 0.25
                self._current_min_it_outer = 4
            else:
                self._current_min_it_outer = int(self.min_it_outer)

        message = "Starting update call"
        self._print_message(message=message, level=1)
        (
            self._global_dev,
            self._it_outer,
        ) = self._outer_update(
            X=X_dict,
            y=y,
            w=sample_weight,
        )

        self._x_gram = copy.copy(self._x_gram_new)
        self._y_gram = copy.copy(self._y_gram_new)
        self._sum_of_weights = copy.copy(self._sum_of_weights_new)
        self._mean_of_weights = copy.copy(self._mean_of_weights_new)
        message = "Finished update call"
        self._print_message(message=message, level=1)
        return self

    def _outer_update(self, X, y, w):
        ## for new observations:
        global_di = np.sum(-2 * self.distribution.logpdf(y, self._fv) * w)
        global_dev = (1 - self._forget[0]) ** y.shape[0] * self._global_dev + global_di
        global_dev_old = global_dev + 1000
        it_outer = 0

        while True:
            # Check relative congergence
            if it_outer >= self._current_min_it_outer:
                if (
                    np.abs(global_dev_old - global_dev) / np.abs(global_dev_old)
                    < self.rel_tol_outer
                ):
                    break

                if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                    break

                # if global_dev > global_dev_old:
                #     message = (
                #         f"Outer iteration {it_outer}: Global deviance increased. Breaking."
                #         f"Current LL {global_dev}, old LL {global_dev_old}"
                #     )
                #     self._print_message(message=message, level=1)
                #     break

                if it_outer >= self.max_it_outer:
                    break

            global_dev_old = global_dev
            it_outer += 1

            for param in self._param_order:
                global_dev = self._inner_update(
                    X=X,
                    y=y,
                    w=w,
                    it_outer=it_outer,
                    param=param,
                    dv=global_dev,
                )
                message = f"Outer iteration {it_outer}: Fitted param {param}: Current deviance {global_dev}"
                self._print_message(message=message, level=2)

            message = (
                f"Outer iteration {it_outer}: Finished: current deviance {global_dev}"
            )
            self._print_message(message=message, level=1)

        return global_dev, it_outer

    def _outer_fit(self, X, y, w):
        global_di = -2 * self.distribution.logpdf(y, self._fv)
        global_dev = np.sum(w * global_di)
        global_dev_old = global_dev + 1000
        it_outer = 0

        while True:
            # Check relative congergence
            if it_outer >= self._current_min_it_outer:
                if (
                    np.abs(global_dev_old - global_dev) / np.abs(global_dev_old)
                    < self.rel_tol_outer
                ):
                    break

                if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                    break

                if it_outer >= self.max_it_outer:
                    break

                if global_dev > global_dev_old:
                    message = (
                        f"Outer iteration {it_outer}: Global deviance increased. Breaking."
                        f"Current deviance {global_dev}, old deviance {global_dev_old}"
                    )
                    self._print_message(message=message, level=0)
                    break

            global_dev_old = float(global_dev)
            it_outer += 1

            for param in self._param_order:
                global_dev = self._inner_fit(
                    X=X,
                    y=y,
                    w=w,
                    param=param,
                    it_outer=it_outer,
                    dv=global_dev,
                )
                message = f"Outer iteration {it_outer}: Fitted param {param}: current deviance {global_dev}"
                self._print_message(message=message, level=2)

            message = f"Outer iteration {it_outer}: Finished. Current deviance {global_dev}, old deviance {global_dev_old}"
            self._print_message(message=message, level=1)

        return (global_dev, it_outer)

    def _inner_fit(
        self,
        X,
        y,
        w,
        it_outer,
        param,
        dv,
    ):
        dv_start = np.sum(-2 * self.distribution.logpdf(y, self._fv) * w)
        dv_iterations = np.repeat(dv_start, self.max_it_inner + 1)
        fv_it = copy.copy(self._fv)
        fv_it_new = copy.copy(self._fv)
        step_decrease_counter = 0
        terminate = False
        bad_state = False

        message = f"Starting inner iteration param {param}, outer iteration {it_outer}, start DV {dv_start}"
        self._print_message(message=message, level=2)

        for it_inner in range(self._schedule_iteration[it_outer - 1]):
            # We can improve the fit by taking the conditional
            # start values for the first outer iteration and the first inner iteration
            # as soon the first parameter is fitted.
            step_it = self._schedule_step_size[it_outer - 1, it_inner, param]

            if (
                (it_inner == 0)
                and (it_outer == 1)
                and (param >= 1)
                and self.cond_start_val
            ):
                fv_it = self.distribution.calculate_conditional_initial_values(
                    y=y,
                    theta=fv_it,
                    param=param,
                )
                message = f"Outer iteration {it_outer}: Fitting Parameter {param}: Using conditional start value {fv_it[0, param]}."
                self._print_message(message=message, level=3)

            eta = self.distribution.link_function(fv_it[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, fv_it, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv_it, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            if self.debug:
                key = (param, it_outer, it_inner)
                self._debug_weights[key] = copy.copy(wt)
                self._debug_working_vectors[key] = copy.copy(wv)
                self._debug_dl1dlp1[key] = copy.copy(dl1dp1)
                self._debug_dl2dlp2[key] = copy.copy(dl2dp2)
                self._debug_eta[key] = copy.copy(eta)
                self._debug_fv[key] = copy.copy(fv_it)
                self._debug_dv[key] = float(dv_iterations[it_inner])

            ## Create the X and Y Gramian and the weight
            x_gram_it = self._method[param].init_x_gram(
                X=X[param],
                weights=(w * wt),
                forget=self._forget[param],
            )
            y_gram_it = self._method[param].init_y_gram(
                X=X[param],
                y=wv,
                weights=(w * wt),
                forget=self._forget[param],
            )
            # Select the model if we have a path-based method
            if self._method[param]._path_based_method:
                if (it_inner == 0) & (it_outer == 1):
                    beta_path_it = self._method[param].fit_beta_path(
                        x_gram=x_gram_it,
                        y_gram=y_gram_it,
                        is_regularized=self._is_regularized[param],
                    )
                elif it_inner > 0:
                    beta_path_it = self._method[param].update_beta_path(
                        x_gram=x_gram_it,
                        y_gram=y_gram_it,
                        is_regularized=self._is_regularized[param],
                        beta_path=beta_path_it,
                    )
                elif (it_outer > 1) & (it_inner == 0):
                    beta_path_it = self._method[param].update_beta_path(
                        x_gram=x_gram_it,
                        y_gram=y_gram_it,
                        is_regularized=self._is_regularized[param],
                        beta_path=self.coef_path_[param],
                    )
                else:
                    print("This should not happen")

                beta_it, model_selection_data_it, best_ic_it = self._fit_select_model(
                    X=X[param],
                    y=y,
                    w=w,
                    wv=wv,
                    wt=wt,
                    beta_path=beta_path_it,
                    param=param,
                )
                self._best_ic_iterations[param, it_outer - 1, it_inner] = best_ic_it

            else:
                beta_path_it = None
                beta_it = self._method[param].fit_beta(
                    x_gram=x_gram_it,
                    y_gram=y_gram_it,
                    is_regularized=self._is_regularized[param],
                )

            if self.debug:
                self._debug_coef[key] = copy.copy(beta_it)
                self._debug_coef_path[key] = copy.copy(beta_path_it)

            # Calculate the prediction, residuals and RSS
            f = init_forget_vector(self._forget[param], y.shape[0])
            prediction_it = step_it * (X[param] @ beta_it.T) + (1 - step_it) * eta
            residuals_it = wv - prediction_it
            rss_it = np.sum(residuals_it**2 * wt * w * f) / np.mean(wt * w * f)

            # Calculate the fitted values and the deviance
            fv_it_new[:, param] = self.distribution.link_inverse(
                prediction_it, param=param
            )
            dv_it = np.sum(-2 * self.distribution.logpdf(y, fv_it_new) * w)
            dv_old = dv_iterations[it_inner]
            dv_increasing = dv_it > dv_old

            # This should really not happen unless your start values are
            # way to good and the model cannot reach these
            bad_state = dv_it > dv_iterations[0]

            # print(dv_it, dv_old, dv_increasing)
            message = f"Outer iteration {it_outer}: Fitting Parameter {param}: Inner iteration {it_inner}: Current Deviance {dv_it}"
            self._print_message(message=message, level=3)

            if dv_increasing and it_inner < (
                self._schedule_iteration[it_outer - 1] - 1
            ):
                # print("Blabal")
                step_decrease_counter += 1
                self._schedule_step_size[it_outer - 1, it_inner + 1, param] = (
                    step_it / 2
                )
                self._print_message(
                    f"Deviance increasing, step size halved. {step_decrease_counter}",
                    level=1,
                )
                if step_decrease_counter > 5:
                    message = f"Step size too small. Parameter {param}, Outer iteration {it_outer}, Inner iteration {it_inner}."
                    self._print_message(message=message, level=1)
                    terminate = True

            if (it_outer == 1) & (it_inner >= 1) | (it_outer >= 2):
                # Allow to break in principle.
                if abs(dv_old - dv_it) <= self.abs_tol_inner:
                    terminate = True
                if abs(dv_old - dv_it) / abs(dv_old) < self.rel_tol_inner:
                    terminate = True
                if it_inner == (self.max_it_inner - 1):
                    message = f"Reached max inner iteration in inner fit. Parameter:{param}, Outer iteration: {it_outer}, Inner iteration: {it_inner}."
                    self._print_message(message=message, level=3)
                    terminate = True

            if terminate and (it_outer == 1) and bad_state:
                message = (
                    f"The model ended in a bad state in the first outer iteration of param {param}. This is not a good sign.  \n"
                    f"The deviance increased from the start values to the current fit in inner iteration {it_inner}. \n"
                    f"The deviance is {dv_it} and the starting deviance for this inner iterations is {dv_iterations[0]}. \n"
                    "Please check your data and model. \n"
                    "Please turn on logging (verbose=3, debug=True) and check the debug information. \n"
                    "Consider using a pre-fit via the iteration_schedule and set the inner iterations to 1-2 for the first outer iteration."
                )
                self._print_message(message=message, level=0)

            if terminate:
                break
            if not dv_increasing:
                # print("deviance decreasing, write to fv_it")
                fv_it[:, param] = fv_it_new[:, param]
                # Set the deviance for the next inner iteration
                dv_iterations[(it_inner + 1) :] = dv_it

        if (not bad_state) or (it_outer == 1):
            # Write everything to the class
            self._x_gram[param] = x_gram_it
            self._y_gram[param] = y_gram_it
            self.coef_[param] = beta_it
            self.coef_path_[param] = beta_path_it
            self._fv[:, param] = fv_it_new[:, param]

            # Sum and mean of the weights
            self._sum_of_weights[param] = np.sum(w * wt)
            self._mean_of_weights[param] = np.mean(w * wt)

            # RSS
            self._rss[param] = rss_it
            self._rss_iterations[param, it_outer - 1, it_inner - 1] = rss_it

            if self._method[param]._path_based_method:
                self._model_selection_data[param] = model_selection_data_it
                self._best_ic[param] = best_ic_it

        self._it_inner[param] = it_inner
        return dv_it

    def _inner_update(
        self,
        X,
        y,
        w,
        it_outer,
        dv,
        param,
    ):
        # di = -2 * self.distribution.logpdf(y, self._fv )
        # dv = (1 - self._forget[0]) * self._global_dev + np.sum(di * w)
        # olddv = dv + 1

        dv_start = (
            np.sum(-2 * self.distribution.logpdf(y, self._fv) * w)
            + (1 - self._forget[0]) ** y.shape[0]
            * self._global_dev  # global dev is previous observation / fit
        )
        dv_iterations = np.repeat(dv_start, self.max_it_inner + 1)
        fv_it = copy.copy(self._fv)
        fv_it_new = copy.copy(self._fv)
        step_decrease_counter = 0
        terminate = False

        for it_inner in range(self._schedule_iteration[it_outer - 1]):
            step_it = self._schedule_step_size[it_outer - 1, it_inner, param]

            eta = self.distribution.link_function(fv_it[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, fv_it, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv_it, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, -1e10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            if self.debug:
                key = (param, it_outer, it_inner)
                self._debug_weights[key] = copy.copy(wt)
                self._debug_working_vectors[key] = copy.copy(wv)
                self._debug_dl1dlp1[key] = copy.copy(dl1dp1)
                self._debug_dl2dlp2[key] = copy.copy(dl2dp2)
                self._debug_eta[key] = copy.copy(eta)
                self._debug_fv[key] = copy.copy(fv_it)

            x_gram_it = self._method[param].update_x_gram(
                gram=self._x_gram[param],
                X=X[param],
                weights=(w * wt),
                forget=self._forget[param],
            )
            y_gram_it = self._method[param].update_y_gram(
                gram=self._y_gram[param],
                X=X[param],
                y=wv,
                weights=(w * wt),
                forget=self._forget[param],
            )
            # Select the model if we have a path-based method
            if self._method[param]._path_based_method:
                beta_path_it = self._method[param].update_beta_path(
                    x_gram=x_gram_it,
                    y_gram=y_gram_it,
                    beta_path=self.coef_path_[param],
                    is_regularized=self._is_regularized[param],
                )

                beta_it, model_selection_data_it, best_ic_it = (
                    self._update_select_model(
                        X=X[param],
                        y=y,
                        w=w,
                        wv=wv,
                        wt=wt,
                        beta_path=beta_path_it,
                        model_selection_data=self._model_selection_data_old[param],
                        param=param,
                    )
                )
                self._best_ic[param] = best_ic_it
                self._best_ic_iterations[param, it_outer - 1, it_inner] = best_ic_it
            else:
                beta_it = self._method[param].update_beta(
                    x_gram=x_gram_it,
                    y_gram=y_gram_it,
                    beta=self.coef_[param],
                    is_regularized=self._is_regularized[param],
                )
                beta_path_it = None
                model_selection_data_it = None

            if self.debug:
                self._debug_coef[key] = copy.copy(beta_it)
                self._debug_coef_path[key] = copy.copy(beta_path_it)

            # Calculate the prediction, residuals and RSS
            f = init_forget_vector(self._forget[param], y.shape[0])
            prediction_it = X[param] @ beta_it.T
            residuals_it = wv - prediction_it
            rss_it = np.sum(residuals_it**2 * wt * w * f) / np.mean(wt * w * f)

            denom = online_mean_update(
                self._mean_of_weights[param],
                wt,
                self._forget[param],
                self.n_observations_,
            )
            sum_of_rss_it = (
                rss_it
                + (1 - self._forget[param]) ** y.shape[0]
                * (self._rss_old[param] * self._mean_of_weights[param])
            ) / denom

            # Calculate the fitted values and the deviance
            fv_it_new[:, param] = self.distribution.link_inverse(
                prediction_it, param=param
            )
            dv_it = (
                np.sum(-2 * self.distribution.logpdf(y, fv_it_new) * w)
                + (1 - self._forget[0]) ** y.shape[0] * self._global_dev
            )
            dv_old = dv_iterations[it_inner]
            dv_increasing = dv_it > dv_old

            if dv_increasing:
                step_decrease_counter += 1
                self._schedule_step_size[it_outer - 1, it_inner, param] = step_it / 2
                self._print_message(
                    f"Deviance increasing, step size halved. {step_decrease_counter}",
                    level=1,
                )
                if step_decrease_counter > 5:
                    message = f"Step size too small. Parameter {param}, Outer iteration {it_outer}, Inner iteration {it_inner}."
                    self._print_message(message=message, level=1)
                    terminate = True

            if (not dv_increasing) | (it_inner < self.max_it_inner - 1):
                # Allow to break in principle.
                if abs(dv_old - dv_it) <= self.abs_tol_inner:
                    terminate = True
                if abs(dv_old - dv_it) / abs(dv_old) < self.rel_tol_inner:
                    terminate = True
                if it_inner == (self.max_it_inner - 1):
                    message = f"Reached max inner iteration in inner fit. Parameter:{param}, Outer iteration: {it_outer}, Inner iteration: {it_inner}."
                    self._print_message(message=message, level=3)
                    terminate = True

            message = f"Outer iteration {it_outer}: Fitting Parameter {param}: Inner iteration {it_inner}: Current DV {dv_it}, previous DV {dv_old}, step size {step_it}"
            self._print_message(message=message, level=3)

            if terminate:
                break
            if not dv_increasing:
                # print("deviance decreasing, write to fv_it")
                fv_it[:, param] = fv_it_new[:, param]
                # Set the deviance for the next inner iteration
                dv_iterations[(it_inner + 1) :] = dv_it

            # olddv = dv
            # di = -2 * self.distribution.logpdf(y, self._fv )
            # dv = np.sum(di * w)

        # Assign to class variables
        self._x_gram_new[param] = x_gram_it
        self._y_gram_new[param] = y_gram_it
        self._fv[:, param] = fv_it_new[:, param]

        self.coef_[param] = beta_it
        self.coef_path_[param] = beta_path_it
        self._rss[param] = sum_of_rss_it
        self._rss_iterations[param, it_outer - 1, it_inner] = sum_of_rss_it
        self._model_selection_data[param] = model_selection_data_it

        # Update the weights
        self._sum_of_weights_new[param] = (
            np.sum(w * wt) + (1 - self._forget[param]) * self._sum_of_weights[param]
        )
        self._mean_of_weights_new[param] = (
            self._sum_of_weights_new[param] / self.n_training_[param]
        )
        self._it_inner[param] = it_inner
        return dv_it

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the mean of the response distribution.

        Args:
            X (np.ndarray): Covariate matrix $X$. Shape should be (n_samples, n_features).
        Raises:
            NotFittedError: If the model is not fitted yet.

        Returns:
            Predictions (np.ndarray): Predictions
        """
        theta = self.predict_distribution_parameters(X)
        return self.distribution.mean(theta)

    def predict_median(self, X: np.ndarray):
        """Predict the median of the distribution.

        Args:
            X (np.ndarray): Covariate matrix $X$. Shape should be (n_samples, n_features).
        Raises:
            NotFittedError: If the model is not fitted yet.

        Returns:
            Predictions (np.ndarray): Predicted median of the distribution. Shape will be (n_samples,).
        """
        theta = self.predict_distribution_parameters(X)
        return self.distribution.median(theta)

    def predict_distribution_parameters(
        self,
        X: np.ndarray,
        what: str = "response",
        return_contributions: bool = False,
    ) -> np.ndarray:
        """Predict the distibution parameters given input data.

        Args:
            X (np.ndarray): Design matrix.
            what (str, optional): Predict the response or the link. Defaults to "response". Remember the  GAMLSS models $g(\\theta) = X^T\\beta$. Predict `"link"` will output $X^T\\beta$, predict `"response"` will output $g^{-1}(X^T\\beta)$. Usually, you want predict = `"response"`.
            return_contributions (bool, optional): Whether to return a `Tuple[prediction, contributions]` where the contributions of the individual covariates for each distribution parameter's predicted value is specified. Defaults to False.

        Raises:
            ValueError: Raises if `what` is not in `["link", "response"]`.

        Returns:
            Predictions (np.ndarray): Predicted values for the distribution of shape (n_samples, n_params) where n_params is the number of distribution parameters.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, dtype=[np.float64, np.float32])

        X_scaled = self._scaler.transform(X=X)
        X_dict = {
            p: self.make_model_array(X_scaled, p)
            for p in range(self.distribution.n_params)
        }
        prediction = [x @ b.T for x, b in zip(X_dict.values(), self.coef_.values())]

        if return_contributions:
            contribution = [
                x * b.T for x, b in zip(X_dict.values(), self.coef_.values())
            ]

        if what == "response":
            prediction = [
                self.distribution.link_inverse(p, param=i)
                for i, p in enumerate(prediction)
            ]
        elif what == "link":
            pass
        else:
            raise ValueError("Should be 'response' or 'link'.")

        prediction = np.stack(prediction, axis=1)

        if return_contributions:
            return (prediction, contribution)
        else:
            return prediction

    def predict_quantile(
        self,
        X: np.ndarray,
        quantile: float | np.ndarray,
    ) -> np.ndarray:
        """Predict the quantile(s) of the distribution.

        Args:
            X (np.ndarray): Covariate matrix $X$. Shape should be (n_samples, n_features).
            quantile (float | np.ndarray): Quantile(s) to predict.

        Returns:
            np.ndarray: Predicted quantile(s) of the distribution. Shape will be (n_samples, n_quantiles).
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, dtype=[np.float64, np.float32])

        theta = self.predict_distribution_parameters(X, what="response")
        if isinstance(quantile, np.ndarray):
            quantile_pred = self.distribution.ppf(quantile[:, None], theta).T
        else:
            quantile_pred = self.distribution.ppf(quantile, theta).reshape(-1, 1)

        return quantile_pred

    def get_debug_information(
        self,
        variable: str = "coef",
        param: int = 0,
        it_outer: int = 1,
        it_inner: int = 1,
    ):
        """Get debug information for a specific variable, parameter, outer iteration and inner iteration.

        We currently support the following variables:

        * "X_dict": The design matrix for the distribution parameter.
        * "X_scaled": The scaled design matrix.
        * "weights": The sample weights for the distribution parameter.
        * "working_vectors": The working vectors for the distribution parameter.
        * "dl1dlp1": The first derivative of the log-likelihood with respect to the distribution parameter.
        * "dl2dlp2": The second derivative of the log-likelihood with respect to the distribution parameter.
        * "eta": The linear predictor for the distribution parameter.
        * "fv": The fitted values for the distribution parameter.
        * "dv": The deviance for the distribution parameter.
        * "coef": The coefficients for the distribution parameter.
        * "coef_path": The coefficients path for the distribution parameter.

        Args:
            variable (str): The variable to get debug information for. Defaults to "coef".
            param (int): The distribution parameter to get debug information for. Defaults to 0.
            it_outer (int): The outer iteration to get debug information for. Defaults to 1.
            it_inner (int): The inner iteration to get debug information for. Defaults to 1.
        Returns:
            Any: The debug information for the specified variable, parameter, outer iteration and inner iteration.
        Raises:
            ValueError: If debug mode is not enabled.

        """
        if not self.debug:
            raise ValueError(
                "Debug mode is not enabled. Please set debug=True when initializing the OnlineDistributionalRegression estimator."
            )

        key = (param, it_outer, it_inner + 1)
        return getattr(self, f"_debug_{variable}")[key]
