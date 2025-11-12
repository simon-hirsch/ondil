import copy
import numbers
from typing import Any, Dict, Literal, NamedTuple, Optional, Union
import warnings

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
from ..terms.linear_terms import LinearTerm
from ..utils import calculate_effective_training_length, online_mean_update

if HAS_PANDAS:
    import pandas as pd
if HAS_POLARS:
    import polars as pl  # noqa


class InnerFitResult(NamedTuple):
    param: int
    inner_iteration: int
    fitted_values: np.ndarray
    deviance: float
    terms: list


class OnlineStructuredAdditiveDistributionRegressor(
    OndilEstimatorMixin,
    RegressorMixin,
    BaseEstimator,
):

    _parameter_constraints = {
        "distribution": [Distribution],
        "estimation_method": [EstimationMethod],
        "terms": [dict, type(None)],
        "scaler": [OnlineScaler, None],
    }

    def __init__(
        self,
        distribution: Distribution = Normal(),
        terms: Optional[Dict[str, Any]] = None,
        scale_inputs: bool = True,
        verbose: int = 0,
        max_outer_iterations: int = 10,
        max_inner_iterations: int = 10,
        rel_tol_outer: float = 1e-4,
        rel_tol_inner: float = 1e-4,
        abs_tol_outer: float = 1e-4,
        abs_tol_inner: float = 1e-4,
    ):
        self.distribution = distribution
        self.terms = terms
        self.scale_inputs = scale_inputs
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.verbose = verbose

    def _prepare_terms(self):
        if self.terms is None:
            self.terms_ = {
                p: [LinearTerm(features="all", method="ols")]
                for p in range(self.distribution.n_params)
            }
        else:
            self.terms_ = copy.deepcopy(self.terms)

        for p in range(self.distribution.n_params):
            for t, term in enumerate(self.terms_[p]):
                self.terms_[p][t] = term._prepare_term()

    def _prepare_scaler(self):
        if self.scale_inputs:
            self.scaler_ = OnlineScaler()
        else:
            self.scaler_ = None

    def _check_convergence(
        self, old_deviance: float, new_deviance: float, tolerance: float
    ) -> bool:
        if np.abs(old_deviance - new_deviance) < tolerance:
            return True
        if np.abs(old_deviance - new_deviance) / np.abs(old_deviance) < tolerance:
            return True
        return False

    def _inner_fit(
        self,
        X,
        y,
        sample_weight,
        outer_iteration: int,
        param: int,
    ):
        deviance_start = -2 * np.sum(
            self.distribution.logpdf(y, self._fitted_values) * sample_weight
        )
        deviance_iteration = np.repeat(deviance_start, self.max_inner_iterations + 1)
        fitted_values_iteration = self._fitted_values.copy()

        terms = copy.deepcopy(self.terms_[param])
        terms_iteration = copy.deepcopy(terms)

        for inner_iteration in range(self.max_inner_iterations):
            eta_start = self.distribution.link_function(
                fitted_values_iteration[:, param], param=param
            )
            eta_iteration = np.zeros_like(eta_start)
            dr = 1 / self.distribution.link_inverse_derivative(eta_start, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, fitted_values_iteration, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fitted_values_iteration, param=param)
            estimation_weight = np.clip(-(dl2dp2 / (dr * dr)), 1e-10, 1e10)
            working_vector = eta_start + dl1dp1 / (dr * estimation_weight)

            # Update the terms sequentially
            # TODO: don't save directly to self.terms_ during fitting
            # instead, create temporary terms to allow for rollback if deviance increases
            # Also this would allow parallel fitting of the CG algorithm
            for t, term in enumerate(terms):
                working_vector_term = working_vector - eta_iteration
                terms_iteration[t] = term.fit(
                    X=X,
                    y=working_vector_term,
                    sample_weight=sample_weight * estimation_weight,
                )
                eta_iteration += term.predict(X=X)

            fitted_values_iteration[:, param] = self.distribution.link_inverse(
                eta_iteration, param=param
            )
            deviance_iteration[inner_iteration + 1] = -2 * np.sum(
                self.distribution.logpdf(y, fitted_values_iteration) * sample_weight
            )
            deviance_new = deviance_iteration[inner_iteration + 1]
            deviance_old = deviance_iteration[inner_iteration]

            message = (
                f"  Inner iteration {inner_iteration + 1}, "
                f"param {param}, "
                f"deviance: {deviance_new:.3f}"
                f" (improvement: {deviance_old - deviance_new:.3f})"
            )
            self._print_message(message=message, level=2)

            if self._check_convergence(
                old_deviance=deviance_old,
                new_deviance=deviance_new,
                tolerance=self.abs_tol_inner,
            ):
                break

            if deviance_new > deviance_old:
                message = (
                    f"Deviance increased from {deviance_old:.3f} to {deviance_new:.3f}. "
                    f"Stopping inner optimization for param {param}."
                )
                self._print_message(message, level=2)
                break
            else:
                terms = terms_iteration

        return InnerFitResult(
            param=param,
            inner_iteration=inner_iteration,
            deviance=deviance_iteration[inner_iteration + 1],
            fitted_values=fitted_values_iteration[:, param],
            terms=terms,
        )

    def _outer_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ):
        self._iterations = np.zeros(
            shape=(self.max_outer_iterations, self.distribution.n_params)
        )
        self._deviance = np.full(
            shape=(self.max_outer_iterations + 1, self.distribution.n_params),
            fill_value=np.sum(
                -2 * self.distribution.logpdf(y, self._fitted_values) * sample_weight
            ),
        )

        for outer_iteration in range(self.max_outer_iterations):
            message = (
                f"Outer iteration {outer_iteration + 1}, "
                f"deviance: {self._deviance[outer_iteration, +1]:.3f}"
            )
            self._print_message(message, level=1)

            for param in range(self.distribution.n_params):
                # We want to keep the old fitted values if deviance does not improve
                # This ensures that the model does not diverge
                # And we can still check for convergence

                # This gives the flat index in the global_deviance array
                # For the previous outer iteration and parameter combination
                idx = divmod(
                    (outer_iteration + 1) * self.distribution.n_params + param - 1,
                    self.distribution.n_params,
                )

                result = self._inner_fit(
                    X=X,
                    y=y,
                    sample_weight=sample_weight,
                    outer_iteration=outer_iteration,
                    param=param,
                )
                self._iterations[outer_iteration, param] = result.inner_iteration + 1

                if result.deviance < self._deviance[idx]:
                    # TODO: only take terms if we improve on the deviance
                    # Implement this properly
                    self._fitted_values[:, param] = result.fitted_values
                    self._deviance[outer_iteration + 1, param] = result.deviance
                    self.terms_[param] = result.terms
                else:
                    self._deviance[outer_iteration + 1, param] = self._deviance[idx]
                    # TODO: implement stepsize reduction
                    # Need to reduce the stepsize to this parameter to avoid divergence
                    # Do we want an outer and an inner stepsize?

                message = (
                    f"Outer iteration {outer_iteration + 1}: "
                    f"param {param}, "
                    f"deviance: {self._deviance[outer_iteration + 1, param]:.3f}"
                )
                self._print_message(message, level=1)

            if self._check_convergence(
                old_deviance=self._deviance[outer_iteration, -1],
                new_deviance=self._deviance[outer_iteration + 1, -1],
                tolerance=self.abs_tol_outer,
            ):
                break

        # Do we want to return something here?

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        X, y = validate_data(self, X=X, y=y, reset=True, dtype=[np.float64, np.float32])
        _ = type_of_target(y, raise_unknown=True)
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)
        # self._validate_inputs(X, y)

        # prepare terms and scaler
        self._prepare_terms()
        self._prepare_scaler()

        self._n_observations = np.sum(sample_weight)
        self._fitted_values = self.distribution.initial_values(y)

        X_scaled = self.scaler_.fit_transform(X) if self.scaler_ else X

        self._outer_fit(
            X=X_scaled,
            y=y,
            sample_weight=sample_weight,
        )
        return self
