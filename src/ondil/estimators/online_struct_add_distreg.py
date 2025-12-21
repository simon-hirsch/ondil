import copy
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight, validate_data

from .. import HAS_PANDAS, HAS_POLARS
from ..base import Distribution, EstimationMethod, OndilEstimatorMixin
from ..distributions import Normal
from ..gram import init_forget_vector
from ..logging import logger
from ..scaler import OnlineScaler
from ..terms.linear_terms import LinearTerm

if HAS_PANDAS:
    pass
if HAS_POLARS:
    import polars as pl  # noqa


class InnerFitResult(NamedTuple):
    param: int
    inner_iteration: int
    fitted_values: np.ndarray
    deviance: float
    terms: list


class InnerUpdateResult(NamedTuple):
    param: int
    inner_iteration: int
    fitted_values: np.ndarray
    deviance: float
    terms: list


class OuterFitResult(NamedTuple):
    terms: dict
    deviance: float


class OuterUpdateResult(NamedTuple):
    terms: dict
    deviance: float


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
        learning_rate: float = 0.0,
        max_outer_iterations: int = 10,
        max_inner_iterations: int = 10,
        rel_tol_outer: float = 1e-3,
        rel_tol_inner: float = 1e-3,
        abs_tol_outer: float = 1e-3,
        abs_tol_inner: float = 1e-3,
    ):
        self.distribution = distribution
        self.terms = terms
        self.scale_inputs = scale_inputs
        self.learning_rate = learning_rate
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.verbose = verbose

    def _prepare_terms(self):
        logger.trace(
            "Preparing terms for fitting. Fitted terms will be stored in self.terms_",
        )
        if self.terms is None:
            logger.info(
                "No terms specified. Using default linear terms for all distribution parameters."
            )
            self.terms_ = {
                p: [LinearTerm(features="all", method="ols")]
                for p in range(self.distribution.n_params)
            }
        else:
            self.terms_ = copy.deepcopy(self.terms)

        for p in range(self.distribution.n_params):
            methods = set()
            for t, term in enumerate(self.terms_[p]):
                self.terms_[p][t] = term._prepare_term()
                set.add(methods, self.terms_[p][t].method)
            if len(methods) < len(self.terms_[p]):
                logger.warning(
                    f"Multiple terms with the same method for parameter {p}. "
                    f"Consider combining terms of the same method into a single term for improved accuracy and performance."
                )

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
        step_decrease_counter = 0
        step = 1.0
        forget_weight = init_forget_vector(
            size=y.shape[0],
            forget=self.learning_rate,
        )
        # Start values might be overly optimistic
        # Therefore we can have an increasing deviance in the first iteration
        # But then we want to keep the fitted values
        # So we compare against the starting deviance
        # based on CONSTANT initial values
        if outer_iteration == 0:
            deviance_start = -2 * np.sum(
                self.distribution.loglikelihood(
                    y=y,
                    theta=self.distribution.constant_initial_values(y),
                )
                * sample_weight
                * forget_weight
            )
        else:
            deviance_start = -2 * np.sum(
                self.distribution.loglikelihood(y, self._fitted_values)
                * sample_weight
                * forget_weight
            )
        deviance_iteration = np.repeat(deviance_start, self.max_inner_iterations + 1)
        fitted_values_iteration = self._fitted_values.copy()

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
            terms = []
            for t, term in enumerate(self.terms_[param]):
                working_vector_term = working_vector - eta_iteration
                terms.append(
                    term.fit(
                        X=X,
                        y=working_vector_term,
                        fitted_values=fitted_values_iteration,
                        target_values=y,
                        distribution=self.distribution,
                        sample_weight=sample_weight,
                        estimation_weight=estimation_weight,
                    )
                )
                eta_iteration += (
                    step
                    * terms[t].predict_in_sample_during_fit(
                        X=X,
                        y=working_vector_term,  # weird, TODO: remove this from the predict method
                        fitted_values=fitted_values_iteration,
                        target_values=y,
                        distribution=self.distribution,
                    )
                )

            # Line search could be implemented here
            # For now we just do a simple step size adjustment
            eta_iteration += (1 - step) * eta_start
            fitted_values_iteration[:, param] = self.distribution.link_inverse(
                eta_iteration, param=param
            )
            deviance_iteration[inner_iteration + 1] = -2 * np.sum(
                self.distribution.loglikelihood(y, fitted_values_iteration)
                * sample_weight
                * forget_weight
            )
            deviance_new = deviance_iteration[inner_iteration + 1]
            deviance_old = deviance_iteration[inner_iteration]

            message = (
                f"Inner iteration {inner_iteration + 1}, "
                f"param {param}, "
                f"deviance: {deviance_new:.3f}"
                f" (improvement: {deviance_old - deviance_new:.3f})"
            )
            logger.info(message)

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
                logger.info(message)
                if step_decrease_counter < 5:
                    step_decrease_counter += 1
                    step = step / 2
                    message = f"Reducing step size to {step:.5f} and retrying."
                    logger.info(message)
                else:
                    break

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
        forget_weight = init_forget_vector(size=y.shape[0], forget=self.learning_rate)
        self._iterations_outer = np.zeros(
            shape=(self.max_outer_iterations, self.distribution.n_params)
        )
        self._deviance_start = (
            np.sum(
                -2
                * self.distribution.loglikelihood(y, self._fitted_values)
                * sample_weight
                * forget_weight
            ),
        )
        self._deviance_outer = np.full(
            shape=(self.max_outer_iterations + 1, self.distribution.n_params),
            fill_value=self._deviance_start,
        )
        self._step_sizes = np.ones(
            shape=(self.max_outer_iterations, self.distribution.n_params)
        )

        terms = {}

        for outer_iteration in range(self.max_outer_iterations):
            message = (
                f"Outer iteration {outer_iteration + 1}, "
                f"deviance: {self._deviance_outer[outer_iteration, -1]:.3f}"
            )
            logger.info(message)

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
                self._iterations_outer[outer_iteration, param] = (
                    result.inner_iteration + 1
                )

                if (result.deviance <= self._deviance_outer[idx]) or (
                    outer_iteration == 0
                ):
                    # TODO: only take terms if we improve on the deviance
                    # Implement this properly
                    self._fitted_values[:, param] = result.fitted_values
                    self._deviance_outer[outer_iteration + 1, param] = result.deviance
                    terms[param] = result.terms
                else:
                    self._deviance_outer[outer_iteration + 1, param] = (
                        self._deviance_outer[idx]
                    )
                    # TODO: implement stepsize reduction
                    # Need to reduce the stepsize to this parameter to avoid divergence
                    # Do we want an outer and an inner stepsize?

                message = (
                    f"Outer iteration {outer_iteration + 1}: "
                    f"param {param}, "
                    f"deviance: {self._deviance_outer[outer_iteration + 1, param]:.3f}"
                )
                logger.info(message)

            if self._check_convergence(
                old_deviance=self._deviance_outer[outer_iteration, -1],
                new_deviance=self._deviance_outer[outer_iteration + 1, -1],
                tolerance=self.abs_tol_outer,
            ):
                break

        return OuterFitResult(
            deviance=self._deviance_outer[outer_iteration + 1, -1],
            terms=terms,
        )

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

        result = self._outer_fit(
            X=X_scaled,
            y=y,
            sample_weight=sample_weight,
        )
        self._deviance_final = result.deviance
        self.terms_ = result.terms
        logger.success("Model fit completed.")
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        X, y = validate_data(
            self, X=X, y=y, reset=False, dtype=[np.float64, np.float32]
        )
        _ = type_of_target(y, raise_unknown=True)
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)

        if self.scaler_:
            self.scaler_ = self.scaler_.update(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        self._n_observations += np.sum(sample_weight)
        self._fitted_values = self.predict(X)

        result = self._outer_update(
            X=X_scaled,
            y=y,
            sample_weight=sample_weight,
        )
        self.terms_.update(result.terms)
        self._deviance_final = result.deviance
        logger.success("Model update completed.")
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.scaler_:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        fitted_values = np.zeros(
            shape=(X.shape[0], self.distribution.n_params), dtype=X.dtype
        )

        for param in range(self.distribution.n_params):
            eta = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
            for term in self.terms_[param]:
                eta += term.predict_out_of_sample(
                    X=X_scaled,
                    distribution=self.distribution,
                )
            fitted_values[:, param] = self.distribution.link_inverse(eta, param=param)

        return fitted_values

    def _outer_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> OuterUpdateResult:
        # TODO: This needs the forget weights?
        self._deviance_start = (
            np.sum(
                -2
                * self.distribution.loglikelihood(y, self._fitted_values)
                * sample_weight
            )
            + (1 - self.learning_rate) ** y.shape[0] * self._deviance_final
        )
        self._deviance_outer = np.full(
            shape=(self.max_outer_iterations + 1, self.distribution.n_params),
            fill_value=self._deviance_start,
        )

        self._iterations_outer = np.zeros(
            shape=(self.max_outer_iterations, self.distribution.n_params)
        )

        # We start at the old terms in each inner update
        # Only the fitted values get updated (sufficient for the working vector)
        # But otherwise we run gramian updates on the terms in each inner x outer update
        # and hence over-adjust.
        # We need a place to store the new terms
        # Until we assign at the end of the outer outer update
        new_terms = {}

        for outer_iteration in range(self.max_outer_iterations):
            message = (
                f"Outer iteration {outer_iteration + 1}, "
                f"deviance: {self._deviance_outer[outer_iteration, +1]:.3f}"
            )
            logger.info(message)

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

                result = self._inner_update(
                    X=X,
                    y=y,
                    sample_weight=sample_weight,
                    outer_iteration=outer_iteration,
                    param=param,
                )
                self._iterations_outer[outer_iteration, param] = (
                    result.inner_iteration + 1
                )

                if result.deviance < self._deviance_outer[idx]:
                    # TODO: only take terms if we improve on the deviance
                    # Implement this properly
                    self._fitted_values[:, param] = result.fitted_values
                    self._deviance_outer[outer_iteration + 1, param] = result.deviance
                    new_terms[param] = result.terms
                else:
                    self._deviance_outer[outer_iteration + 1, param] = (
                        self._deviance_outer[idx]
                    )
                    # TODO: implement stepsize reduction
                    # Need to reduce the stepsize to this parameter to avoid divergence
                    # Do we want an outer and an inner stepsize?

                message = (
                    f"Outer iteration {outer_iteration + 1}: "
                    f"param {param}, "
                    f"deviance: {self._deviance_outer[outer_iteration + 1, param]:.3f}"
                )
                logger.info(message)

            if self._check_convergence(
                old_deviance=self._deviance_outer[outer_iteration, -1],
                new_deviance=self._deviance_outer[outer_iteration + 1, -1],
                tolerance=self.abs_tol_outer,
            ):
                break

        return OuterUpdateResult(
            deviance=self._deviance_outer[outer_iteration + 1, -1],
            terms=new_terms,
        )

    def _inner_update(
        self,
        X,
        y,
        sample_weight,
        outer_iteration: int,
        param: int,
    ):
        step_decrease_counter = 0
        step = 1.0

        deviance_start = (
            np.sum(
                -2
                * self.distribution.loglikelihood(y, self._fitted_values)
                * sample_weight
            )
            + (1 - self.learning_rate) ** y.shape[0] * self._deviance_final
        )

        deviance_iteration = np.repeat(deviance_start, self.max_inner_iterations + 1)
        fitted_values_iteration = self._fitted_values.copy()

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

            terms = []
            for t, term in enumerate(self.terms_[param]):
                working_vector_term = working_vector - eta_iteration
                terms.append(
                    term.update(
                        X=X,
                        y=working_vector_term,
                        fitted_values=fitted_values_iteration,
                        target_values=y,
                        distribution=self.distribution,
                        sample_weight=sample_weight,
                        estimation_weight=estimation_weight,
                    )
                )
                eta_iteration += (
                    step
                    * terms[t].predict_in_sample_during_update(
                        X=X,
                        y=working_vector_term,  # weird, TODO: remove this from the predict method
                        fitted_values=fitted_values_iteration,
                        target_values=y,
                        distribution=self.distribution,
                    )
                )

            eta_iteration += (1 - step) * eta_start

            fitted_values_iteration[:, param] = self.distribution.link_inverse(
                eta_iteration, param=param
            )
            deviance_iteration[inner_iteration + 1] = (
                -2
                * np.sum(
                    self.distribution.loglikelihood(y, fitted_values_iteration)
                    * sample_weight
                )
                + (1 - self.learning_rate) ** y.shape[0] * self._deviance_final
            )
            deviance_new = deviance_iteration[inner_iteration + 1]
            deviance_old = deviance_iteration[inner_iteration]

            message = (
                f"Inner iteration {inner_iteration + 1}, "
                f"param {param}, "
                f"deviance: {deviance_new:.3f}"
                f" (improvement: {deviance_old - deviance_new:.3f})"
            )
            logger.info(message)

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
                logger.info(message)
                if step_decrease_counter < 5:
                    step_decrease_counter += 1
                    step = step / 2
                    message = f"Reducing step size to {step:.5f} and retrying."
                    logger.info(message)
                    # Reset fitted values and terms
                else:
                    logger.info(
                        "Maximum step size reductions reached. Stopping inner update."
                    )
                    break

        return InnerFitResult(
            param=param,
            inner_iteration=inner_iteration,
            deviance=deviance_iteration[inner_iteration + 1],
            fitted_values=fitted_values_iteration[:, param],
            terms=terms,
        )
