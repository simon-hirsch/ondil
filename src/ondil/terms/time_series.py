import copy
from dataclasses import dataclass, replace

import numpy as np

from ..base import Distribution, EstimationMethod
from ..base.terms import FeatureTransformation
from ..design_matrix import add_intercept, make_lags
from ..methods import get_estimation_method
from .features import TimeSeriesFeature
from .linear import _LinearBaseTerm, _LinearPathModelSelection


@dataclass(frozen=True)
class ARTermState:
    is_regularized: np.ndarray | None
    g: np.ndarray | None
    h: np.ndarray | None
    coef_: np.ndarray | None
    memory_fitted_values: np.ndarray | None
    memory_target_values: np.ndarray | None


@dataclass(frozen=True)
class ARTermPathState:
    is_regularized: np.ndarray | None
    g: np.ndarray | None
    h: np.ndarray | None
    coef_: np.ndarray | None
    coef_path_: np.ndarray | None
    best_idx: int | None
    ic_values_: np.ndarray | None
    rss: np.ndarray | None
    n_observations: int | None
    n_nonzero_coef: np.ndarray | None
    memory_fitted_values: np.ndarray | None
    memory_target_values: np.ndarray | None


class TimeSeriesTerm(_LinearBaseTerm):
    """Autoregressive term for time series modeling.

    This is a rather generic base class which, in the fit and update methods, will
    take the "y" values and create lagged versions of them to be used as predictors.

    However, we usually do not want to use the lagged versions of the "y" values but
    rather some values of the distributional parameters, e.g. the fitted mean or
    the target values (since y is the working vector in distributional regression).

    Therefore we use this class as a base class for the actual AutoregressiveTerm and
    we only need to override the method that creates the lagged values.

    Parameters
    ----------
    lags : list[int]
        List of lag orders to include in the autoregressive term.
    method : EstimationMethod | str, default="ols"
        Estimation method to use. Can be an instance of EstimationMethod or a string
        identifier for a predefined method.
    intercept : bool, default=True
        Whether to include an intercept in the model.
    """

    allow_online_updates: bool = True

    def __init__(
        self,
        effects: list[FeatureTransformation],
        method: EstimationMethod = "ols",
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: bool = False,
        regularize_intercept: None | bool = False,
        constraint_matrix: np.ndarray | None = None,
        constraint_bounds: np.ndarray | None = None,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            method=method,
            forget=forget,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
        )
        self.effects = effects

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if self._method._path_based_method:
            raise ValueError("Path-based methods are not supported for LinearTerm.")
        self.lags = [
            np.max(term.lags) if hasattr(term, "lags") else 0 for term in self.effects
        ]
        return self

    def fit(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "TimeSeriesTerm":
        g, h, coef_, is_regularized = self._fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
            estimation_weight=estimation_weight,
        )
        # not that max(self.lags) does not work if lags is an integer.
        # np.max converts it to an array first.
        new = copy.copy(self)
        new._state = ARTermState(
            is_regularized=is_regularized,
            g=g,
            h=h,
            coef_=coef_,
            memory_fitted_values=fitted_values[-np.max(self.lags) :],
            memory_target_values=target_values[-np.max(self.lags) :],
        )
        return new

    def predict_in_sample_during_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        X_mat = self.make_design_matrix_in_sample_during_fit(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )
        X_mat = self.remove_problematic_columns(X_mat)
        return X_mat @ self._state.coef_

    def predict_in_sample_during_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        X_mat = self.make_design_matrix_in_sample_during_update(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )
        X_mat = self.remove_problematic_columns(X_mat)
        return X_mat @ self._state.coef_

    def predict_out_of_sample(
        self,
        X: np.ndarray,  # for api compatibility; not used
        distribution: Distribution,
    ):
        X_mat = self.make_design_matrix_out_of_sample(X=X, distribution=distribution)
        # X_mat = self.remove_problematic_columns(X_mat)
        return X_mat @ self._state.coef_

    def update(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "TimeSeriesTerm":
        if self._state is None:
            raise ValueError("Term must be fitted before it can be updated.")

        g, h, coef_ = self._update(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
            estimation_weight=estimation_weight,
        )

        new_instance = copy.copy(self)
        new_instance._state = replace(
            self._state,
            is_regularized=self._state.is_regularized,
            g=g,
            h=h,
            coef_=coef_,
            memory_fitted_values=np.concatenate(
                (self._state.memory_fitted_values, fitted_values),
            )[-np.max(self.lags) :],
            memory_target_values=np.concatenate(
                (self._state.memory_target_values, target_values),
            )[-np.max(self.lags) :],
        )
        return new_instance

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        X_mats = [
            term.make_design_matrix_in_sample_during_fit(
                X=X,
                fitted_values=fitted_values,
                target_values=target_values,
                distribution=distribution,
            )
            for term in self.effects
        ]
        X_mat = np.hstack(X_mats)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        X_mats = [
            term.make_design_matrix_in_sample_during_update(
                X=X,
                fitted_values=fitted_values,
                target_values=target_values,
                distribution=distribution,
                state=self._state,
            )
            for term in self.effects
        ]
        X_mat = np.hstack(X_mats)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat

    def make_design_matrix_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ):
        X_mats = [
            term.make_design_matrix_out_of_sample(
                X,
                distribution,
                state=self._state,
            )
            for term in self.effects
        ]
        X_mat = np.hstack(X_mats)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat


class RegularizedTimeSeriesTerm(
    _LinearPathModelSelection,
    TimeSeriesTerm,
    # Left to right inheritance to ensure correct MRO
    # we get the _fit and the _update methods from
    # _LinearPathModelSelection
):
    """Regularized autoregressive term for time series modeling with model selection.

    This class extends the TimeSeriesTerm to include regularization
    and model selection capabilities using information criteria.

    Parameters
    ----------
    lags : list[int]
        List of lag orders to include in the autoregressive term.
    method : EstimationMethod | str, default="lasso"
        Estimation method to use. Can be an instance of EstimationMethod or a string
        identifier for a predefined method.
    intercept : bool, default=True
        Whether to include an intercept in the model.
    ic : str, default="aic"
        Information criterion to use for model selection.
    """

    def __init__(
        self,
        effects: list[FeatureTransformation],
        method: EstimationMethod = "lasso",
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: np.ndarray | None = None,
        regularize_intercept: bool = False,
        weighted_regularization: bool = False,
        ic: str = "aic",
        constraint_matrix: np.ndarray | None = None,
        constraint_bounds: np.ndarray | None = None,
    ):
        super().__init__(
            method=method,
            fit_intercept=fit_intercept,
            forget=forget,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
            weighted_regularization=weighted_regularization,
            ic=ic,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
        )
        self.effects = effects

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if not self._method._path_based_method:
            raise ValueError("Non-Path-based methods are not supported for LinearTerm.")
        self.lags = [
            np.max(term.lags) if hasattr(term, "lags") else 0 for term in self.effects
        ]
        return self

    def fit(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "RegularizedTimeSeriesTerm":
        (
            is_regularized,
            g,
            h,
            coef_,
            coef_path_,
            best_idx,
            ic_values_,
            rss,
            n_observations,
            n_nonzero_coef,
        ) = self._fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
            estimation_weight=estimation_weight,
        )
        self._state = ARTermPathState(
            is_regularized=is_regularized,
            g=g,
            h=h,
            coef_=coef_,
            coef_path_=coef_path_,
            best_idx=best_idx,
            ic_values_=ic_values_,
            rss=rss,
            n_observations=n_observations,
            n_nonzero_coef=n_nonzero_coef,
            memory_fitted_values=fitted_values[-np.max(self.lags) :],
            memory_target_values=target_values[-np.max(self.lags) :],
        )
        return self

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "RegularizedTimeSeriesTerm":
        (
            g,
            h,
            coef_,
            coef_path_,
            best_idx,
            ic_values_,
            rss,
            n_observations,
            n_nonzero_coef,
        ) = self._update(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
            estimation_weight=estimation_weight,
        )
        # Create a new instance with updated values
        new_instance = copy.copy(self)
        new_instance._state = replace(
            self._state,
            g=g,
            h=h,
            coef_=coef_,
            coef_path_=coef_path_,
            best_idx=best_idx,
            ic_values_=ic_values_,
            rss=rss,
            n_observations=n_observations,
            n_nonzero_coef=n_nonzero_coef,
            memory_fitted_values=np.concatenate(
                (self._state.memory_fitted_values, fitted_values),
            )[-np.max(self.lags) :],
            memory_target_values=np.concatenate(
                (self._state.memory_target_values, target_values),
            )[-np.max(self.lags) :],
        )
        return new_instance
