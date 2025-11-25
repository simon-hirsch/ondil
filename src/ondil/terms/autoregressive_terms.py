import copy
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..base import EstimationMethod, Term
from ..design_matrix import add_intercept, make_lags
from ..methods import get_estimation_method


@dataclass(frozen=True)
class ARTermState:
    is_regularized: np.ndarray | None
    g: np.ndarray | None
    h: np.ndarray | None
    coef_: np.ndarray | None
    memory: np.ndarray | None


class _AutoregressiveTerm(Term):
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

    features: np.ndarray | list[int] | Literal["all"]
    method: EstimationMethod | str
    fit_intercept: bool = True
    forget: float = 0.0
    is_regularized: bool = False
    regularize_intercept: None | bool = None

    def __init__(
        self,
        lags: np.ndarray | list[int] | int = 1,
        method: EstimationMethod = "ols",
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: bool = False,
        regularize_intercept: None | bool = None,
    ):
        self.fit_intercept = fit_intercept
        self.method = method
        self.forget = forget
        self.is_regularized = is_regularized
        self.regularize_intercept = regularize_intercept
        self.lags = lags

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if self._method._path_based_method:
            raise ValueError("Path-based methods are not supported for LinearTerm.")
        return self

    def fit(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ) -> "_AutoregressiveTerm":
        X_mat, memory = self.make_design_matrix_in_sample_during_fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
        )
        self.find_zero_variance_columns(X_mat)
        X_mat = self.remove_zero_variance_columns(X_mat)

        if self.is_regularized:
            is_regularized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            is_regularized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            is_regularized[0] = False

        g = self._method.init_x_gram(
            X=X_mat,
            weights=sample_weight,
            forget=self.forget,
        )
        h = self._method.init_y_gram(
            X=X_mat,
            y=y,
            weights=sample_weight,
            forget=self.forget,
        )
        coef_ = self._method.fit_beta(
            x_gram=g,
            y_gram=h,
            is_regularized=is_regularized,
        )

        # not that max(self.lags) does not work if lags is an integer.
        # np.max converts it to an array first.
        new = copy.copy(self)
        new._state = ARTermState(
            is_regularized=is_regularized,
            g=g,
            h=h,
            coef_=coef_,
            memory=memory,
        )
        return new

    def predict_in_sample_during_fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
    ) -> np.ndarray:
        X_mat, _ = self.make_design_matrix_in_sample_during_fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
        )
        X_mat = self.remove_zero_variance_columns(X_mat)
        return X_mat @ self._state.coef_

    def predict_in_sample_during_update(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
    ):
        X_mat, _ = self.make_design_matrix_in_sample_during_update(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
        )
        X_mat = self.remove_zero_variance_columns(X_mat)
        return X_mat @ self._state.coef_

    def make_design_matrix_out_of_sample(
        self,
        X,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )
        X_mat = make_lags(
            y=self._state.memory,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[[-1], :]

    # TODO: Can this go to the base class?
    def predict_out_of_sample(
        self,
        X: np.ndarray,  # for api compatibility; not used
    ):
        X_mat = self.make_design_matrix_out_of_sample(X=X)
        X_mat = self.remove_zero_variance_columns(X_mat)
        return X_mat @ self._state.coef_

    def update(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
        distribution=None,
        sample_weight: np.ndarray = None,
    ) -> "_AutoregressiveTerm":
        if self._state is None:
            raise ValueError("Term must be fitted before it can be updated.")

        X_mat, memory = self.make_design_matrix_in_sample_during_update(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
        )
        X_mat = self.remove_zero_variance_columns(X_mat)

        g = self._method.update_x_gram(
            gram=self._state.g,
            X=X_mat,
            weights=sample_weight,
            forget=self.forget,
        )
        h = self._method.update_y_gram(
            gram=self._state.h,
            X=X_mat,
            y=y,
            weights=sample_weight,
            forget=self.forget,
        )
        coef_ = self._method.fit_beta(
            x_gram=g,
            y_gram=h,
            is_regularized=self._state.is_regularized,
        )

        new_instance = copy.copy(self)
        new_instance._state = replace(
            self._state,
            is_regularized=self._state.is_regularized,
            g=g,
            h=h,
            coef_=coef_,
            memory=memory,
        )
        return new_instance


class AutoregressiveThetaTerm(_AutoregressiveTerm):
    """Autoregressive term using target values for lagged predictors."""

    def __init__(
        self,
        lags: np.ndarray | list[int] | int = 1,
        method: EstimationMethod = "ols",
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: bool = False,
        regularize_intercept: None | bool = None,
        param: int = 0,
    ):
        super().__init__(
            lags=lags,
            method=method,
            fit_intercept=fit_intercept,
            forget=forget,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
        )
        self.param = param

    def make_design_matrix_in_sample_during_fit(
        self,
        X,
        y,
        fitted_values,
        target_values,
        distribution=None,
    ):
        X_mat = make_lags(y=fitted_values[:, self.param], lags=self.lags)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat, fitted_values[-np.max(self.lags), self.param]

    def make_design_matrix_in_sample_during_update(
        self,
        X,
        y,
        fitted_values,
        target_values,
        distribution=None,
    ):
        lagged_value = np.concatenate((
            self._state.memory,
            fitted_values[:, self.param],
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-y.shape[0] :, :], lagged_value[-np.max(self.lags), self.param]


class AutoregressiveTargetTerm(_AutoregressiveTerm):
    """Autoregressive term using fitted values for lagged predictors."""

    def make_design_matrix_in_sample_during_fit(
        self,
        X,
        y,
        fitted_values,
        target_values,
    ):
        X_mat = make_lags(y=target_values, lags=self.lags)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat, target_values[-np.max(self.lags) :]

    def make_design_matrix_in_sample_during_update(
        self,
        X,
        y,
        fitted_values,
        target_values,
        distribution=None,
    ):
        lagged_value = np.concatenate((
            self._state.memory,
            target_values,
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-y.shape[0] :, :], lagged_value[-np.max(self.lags) :]


class AutoregressiveSquaredResidualTerm(_AutoregressiveTerm):
    """Autoregressive term using fitted values for lagged predictors."""

    def make_design_matrix_in_sample_during_fit(
        self,
        X,
        y,
        fitted_values,
        target_values,
        distribution=None,
    ):
        squared_residuals = (target_values - fitted_values[:, 0]) ** 2
        X_mat = make_lags(y=squared_residuals, lags=self.lags)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat, squared_residuals[-np.max(self.lags) :]

    def make_design_matrix_in_sample_during_update(
        self,
        X,
        y,
        fitted_values,
        target_values,
        distribution=None,
    ):
        squared_residuals = (target_values - fitted_values[:, 0]) ** 2
        lagged_value = np.concatenate((
            self._state.memory,
            squared_residuals,
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-y.shape[0] :, :], lagged_value[-np.max(self.lags) :]
