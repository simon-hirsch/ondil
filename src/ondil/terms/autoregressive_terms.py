import copy
from typing import Literal
from dataclasses import dataclass, replace, field

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
    memory = np.ndarray | None


class AutoregressiveTerm(Term):
    """Autoregressive term for time series modeling.

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
        features: np.ndarray | list[int] | Literal["all"],
        method: EstimationMethod,
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: bool = False,
        regularize_intercept: None | bool = None,
        lags: np.ndarray | list[int] | int = 1,
    ):
        self.features = features
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
        sample_weight: np.ndarray = None,
    ) -> "AutoregressiveTerm":
        X_mat = make_lags(y=y, lags=self.lags)

        if self.fit_intercept:
            X_mat = add_intercept(X_mat)

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
        self._state = ARTermState(
            is_regularized=is_regularized,
            g=g,
            h=h,
            coef_=coef_,
            memory=y[-np.max(self.lags) :],
        )
        return self

    def predict(
        self,
        X: np.ndarray,  # for api compatibility; not used
    ):
        X_mat = make_lags(
            y=self._state.memory,
            lags=self.lags,
        )[-1:, :]

        if self.fit_intercept:
            X_mat = add_intercept(X_mat)

        return X_mat @ self._state.coef_

    def update(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "AutoregressiveTerm":
        if self._state is None:
            raise ValueError("Term must be fitted before it can be updated.")

        y_full = np.concatenate([self._state.memory, y])
        X_mat = make_lags(
            y=y_full,
            lags=self.lags,
        )[-len(y) :, :]

        if self.fit_intercept:
            X_mat = add_intercept(X_mat)

        g = self._method.update_x_gram(
            x_gram=self._state.g,
            X=X_mat,
            weights=sample_weight,
            forget=self.forget,
        )
        h = self._method.update_y_gram(
            y_gram=self._state.h,
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
            memory=y_full[-np.max(self.lags) :],
        )

        return
