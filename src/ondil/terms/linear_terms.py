from typing import Literal

import numpy as np

from ..base import EstimationMethod, Term
from ..design_matrix import add_intercept, subset_array
from ..methods import get_estimation_method


class LinearTerm(Term):
    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"],
        method: EstimationMethod,
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: bool = False,
        regularize_intercept: None | bool = None,
    ):
        self.features = features
        self.fit_intercept = fit_intercept
        self.method = method
        self.forget = forget
        self.is_regularized = is_regularized
        self.regularize_intercept = regularize_intercept

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if self._method._path_based_method:
            raise ValueError("Path-based methods are not supported for LinearTerm.")
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "LinearTerm":

        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

        if self.is_regularized:
            self._is_reguarlized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            self._is_reguarlized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            self._is_reguarlized[0] = False

        self._g = self._method.init_x_gram(
            X=X_mat,
            weights=sample_weight,
            forget=self.forget,
        )
        self._h = self._method.init_y_gram(
            X=X_mat,
            y=y,
            weights=sample_weight,
            forget=self.forget,
        )
        self._coef_ = self._method.fit_beta(
            x_gram=self._g,
            y_gram=self._h,
            is_regularized=self._is_reguarlized,
        )

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

        return X_mat @ self._coef_

    def update(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "LinearTerm":
        super().update(X, residuals, sample_weight)
