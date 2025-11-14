from typing import Literal

import numpy as np

from ..base import EstimationMethod, Term
from ..design_matrix import add_intercept, subset_array
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method


class LinearTerm(Term):
    """Linear term for structured additive distributional regression."""

    allow_online_updates: bool = True

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
            self._is_regularized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            self._is_regularized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            self._is_regularized[0] = False

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
        self.coef_ = self._method.fit_beta(
            x_gram=self._g,
            y_gram=self._h,
            is_regularized=self._is_regularized,
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

        return X_mat @ self.coef_

    def update(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "LinearTerm":
        super().update(X, residuals, sample_weight)


class RegularizedLinearTermIC(Term):
    """Linear term with regularization and information criterion for model selection."""

    allow_online_updates: bool = True

    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"],
        method: EstimationMethod | str = "lasso",
        fit_intercept: bool = True,
        forget: float = 0.0,
        ic: Literal["aic", "bic", "aicc", "hqc"] = "aic",
        is_regularized: bool = False,
        regularize_intercept: None | bool = None,
    ):
        self.features = features
        self.fit_intercept = fit_intercept
        self.method = method
        self.forget = forget
        self.is_regularized = is_regularized
        self.regularize_intercept = regularize_intercept
        self.ic = ic

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if not self._method._path_based_method:
            raise ValueError("Non-path-based methods are not supported for LinearTerm.")
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "RegularizedLinearTermIC":
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

        if self.is_regularized:
            self._is_regularized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            self._is_regularized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            self._is_regularized[0] = False

        forget_weights = init_forget_vector(self.forget, y.shape[0])

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
        self.coef_path_ = self._method.fit_beta_path(
            x_gram=self._g,
            y_gram=self._h,
            is_regularized=self._is_regularized,
        )

        self.n_observations = y.shape[0]
        self.n_nonzero_coef = np.count_nonzero(self.coef_path_, axis=1)

        residuals = y[:, None] - X_mat @ self.coef_path_.T
        rss = np.sum((residuals**2) * (sample_weight * forget_weights)[:, None], axis=0)
        rss = rss / np.mean(sample_weight * forget_weights)
        self.ic_values_ = InformationCriterion(
            n_observations=self.n_observations,
            n_parameters=self.n_nonzero_coef,
            criterion=self.ic,
        ).from_rss(rss)

        self.best_idx = np.argmin(self.ic_values_)
        self.coef_ = self.coef_path_[self.best_idx, :]
        self.rss = rss
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

        return X_mat @ self.coef_

    def update(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "LinearTerm":
        super().update(X, residuals, sample_weight)
