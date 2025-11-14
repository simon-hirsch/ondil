import copy
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..base import EstimationMethod, Term
from ..design_matrix import add_intercept, subset_array
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method


@dataclass(frozen=True)
class LinearTermState:
    is_regularized: np.ndarray | None
    g: np.ndarray | None
    h: np.ndarray | None
    coef_: np.ndarray | None


@dataclass(frozen=True)
class RegularizedLinearTermICState:
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


class LinearTerm(Term):
    """Linear term for structured additive distributional regression."""

    allow_online_updates: bool = True

    features: np.ndarray | list[int] | Literal["all"]
    method: EstimationMethod | str = "ols"
    fit_intercept: bool = True
    forget: float = 0.0
    is_regularized: bool = False
    regularize_intercept: None | bool = None

    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"] = "all",
        method: EstimationMethod = "ols",
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

        self._state = LinearTermState(
            is_regularized=is_regularized,
            g=g,
            h=h,
            coef_=coef_,
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

        return X_mat @ self._state.coef_

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "LinearTerm":
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

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
        # Create a new instance with updated values
        new_instance = copy.copy(self)
        new_instance._state = replace(
            self._state,
            g=g,
            h=h,
            coef_=coef_,
        )
        return new_instance


class RegularizedLinearTermIC(Term):
    """Linear term with regularization and information criterion for model selection."""

    allow_online_updates: bool = True

    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"] = "all",
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
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ) -> "RegularizedLinearTermIC":
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

        if self.is_regularized:
            is_regularized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            is_regularized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            is_regularized[0] = False

        forget_weights = init_forget_vector(self.forget, y.shape[0])

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
        coef_path_ = self._method.fit_beta_path(
            x_gram=g,
            y_gram=h,
            is_regularized=is_regularized,
        )

        n_observations = y.shape[0]
        n_nonzero_coef = np.count_nonzero(coef_path_, axis=1)

        residuals = y[:, None] - X_mat @ coef_path_.T
        rss = np.sum((residuals**2) * (sample_weight * forget_weights)[:, None], axis=0)
        rss = rss / np.mean(sample_weight * forget_weights)
        ic_values_ = InformationCriterion(
            n_observations=n_observations,
            n_parameters=n_nonzero_coef,
            criterion=self.ic,
        ).from_rss(rss)

        best_idx = np.argmin(ic_values_)
        coef_ = coef_path_[best_idx, :]

        self._state = RegularizedLinearTermICState(
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

        return X_mat @ self._state.coef_

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ) -> "RegularizedLinearTermIC":
        """Update the Term.

        Returns an updated copy and leaves the original unchanged.

        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Target array
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            LinearTerm: _description_
        """
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)

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
        coef_path_ = self._method.fit_beta_path(
            x_gram=g,
            y_gram=h,
            is_regularized=self._state.is_regularized,
        )
        n_observations = self._state.n_observations + y.shape[0]
        n_nonzero_coef = np.count_nonzero(coef_path_, axis=1)

        forget_weights = init_forget_vector(self.forget, y.shape[0])
        residuals = y[:, None] - X_mat @ coef_path_.T

        # TODO: Verify if this is correct
        rss_new = np.sum(
            (residuals**2) * (sample_weight * forget_weights)[:, None], axis=0
        )
        rss_new = rss_new / np.mean(sample_weight * forget_weights)
        rss = (1 - self.forget) ** y.shape[0] * self._state.rss + rss_new
        ic_values_ = InformationCriterion(
            n_observations=n_observations,
            n_parameters=n_nonzero_coef,
            criterion=self.ic,
        ).from_rss(rss)
        best_idx = np.argmin(ic_values_)
        coef_ = coef_path_[best_idx, :]

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
        )
        return new_instance
