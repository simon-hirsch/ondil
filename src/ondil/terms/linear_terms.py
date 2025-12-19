import copy
from dataclasses import dataclass, replace
from typing import Literal, Tuple

import numpy as np

from ..base import Distribution, EstimationMethod, Term
from ..base.terms import FeatureTransformation
from ..design_matrix import add_intercept, subset_array
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..incremental_statistics import calculate_statistics, update_statistics


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


class LinearFeatures(FeatureTransformation):
    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"] = "all",
    ):
        self.features = features

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ) -> np.ndarray:
        return subset_array(X, self.features)

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ) -> np.ndarray:
        return subset_array(X, self.features)

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        **kwargs,
    ) -> np.ndarray:
        return subset_array(X, self.features)


class _LinearBaseTerm(Term):
    """Base class for linear terms.

    This class should not be used directly.
    Provides _fit() and _update() methods for further use in child classes."""

    def __init__(
        self,
        method,
        forget: float = 0.0,
        fit_intercept: bool = True,
        is_regularized: np.ndarray | None = None,
        regularize_intercept: None | bool = None,
    ):
        self.method = method
        self.fit_intercept = fit_intercept
        self.is_regularized = is_regularized
        self.regularize_intercept = regularize_intercept
        self.forget = forget

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "LinearTerm":
        X_mat = self.make_design_matrix_in_sample_during_fit(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )
        self.find_multicollinear_columns(X_mat)
        self.find_zero_variance_columns(X_mat)

        X_mat = self.remove_problematic_columns(X_mat)

        if self.is_regularized:
            is_regularized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            is_regularized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            is_regularized[0] = False

        g = self._method.init_x_gram(
            X=X_mat,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        h = self._method.init_y_gram(
            X=X_mat,
            y=y,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        coef_ = self._method.fit_beta(
            x_gram=g,
            y_gram=h,
            is_regularized=is_regularized,
        )
        return g, h, coef_, is_regularized

    def _update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "LinearTerm":
        X_mat = self.make_design_matrix_in_sample_during_update(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )
        X_mat = self.remove_problematic_columns(X_mat)

        g = self._method.update_x_gram(
            gram=self._state.g,
            X=X_mat,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        h = self._method.update_y_gram(
            gram=self._state.h,
            X=X_mat,
            y=y,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        coef_ = self._method.fit_beta(
            x_gram=g,
            y_gram=h,
            is_regularized=self._state.is_regularized,
        )
        return g, h, coef_


class LinearTerm(_LinearBaseTerm):
    """Linear term for structured additive distributional regression."""

    allow_online_updates: bool = True

    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"] = "all",
        method: EstimationMethod = "ols",
        fit_intercept: bool = True,
        forget: float = 0.0,
        is_regularized: bool = False,
        regularize_intercept: None | bool = None,
    ):
        super().__init__(
            method=method,
            fit_intercept=fit_intercept,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
            forget=forget,
        )
        self.features = features

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if self._method._path_based_method:
            raise ValueError("Path-based methods are not supported for LinearTerm.")
        return self

    def make_design_matrix(
        self,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)
        return X_mat

    def make_design_matrix_in_sample_during_fit(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def make_design_matrix_in_sample_during_update(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def make_design_matrix_out_of_sample(self, X, **kwargs):
        return self.make_design_matrix(X)

    def predict_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        X_mat = self.make_design_matrix_out_of_sample(X=X)
        # X_mat = self.remove_problematic_columns(X_mat)
        return X_mat @ self.coef_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "LinearTerm":
        g, h, coef_, is_regularized = self._fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
            estimation_weight=estimation_weight,
        )
        self._state = LinearTermState(
            is_regularized=is_regularized,
            g=g,
            h=h,
            coef_=coef_,
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
    ) -> "LinearTerm":
        g, h, coef_ = self._update(
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
        )
        return new_instance


class InterceptTerm(LinearTerm):
    """Intercept term for structured additive distributional regression."""

    def __init__(self, forget=0.0):
        super().__init__(features=[], forget=forget, fit_intercept=True, method="ols")

    def make_design_matrix(
        self,
        X: np.ndarray | None,
        target_values: np.ndarray | None = None,
    ):
        if X is not None:
            n_samples = X.shape[0]
        elif target_values is not None:
            n_samples = target_values.shape[0]
        else:
            n_samples = 1

        return np.ones((n_samples, 1))


class _LinearPathModelSelectionIC(Term):
    """Linear term with regularization and information criterion for model selection."""

    allow_online_updates: bool = True

    def __init__(
        self,
        method: EstimationMethod | str = "lasso",
        fit_intercept: bool = True,
        forget: float = 0.0,
        ic: Literal["aic", "bic", "aicc", "hqc"] = "aic",
        is_regularized: np.ndarray | None = None,
        regularize_intercept: None | bool = None,
        weighted_regularization: bool = False,
    ):
        self.fit_intercept = fit_intercept
        self.method = method
        self.forget = forget
        self.is_regularized = is_regularized
        self.regularize_intercept = regularize_intercept
        self.weighted_regularization = weighted_regularization
        self.ic = ic

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if not self._method._path_based_method:
            raise ValueError("Non-path-based methods are not supported for LinearTerm.")
        return self

    @property
    def coef_(self) -> np.ndarray:
        """Get the coefficients of the linear term.

        Returns:
            np.ndarray: Coefficients of the linear term.
        """
        if not hasattr(self, "_state"):
            raise AttributeError("The term has not been fitted yet.")
        if hasattr(self, "remove"):
            if len(self.remove) > 0:
                j = len(self._state.coef_) + len(self.remove)
                mask = np.setdiff1d(np.arange(j), list(self.remove))
                beta = np.zeros(j)
                beta[mask] = self._state.coef_
                return beta
            else:
                return self._state.coef_

    def _calculate_design_variance(
        self,
        X: np.ndarray,
        sample_weight: np.ndarray,
    ):
        self.X_mat_mean = np.average(X, weights=sample_weight, axis=0)
        self.X_mat_diff_sq = (X - self.X_mat_mean) ** 2
        self.X_mat_var = np.average(self.X_mat_diff_sq, weights=sample_weight, axis=0)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "RegularizedLinearTermIC":
        X_mat = self.make_design_matrix_in_sample_during_fit(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )
        self.find_multicollinear_columns(X_mat)
        self.find_zero_variance_columns(X_mat)

        X_mat = self.remove_problematic_columns(X_mat)

        if self.is_regularized:
            is_regularized = self.is_regularized
        else:
            n_features = X_mat.shape[1]
            is_regularized = np.repeat(True, n_features)
        if self.fit_intercept and not self.regularize_intercept:
            is_regularized[0] = False

        forget_weights = init_forget_vector(self.forget, y.shape[0])

        self.X_mat_stats = calculate_statistics(
            X=X_mat,
            forget=self.forget,
            sample_weight=sample_weight,
        )

        if self.weighted_regularization:
            regularization_weights = 1 / self.X_mat_stats.var
        else:
            regularization_weights = None

        g = self._method.init_x_gram(
            X=X_mat,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        h = self._method.init_y_gram(
            X=X_mat,
            y=y,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        coef_path_ = self._method.fit_beta_path(
            x_gram=g,
            y_gram=h,
            is_regularized=is_regularized,
            regularization_weights=regularization_weights,
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

        return (
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
        )

    def _update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> Tuple:
        """Update the Term.

        Returns an updated copy and leaves the original unchanged.

        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Target array
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            LinearTerm: _description_
        """
        X_mat = self.make_design_matrix_in_sample_during_update(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )
        X_mat = self.remove_problematic_columns(X_mat)
        self.X_mat_stats = update_statistics(
            incremental_statistics=self.X_mat_stats,
            X=X_mat,
            sample_weight=sample_weight,
        )

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

        if self.weighted_regularization:
            regularization_weights = 1 / self.X_mat_stats.var
        else:
            regularization_weights = None

        coef_path_ = self._method.fit_beta_path(
            x_gram=g,
            y_gram=h,
            is_regularized=self._state.is_regularized,
            regularization_weights=regularization_weights,
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

        return (
            g,
            h,
            coef_,
            coef_path_,
            best_idx,
            ic_values_,
            rss,
            n_observations,
            n_nonzero_coef,
        )


class RegularizedLinearTermIC(_LinearPathModelSelectionIC):
    """Linear term with regularization and information criterion for model selection."""

    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"] = "all",
        method: EstimationMethod | str = "lasso",
        fit_intercept: bool = True,
        forget: float = 0.0,
        ic: Literal["aic", "bic", "aicc", "hqc"] = "aic",
        is_regularized: np.ndarray | None = None,
        regularize_intercept: None | bool = None,
        weighted_regularization: bool = False,
    ):
        super().__init__(
            method=method,
            fit_intercept=fit_intercept,
            forget=forget,
            ic=ic,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
            weighted_regularization=weighted_regularization,
        )
        self.features = features

    def make_design_matrix(
        self,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.fit_intercept:
            X_mat = add_intercept(subset_array(X, self.features))
        else:
            X_mat = subset_array(X, self.features)
        return X_mat

    def make_design_matrix_in_sample_during_fit(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def make_design_matrix_in_sample_during_update(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def make_design_matrix_out_of_sample(self, X, **kwargs):
        return self.make_design_matrix(X)

    def predict_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        X_mat = self.make_design_matrix(X=X)
        # X_mat = self.remove_problematic_columns(X_mat)
        return X_mat @ self.coef_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "RegularizedLinearTermIC":
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

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "RegularizedLinearTermIC":
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
        )
        return new_instance
