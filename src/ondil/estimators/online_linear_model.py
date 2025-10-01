import numbers
import warnings
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ..base import EstimationMethod, OndilEstimatorMixin
from ..design_matrix import add_intercept
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..scaler import OnlineScaler
from ..utils import calculate_effective_training_length


class OnlineLinearModel(OndilEstimatorMixin, RegressorMixin, BaseEstimator):
    "Simple Online Linear Regression for the expected value."

    _parameter_constraints = {
        "forget": [Interval(numbers.Real, 0.0, 1.0, closed="left")],
        "fit_intercept": [bool],
        "scale_inputs": [bool, np.ndarray],
        "regularize_intercept": [bool],
        "method": [EstimationMethod, str],
        "ic": [StrOptions({"aic", "bic", "hqc", "max"})],
    }

    def __init__(
        self,
        forget: float = 0.0,
        scale_inputs: bool | np.ndarray = True,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        method: EstimationMethod | str = "ols",
        ic: Literal["aic", "bic", "hqc", "max"] = "bic",
    ):
        """The basic linear model for many different estimation techniques.

        Args:
            forget (float, optional): Exponential discounting of old observations. Defaults to 0.
            scale_inputs (bool, optional): Whether to scale the $X$ matrix. Defaults to True.
            fit_intercept (bool, optional): Whether to add an intercept in the estimation. Defaults to True.
            regularize_intercept (bool, optional): Whether to regularize the intercept. Defaults to False.
            method (EstimationMethod | str, optional): The estimation method. Can be a string or `EstimationMethod` class. Defaults to "ols".
            ic (Literal["aic", "bic", "hqc", "max"], optional): The information criteria for model selection. Defaults to "bic".
        Raises:
            ValueError: Will raise if you try to regularize the intercept, but not fit it.
        """

        self.forget = forget
        self.method = method
        self.fit_intercept = fit_intercept
        self.scale_inputs = scale_inputs
        self.regularize_intercept = regularize_intercept
        self.ic = ic

    @property
    def beta(self):
        check_is_fitted(self)
        return self.coef_

    @property
    def beta_path(self):
        check_is_fitted(self)
        return self.coef_path_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

    def _prepare_fit(self):
        self._method = get_estimation_method(self.method)
        self._scaler = OnlineScaler(forget=self.forget, to_scale=self.scale_inputs)

        # Raise on inputs that does not make sense
        if not self._method._path_based_method and self.regularize_intercept:
            warnings.warn(
                "Note that you have passed a non-regularized estimation method but want to regularize the intercept. Are you sure?"
            )
        if self.regularize_intercept and not self.fit_intercept:
            raise ValueError(
                "You want to regularize the intercept but not fit an intercept."
            )

    def get_design_matrix(self, X: np.ndarray):
        if self.fit_intercept:
            design = add_intercept(X)
        else:
            design = np.copy(X)
        return design

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "OnlineLinearModel":
        """Initial fit of the online regression model.

        Args:
            X (np.ndarray): The design matrix $X$.
            y (np.ndarray): The response vector $y$.
            sample_weight (Optional[np.ndarray], optional): The sample weights. Defaults to None.
        """

        self._prepare_fit()
        X, y = validate_data(self, X=X, y=y, reset=True, dtype=[np.float64, np.float32])
        _ = type_of_target(y, raise_unknown=True)
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)

        self.n_observations_ = np.sum(sample_weight)
        self.n_features_ = X.shape[1] + int(self.fit_intercept)

        if self.n_observations_ <= self.n_features_ and self.method == "ols":
            raise ValueError(
                f"You have tried to fit using n_samples={self.n_observations_} and n_features={self.n_features_}. "
                "Since we need calculate the inverse Gram matrix, we need at least as many observations as features for OLS-based methods"
            )
        if self.n_observations_ == 1:
            raise ValueError(
                "You have tried to fit using only one sample. This is not supported for online linear regression."
            )

        self.n_training_ = calculate_effective_training_length(
            self.forget, self.n_observations_
        )

        self._scaler.fit(X, sample_weight=sample_weight)
        X_scaled = self._scaler.transform(X)
        X_scaled = self.get_design_matrix(X=X_scaled)

        self._is_regularized = np.repeat(True, self.n_features_)
        if self.fit_intercept:
            self._is_regularized[0] = self.regularize_intercept

        self._x_gram = self._method.init_x_gram(
            X_scaled, weights=sample_weight, forget=self.forget
        )
        self._y_gram = self._method.init_y_gram(
            X_scaled, y, weights=sample_weight, forget=self.forget
        )

        if self._method._path_based_method:
            self.coef_path_ = self._method.fit_beta_path(
                x_gram=self._x_gram,
                y_gram=self._y_gram,
                is_regularized=self._is_regularized,
            )
            residuals = np.expand_dims(sample_weight, -1) * (
                np.expand_dims(y, -1) - X_scaled @ self.coef_path_.T
            )
            self._rss = np.sum(residuals**2, axis=0)
            n_params = np.sum(~np.isclose(self.coef_path_, 0), axis=1)
            ic = InformationCriterion(
                n_observations=self.n_training_,
                n_parameters=n_params,
                criterion=self.ic,
            ).from_rss(rss=self._rss)
            best_ic = np.argmin(ic)
            self.coef_ = self.coef_path_[best_ic, :]
        else:
            self.coef_ = self._method.fit_beta(
                self._x_gram, self._y_gram, self._is_regularized
            )

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Update the regression model.

        Args:
            X (np.ndarray): The new row of the design matrix $X$. Needs to be of shape 1 x n_features or n_obs_new x n_features.
            y (np.ndarray): The new observation of $y$. Needs to be the same shape as `X` or a single observation.
            sample_weight (Optional[np.ndarray], optional): The weight for the new observations. `None` implies all observations have weight 1. Defaults to None.
        """

        X, y = validate_data(
            self, X=X, y=y, reset=False, dtype=[np.float64, np.float32]
        )
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)
        _ = type_of_target(y, raise_unknown=True)

        self.n_observations_ += sample_weight.sum()
        self.n_training_ = calculate_effective_training_length(
            self.forget, self.n_observations_
        )

        self._scaler.update(X, sample_weight=sample_weight)
        X_scaled = self._scaler.transform(X)
        X_scaled = self.get_design_matrix(X=X_scaled)

        self._x_gram = self._method.update_x_gram(
            self._x_gram, X_scaled, forget=self.forget, weights=sample_weight
        )
        self._y_gram = self._method.update_y_gram(
            self._y_gram, X_scaled, y, forget=self.forget, weights=sample_weight
        )
        if self._method._path_based_method:
            self.coef_path_ = self._method.update_beta_path(
                x_gram=self._x_gram,
                y_gram=self._y_gram,
                beta_path=self.coef_path_,
                is_regularized=self._is_regularized,
            )

            residuals = np.expand_dims(y, -1) - X_scaled @ self.coef_path_.T
            self._rss = (1 - self.forget) * self._rss + np.sum(residuals**2, axis=0)
            n_params = np.sum(~np.isclose(self.coef_path_, 0), axis=1)
            ic = InformationCriterion(
                n_observations=self.n_training_,
                n_parameters=n_params,
                criterion=self.ic,
            ).from_rss(rss=self._rss)
            best_ic = np.argmin(ic)
            self.coef_ = self.coef_path_[best_ic, :]
        else:
            self.coef_ = self._method.update_beta(
                x_gram=self._x_gram,
                y_gram=self._y_gram,
                beta=self.coef_,
                is_regularized=self._is_regularized,
            )

        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the coefficient of determination $R^2$.

        Args:
            X (np.ndarray): The design matrix $X$.
            y (np.ndarray): The response vector $y$.

        Returns:
            float: The coefficient of determination $R^2$.
        """
        check_is_fitted(self)
        X, y = validate_data(
            self, X=X, y=y, reset=False, dtype=[np.float64, np.float32]
        )

        prediction = self.predict(X)
        ss_residuals = np.sum((y - prediction) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_residuals / ss_total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the optimal IC selection.

        Args:
            X (np.ndarray): The design matrix $X$.

        Returns:
            np.ndarray: The predictions for the optimal IC.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, dtype=[np.float64, np.float32])

        design = self._scaler.transform(X)
        design = self.get_design_matrix(X=design)
        prediction = design @ self.coef_.T
        return prediction

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        """Predict the full regularization path.

        Args:
            X (np.ndarray): The design matrix $X$.

        Returns:
            np.ndarray: The predictions for the full path.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, dtype=[np.float64, np.float32])

        design = self._scaler.transform(X)
        design = self.get_design_matrix(X=design)
        prediction = design @ self.coef_path_.T
        return prediction
