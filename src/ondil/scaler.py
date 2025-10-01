import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from .base.estimator import OndilEstimatorMixin
from .gram import init_forget_vector


class OnlineScaler(OndilEstimatorMixin, TransformerMixin, BaseEstimator):

    _parameter_constraints = {
        "forget": [Interval(numbers.Real, 0.0, 1.0, closed="left")],
        "to_scale": [bool, np.ndarray],
    }

    def __init__(
        self,
        forget: float = 0.0,
        to_scale: bool | np.ndarray = True,
    ):
        """The online scaler allows for incremental updating and scaling of matrices.

        Args:
            forget (float, optional): The forget factor. Older observations will be exponentially discounted. Defaults to 0.0.
            to_scale (bool | np.ndarray, optional): The variables to scale.
                `True` implies all variables will be scaled.
                `False` implies no variables will be scaled.
                An `np.ndarray` of type `bool` or `int` implies that the columns `X[:, to_scale]` will be scaled, all other columns will not be scaled.
                Defaults to True.
        """
        self.forget = forget
        self.to_scale = to_scale

    def _prepare_estimator(self, X: np.ndarray):
        """Add derived attributes to estimator"""
        if isinstance(self.to_scale, np.ndarray):
            self._selection = self.to_scale
            self._do_scale = True
        elif isinstance(self.to_scale, bool):
            if self.to_scale:
                self._selection = np.arange(X.shape[1])
                self._do_scale = True
            else:
                self._selection = False
                self._do_scale = False

        # Variables
        self.mean_ = 0
        self.var_ = 0
        self._M = 0
        self._cumulative_w = 0  # Track cumulative weights for exponential forgetting

    @property
    def std_(self) -> float | np.ndarray:
        """Standard deviation of the scaled variables."""
        check_is_fitted(self, ["mean_", "var_"])
        if self._do_scale:
            return np.sqrt(self.var_)
        else:
            return 1.0

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,
        y: None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "OnlineScaler":
        """Fit the OnlineScaler() Object for the first time.

        Args:
            X (np.ndarray): Matrix of covariates X.
            y (None, optional): Not used, present for compatibility with sklearn API. Defaults to None.
            sample_weight (np.ndarray, optional): Weights for each sample. Defaults to None (uniform weights).
        """

        X = validate_data(
            self,
            X=X,
            y=None,
            reset=True,
            ensure_min_samples=2,
            dtype=[np.float64, np.float32],
        )
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)
        self._prepare_estimator(X)
        self.n_observations_ = sample_weight.sum()

        if self._do_scale:
            forget_vector = init_forget_vector(self.forget, X.shape[0])
            effective_weights = sample_weight * forget_vector

            self._cumulative_w = np.sum(
                effective_weights
            )  # Initialize cumulative weight
            self.mean_ = np.average(
                X[:, self._selection], weights=effective_weights, axis=0
            )

            # Calculate the variance of each column of x_init and assing it to self.var_
            diff_sq = (X[:, self._selection] - self.mean_) ** 2
            self.var_ = np.average(diff_sq, weights=effective_weights, axis=0)
            self._M = self.var_ * self._cumulative_w

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def update(self, X: np.ndarray, y=None, sample_weight: np.ndarray = None):
        """Update the `OnlineScaler()` for new rows of X.

        Args:
            X (np.ndarray): New data for X.
            y (None, optional): Not used, present for compatibility with sklearn API. Defaults to None.
            sample_weight (np.ndarray, optional): Weights for each sample. Defaults to None (uniform weights).
        """
        check_is_fitted(self, ["mean_", "var_"])
        X = validate_data(
            self,
            X=X,
            y=None,
            reset=False,
            ensure_min_samples=1,
            dtype=[np.float64, np.float32],
        )
        sample_weight = _check_sample_weight(X=X, sample_weight=sample_weight)
        self.n_observations_ += sample_weight.sum()

        # Loop over all rows of new X
        if self._do_scale:
            for i in range(X.shape[0]):
                # Effective weight for the old state
                eff_old_w = self._cumulative_w * (1 - self.forget)
                self._cumulative_w = eff_old_w + sample_weight[i]
                diff_old = X[i, self._selection] - self.mean_

                # Update mean
                self.mean_ = (
                    self.mean_ * eff_old_w + X[i, self._selection] * sample_weight[i]
                ) / self._cumulative_w

                diff_new = X[i, self._selection] - self.mean_

                # Update M (sum of squared deviations)
                self._M = (
                    self._M * (1 - self.forget) + sample_weight[i] * diff_old * diff_new
                )

                # Update variance
                self.var_ = self._M / self._cumulative_w

        else:
            pass

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a mean-std scaled matrix.

        Args:
            X (np.ndarray): X matrix for covariates.

        Returns:
            np.ndarray: Scaled X matrix.
        """
        check_is_fitted(self, ["mean_", "var_"])
        X = validate_data(
            self,
            X=X,
            y=None,
            reset=False,
            ensure_min_samples=1,
            dtype=[np.float64, np.float32],
        )

        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = (X[:, self._selection] - self.mean_) / np.sqrt(
                self.var_
            )
            return out
        else:
            return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Back-transform a scaled X matrix to the original domain.

        Args:
            X (np.ndarray): Scaled X matrix.

        Returns:
            np.ndarray: Scaled back to the original scale.
        """
        check_is_fitted(self, ["mean_", "var_"])
        X = validate_data(
            self,
            X=X,
            reset=False,
            ensure_min_samples=1,
            dtype=[np.float64, np.float32],
        )

        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = (
                X[:, self._selection] * np.sqrt(self.var_) + self.mean_
            )
            return out
        else:
            return X
