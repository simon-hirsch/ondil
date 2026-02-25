from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..base import Distribution
from ..base.terms import FeatureTransformation
from ..design_matrix import make_lags, subset_array

if TYPE_CHECKING:
    from .time_series import ARTermState


class LinearFeature(FeatureTransformation):
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


class TimeSeriesFeature(FeatureTransformation):
    def __init__(self, lags: np.ndarray | list[int] | int = 1):
        self.lags = lags

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        raise NotImplementedError("Not implemented")

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        state: ARTermState,
        **kwargs,
    ):
        raise NotImplementedError("Not implemented")

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
        **kwargs,
    ):
        raise NotImplementedError("Not implemented")


class LaggedTheta(TimeSeriesFeature):
    """Autoregressive term using target values for lagged predictors."""

    def __init__(
        self,
        param: int,
        lags: np.ndarray | list[int] | int = 1,
        on_link_space: bool = False,
    ):
        super().__init__(
            lags=lags,
        )
        self.param = param
        self.on_link_space = on_link_space

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        if self.on_link_space:
            values = distribution.link_function(
                fitted_values[:, self.param], param=self.param
            )
        else:
            values = fitted_values[:, self.param]
        return make_lags(y=values, lags=self.lags)

    def make_design_matrix_in_sample_during_update(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        state: ARTermState,
        **kwargs,
    ):
        if self.on_link_space:
            values = distribution.link_function(
                fitted_values[:, self.param], param=self.param
            )
        else:
            values = fitted_values[:, self.param]

        lagged_value = np.concatenate((
            state.memory_fitted_values[:, self.param],
            values,
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )
        X_mat = make_lags(
            y=state.memory_fitted_values[:, self.param],
            lags=self.lags,
        )
        return X_mat[[-1], :]


class LaggedTarget(TimeSeriesFeature):
    """Autoregressive term using fitted values for lagged predictors."""

    def make_design_matrix_in_sample_during_fit(
        self,
        target_values: np.ndarray,
        **kwargs,
    ):
        X_mat = make_lags(y=target_values, lags=self.lags)
        return X_mat

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        state: ARTermState,
    ):
        lagged_value = np.concatenate((
            state.memory_target_values,
            target_values,
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )
        X_mat = make_lags(
            y=state.memory_target_values,
            lags=self.lags,
        )
        return X_mat[[-1], :]


class LaggedSquaredResidual(TimeSeriesFeature):
    """Autoregressive term using fitted values for lagged predictors."""

    def __init__(
        self,
        lags: np.ndarray | list[int] | int = 1,
        standardize: bool = False,
    ):
        super().__init__(
            lags=lags,
        )
        self.standardize = standardize

    def make_design_matrix_in_sample_during_fit(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        residuals = target_values - distribution.mean(fitted_values)
        if self.standardize:
            residuals = residuals / distribution.variance(fitted_values) ** 0.5
        squared_residuals = residuals**2

        X_mat = make_lags(y=squared_residuals, lags=self.lags)
        return X_mat

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        state: ARTermState,
    ):
        target = np.concatenate((
            state.memory_target_values,
            target_values,
        ))
        fv = np.concatenate((
            state.memory_fitted_values,
            fitted_values,
        ))
        residuals = target - distribution.mean(fv)
        if self.standardize:
            residuals = residuals / distribution.variance(fv) ** 0.5
        squared_residuals = residuals**2

        X_mat = make_lags(
            y=squared_residuals,
            lags=self.lags,
        )
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )

        residuals = state.memory_target_values - distribution.mean(
            state.memory_fitted_values
        )
        if self.standardize:
            residuals = (
                residuals / distribution.variance(state.memory_fitted_values) ** 0.5
            )
        squared_residuals = residuals**2

        X_mat = make_lags(
            y=squared_residuals,
            lags=self.lags,
        )
        return X_mat[[-1], :]


class LaggedAbsoluteResidual(TimeSeriesFeature):
    """Autoregressive term using fitted values for lagged predictors."""

    def __init__(
        self,
        lags: np.ndarray | list[int] | int = 1,
        standardize: bool = False,
    ):
        super().__init__(
            lags=lags,
        )
        self.standardize = standardize

    def make_design_matrix_in_sample_during_fit(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        residuals = target_values - distribution.mean(fitted_values)
        if self.standardize:
            residuals = residuals / distribution.variance(fitted_values) ** 0.5
        abs_residuals = np.abs(residuals)
        X_mat = make_lags(y=abs_residuals, lags=self.lags)
        return X_mat

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        state: ARTermState,
    ):
        target = np.concatenate((
            state.memory_target_values,
            target_values,
        ))
        fv = np.concatenate((
            state.memory_fitted_values,
            fitted_values,
        ))
        residuals = target - distribution.mean(fv)
        if self.standardize:
            residuals = residuals / distribution.variance(fv) ** 0.5
        abs_residuals = np.abs(residuals)

        X_mat = make_lags(
            y=abs_residuals,
            lags=self.lags,
        )
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )

        residuals = state.memory_target_values - distribution.mean(
            state.memory_fitted_values
        )
        if self.standardize:
            residuals = (
                residuals / distribution.variance(state.memory_fitted_values) ** 0.5
            )
        abs_residuals = np.abs(residuals)

        X_mat = make_lags(
            y=abs_residuals,
            lags=self.lags,
        )
        return X_mat[[-1], :]


class LaggedResidual(TimeSeriesFeature):
    r"""Term with lagged residuals as features.

    The lagged residuals are computed as :math:`y_t - \mu_t`, where :math:`\mu_t`
    are the fitted values from the distribution's mean function.
    """

    def make_design_matrix_in_sample_during_fit(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        residual = target_values - distribution.mean(fitted_values)
        X_mat = make_lags(y=residual, lags=self.lags)
        return X_mat

    def make_design_matrix_in_sample_during_update(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        state: ARTermState,
        **kwargs,
    ):
        target = np.concatenate((
            state.memory_target_values,
            target_values,
        ))
        fv = np.concatenate((
            state.memory_fitted_values,
            fitted_values,
        ))
        residual = target - distribution.mean(fv)

        X_mat = make_lags(
            y=residual,
            lags=self.lags,
        )
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )

        residual = state.memory_target_values - distribution.mean(
            state.memory_fitted_values
        )
        X_mat = make_lags(
            y=residual,
            lags=self.lags,
        )
        return X_mat[[-1], :]
