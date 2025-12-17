import copy
from dataclasses import dataclass, replace

import numpy as np

from ..base import Distribution, EstimationMethod, Term
from ..base.terms import FeatureTransformation
from ..design_matrix import add_intercept, make_lags
from ..methods import get_estimation_method
from ..terms.linear_terms import _LinearBaseTerm, _LinearPathModelSelectionIC


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


class JointEstimationTimeSeriesTerm(_LinearBaseTerm):
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
        regularize_intercept: None | bool = None,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            method=method,
            forget=forget,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
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
    ) -> "JointEstimationTimeSeriesTerm":
        g, h, coef_, is_regularized = self._fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
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
        X_mat = self.remove_zero_variance_columns(X_mat)
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
        X_mat = self.remove_zero_variance_columns(X_mat)
        return X_mat @ self._state.coef_

    def predict_out_of_sample(
        self,
        X: np.ndarray,  # for api compatibility; not used
        distribution: Distribution,
    ):
        X_mat = self.make_design_matrix_out_of_sample(X=X, distribution=distribution)
        X_mat = self.remove_zero_variance_columns(X_mat)
        return X_mat @ self._state.coef_

    def update(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "JointEstimationTimeSeriesTerm":
        if self._state is None:
            raise ValueError("Term must be fitted before it can be updated.")

        g, h, coef_ = self._update(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
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


class RegularizedJointEstimationTimeSeriesTerm(
    _LinearPathModelSelectionIC,
    JointEstimationTimeSeriesTerm,
    # Left to right inheritance to ensure correct MRO
    # we get the _fit and the _update methods from
    # _LinearPathModelSelectionIC
):
    """Regularized autoregressive term for time series modeling with model selection.

    This class extends the JointEstimationTimeSeriesTerm to include regularization
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
        is_regularized: bool = True,
        regularize_intercept: None | bool = None,
        ic: str = "aic",
    ):
        super().__init__(
            effects=effects,
            method=method,
            fit_intercept=fit_intercept,
            forget=forget,
            is_regularized=is_regularized,
            regularize_intercept=regularize_intercept,
            ic=ic,
        )


# This defines what a time series feature should implement
# But maybe not all features are needed
# The important part is that the feature has access to the
# - fitted_values
# - target_values to create lagged versions
# - to the distribution (to compute residuals etc based on hte mean/median)
# - to the state (to access memory of previous fitted/target values)


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
        lags: np.ndarray | list[int] | int = 1,
        param: int = 0,
    ):
        super().__init__(
            lags=lags,
        )
        self.param = param

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        **kwargs,
    ):
        X_mat = make_lags(y=fitted_values[:, self.param], lags=self.lags)
        return X_mat

    def make_design_matrix_in_sample_during_update(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        state: ARTermState,
        **kwargs,
    ):
        lagged_value = np.concatenate((
            state.memory_fitted_values[:, self.param],
            fitted_values[:, self.param],
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

    def make_design_matrix_in_sample_during_fit(
        self,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        squared_residuals = (target_values - distribution.mean(fitted_values)) ** 2
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
        squared_residuals = (target - distribution.mean(fv)) ** 2

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

        squared_residuals = (
            state.memory_target_values - distribution.mean(state.memory_fitted_values)
        ) ** 2

        X_mat = make_lags(
            y=squared_residuals,
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
        residual = (target - distribution.mean(fv)) ** 2

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
