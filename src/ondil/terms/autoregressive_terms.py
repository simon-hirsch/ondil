import copy
from dataclasses import dataclass, replace

import numpy as np

from ..base import Distribution, EstimationMethod, Term
from ..design_matrix import add_intercept, make_lags
from ..methods import get_estimation_method


@dataclass(frozen=True)
class ARTermState:
    is_regularized: np.ndarray | None
    g: np.ndarray | None
    h: np.ndarray | None
    coef_: np.ndarray | None
    memory_fitted_values: np.ndarray | None
    memory_target_values: np.ndarray | None


class TimeSeriesTerm(Term):
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
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "TimeSeriesTerm":
        X_mat = self.make_design_matrix_in_sample_during_fit(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
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

    # TODO: Can this go to the base class?
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
    ) -> "TimeSeriesTerm":
        if self._state is None:
            raise ValueError("Term must be fitted before it can be updated.")

        X_mat = self.make_design_matrix_in_sample_during_update(
            X=X,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
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
            memory_fitted_values=np.concatenate(
                (self._state.memory_fitted_values, fitted_values),
            )[-np.max(self.lags) :],
            memory_target_values=np.concatenate(
                (self._state.memory_target_values, target_values),
            )[-np.max(self.lags) :],
        )
        return new_instance

    # These three need to be implemented by subclasses
    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    def make_design_matrix_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")


class JointEstimationTimeSeriesTerm(TimeSeriesTerm):
    allow_online_updates: bool = True

    def __init__(
        self,
        terms: list = None,
        method: EstimationMethod = "ols",
        fit_intercept: bool = True,
        forget: float = 0.0,
    ):
        self.terms = terms
        self.method = method
        self.forget = forget
        self.fit_intercept = fit_intercept
        self.is_regularized = False
        self.regularize_intercept = False

    def _prepare_term(self):
        super()._prepare_term()
        # I don't think this is necessary
        # Since we don't want to initialize the terms individually, but borrow
        # the methods to create design matrices from the individual terms.
        # self.terms = [term._prepare_term() for term in self.terms]
        self.lags = [
            np.max(term.lags) if hasattr(term, "lags") else 0 for term in self.terms
        ]
        return self

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
            )[:, int(term.fit_intercept) :]
            for term in self.terms
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
            )[:, int(term.fit_intercept) :]
            for term in self.terms
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
            )[:, int(term.fit_intercept) :]
            for term in self.terms
        ]
        X_mat = np.hstack(X_mats)
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "JointEstimationTimeSeriesTerm":
        new_instance = super().fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
        )
        new_instance.terms = self.terms
        for term in new_instance.terms:
            term._state = new_instance._state
        return new_instance

    def update(
        self,
        X: np.ndarray,  # for api compatibility; not used
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "TimeSeriesTerm":
        new_instance = super().update(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
            sample_weight=sample_weight,
        )
        new_instance.terms = self.terms
        for term in new_instance.terms:
            term._state = new_instance._state
        return new_instance


class AutoregressiveThetaTerm(TimeSeriesTerm):
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
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        X_mat = make_lags(y=fitted_values[:, self.param], lags=self.lags)
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
        lagged_value = np.concatenate((
            self._state.memory_fitted_values[:, self.param],
            fitted_values[:, self.param],
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )
        X_mat = make_lags(
            y=self._state.memory_fitted_values[:, self.param],
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[[-1], :]


class AutoregressiveTargetTerm(TimeSeriesTerm):
    """Autoregressive term using fitted values for lagged predictors."""

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        X_mat = make_lags(y=target_values, lags=self.lags)
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
        lagged_value = np.concatenate((
            self._state.memory_target_values,
            target_values,
        ))

        X_mat = make_lags(
            y=lagged_value,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )
        X_mat = make_lags(
            y=self._state.memory_target_values,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[[-1], :]


class AutoregressiveSquaredResidualTerm(TimeSeriesTerm):
    """Autoregressive term using fitted values for lagged predictors."""

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        squared_residuals = (target_values - distribution.mean(fitted_values)) ** 2
        X_mat = make_lags(y=squared_residuals, lags=self.lags)
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
        target = np.concatenate((
            self._state.memory_target_values,
            target_values,
        ))
        fv = np.concatenate((
            self._state.memory_fitted_values,
            fitted_values,
        ))
        squared_residuals = (target - distribution.mean(fv)) ** 2

        X_mat = make_lags(
            y=squared_residuals,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )

        squared_residuals = (
            self._state.memory_target_values
            - distribution.mean(self._state.memory_fitted_values)
        ) ** 2

        X_mat = make_lags(
            y=squared_residuals,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[[-1], :]


class LaggedResidualTerm(TimeSeriesTerm):
    r"""Term with lagged residuals as features.

    The lagged residuals are computed as :math:`y_t - \mu_t`, where :math:`\mu_t`
    are the fitted values from the distribution's mean function.
    """

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        residual = target_values - distribution.mean(fitted_values)
        X_mat = make_lags(y=residual, lags=self.lags)
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
        target = np.concatenate((
            self._state.memory_target_values,
            target_values,
        ))
        fv = np.concatenate((
            self._state.memory_fitted_values,
            fitted_values,
        ))
        residual = (target - distribution.mean(fv)) ** 2

        X_mat = make_lags(
            y=residual,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[-target_values.shape[0] :, :]

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
    ):
        if X.shape[0] > 1:
            raise ValueError(
                "Out-of-sample prediction for autoregressive terms can only be done "
                "one step ahead.",
            )

        residual = self._state.memory_target_values - distribution.mean(
            self._state.memory_fitted_values
        )
        X_mat = make_lags(
            y=residual,
            lags=self.lags,
        )
        if self.fit_intercept:
            X_mat = add_intercept(X_mat)
        return X_mat[[-1], :]
