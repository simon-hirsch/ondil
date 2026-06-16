from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..base import Distribution
from ..base.terms import FeatureTransformation
from ..design_matrix import make_lags, subset_array

if TYPE_CHECKING:
    from .time_series import ARTermState


class LinearFeature(FeatureTransformation):
    """Feature transformation that selects linear features from the input data.

    This class allows selecting specific columns or all columns from the input
    feature matrix X for use in linear terms.
    """

    def __init__(
        self,
        features: np.ndarray | list[int] | Literal["all"] = "all",
    ):
        """Initialize the LinearFeature transformation.

        Parameters
        ----------
        features : np.ndarray | list[int] | Literal["all"], default="all"
            Indices of features to select. If "all", selects all features.
        """
        self.features = features

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ) -> np.ndarray:
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        distribution : Distribution
            The distribution object (unused for linear features).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Selected feature columns.
        """
        return subset_array(X, self.features)

    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ) -> np.ndarray:
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        distribution : Distribution
            The distribution object (unused for linear features).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Selected feature columns.
        """
        return subset_array(X, self.features)

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        **kwargs,
    ) -> np.ndarray:
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        distribution : Distribution
            The distribution object (unused for linear features).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Selected feature columns.
        """
        return subset_array(X, self.features)


class TimeSeriesFeature(FeatureTransformation):
    """Base class for time series feature transformations.

    This class provides a template for creating lagged features from time series data.
    Subclasses should implement the specific logic for creating design matrices.
    """

    def __init__(self, lags: np.ndarray | list[int] | int = 1):
        """Initialize the TimeSeriesFeature transformation.

        Parameters
        ----------
        lags : np.ndarray | list[int] | int, default=1
            Lag orders to include in the feature transformation.
        """
        self.lags = lags

    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        **kwargs,
    ):
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        fitted_values : np.ndarray
            Fitted values from the model.
        target_values : np.ndarray
            Target values.
        distribution : Distribution
            The distribution object.
        **kwargs
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
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
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        fitted_values : np.ndarray
            Fitted values from the model.
        target_values : np.ndarray
            Target values.
        distribution : Distribution
            The distribution object.
        state : ARTermState
            State object containing memory for time series.
        **kwargs
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Not implemented")

    def make_design_matrix_out_of_sample(
        self,
        X,
        distribution: Distribution,
        state: ARTermState,
        **kwargs,
    ):
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        distribution : Distribution
            The distribution object.
        state : ARTermState
            State object containing memory for time series.
        **kwargs
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Not implemented")


class LaggedTheta(TimeSeriesFeature):
    """Autoregressive term using fitted parameter values for lagged predictors.

    This feature creates lagged versions of the fitted distributional parameters
    transformed to the link space if specified.
    """

    def __init__(
        self,
        param: int,
        lags: np.ndarray | list[int] | int = 1,
        on_link_space: bool = False,
    ):
        """Initialize the LaggedTheta feature.

        Parameters
        ----------
        param : int
            Index of the distributional parameter to lag.
        lags : np.ndarray | list[int] | int, default=1
            Lag orders to include.
        on_link_space : bool, default=False
            Whether to apply the link function before lagging.
        """
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
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        fitted_values : np.ndarray
            Fitted parameter values.
        distribution : Distribution
            The distribution object for link function.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged fitted parameter values.
        """
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
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        fitted_values : np.ndarray
            Current fitted parameter values.
        target_values : np.ndarray
            Target values (unused).
        distribution : Distribution
            The distribution object for link function.
        state : ARTermState
            State containing memory of previous fitted values.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged fitted parameter values including memory.
        """
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
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        distribution : Distribution
            The distribution object (unused).
        state : ARTermState
            State containing memory of fitted values.

        Returns
        -------
        np.ndarray
            Lagged fitted parameter values for prediction.

        Raises
        ------
        ValueError
            If prediction is not one step ahead.
        """
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
    """Autoregressive term using target values for lagged predictors.

    This feature creates lagged versions of the target values for use in
    autoregressive modeling.
    """

    def make_design_matrix_in_sample_during_fit(
        self,
        target_values: np.ndarray,
        **kwargs,
    ):
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        target_values : np.ndarray
            Target values to lag.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged target values.
        """
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
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        fitted_values : np.ndarray
            Fitted values (unused).
        target_values : np.ndarray
            Current target values.
        distribution : Distribution
            The distribution object (unused).
        state : ARTermState
            State containing memory of previous target values.

        Returns
        -------
        np.ndarray
            Lagged target values including memory.
        """
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
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        distribution : Distribution
            The distribution object (unused).
        state : ARTermState
            State containing memory of target values.

        Returns
        -------
        np.ndarray
            Lagged target values for prediction.

        Raises
        ------
        ValueError
            If prediction is not one step ahead.
        """
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
    """Autoregressive term using squared residuals for lagged predictors.

    This feature creates lagged versions of the squared residuals
    (optionally standardized) for use in volatility modeling.
    """

    def __init__(
        self,
        lags: np.ndarray | list[int] | int = 1,
        standardize: bool = False,
    ):
        """Initialize the LaggedSquaredResidual feature.

        Parameters
        ----------
        lags : np.ndarray | list[int] | int, default=1
            Lag orders to include.
        standardize : bool, default=False
            Whether to standardize residuals by the variance.
        """
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
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        fitted_values : np.ndarray
            Fitted parameter values.
        target_values : np.ndarray
            Target values.
        distribution : Distribution
            The distribution object for computing residuals.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged squared residuals.
        """
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
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        fitted_values : np.ndarray
            Current fitted parameter values.
        target_values : np.ndarray
            Current target values.
        distribution : Distribution
            The distribution object for computing residuals.
        state : ARTermState
            State containing memory of previous values.

        Returns
        -------
        np.ndarray
            Lagged squared residuals including memory.
        """
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
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        distribution : Distribution
            The distribution object for computing residuals.
        state : ARTermState
            State containing memory of fitted and target values.

        Returns
        -------
        np.ndarray
            Lagged squared residuals for prediction.

        Raises
        ------
        ValueError
            If prediction is not one step ahead.
        """
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
    """Autoregressive term using absolute residuals for lagged predictors.

    This feature creates lagged versions of the absolute residuals
    (optionally standardized) for use in volatility modeling.
    """

    def __init__(
        self,
        lags: np.ndarray | list[int] | int = 1,
        standardize: bool = False,
    ):
        """Initialize the LaggedAbsoluteResidual feature.

        Parameters
        ----------
        lags : np.ndarray | list[int] | int, default=1
            Lag orders to include.
        standardize : bool, default=False
            Whether to standardize residuals by the variance.
        """
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
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        fitted_values : np.ndarray
            Fitted parameter values.
        target_values : np.ndarray
            Target values.
        distribution : Distribution
            The distribution object for computing residuals.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged absolute residuals.
        """
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
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        fitted_values : np.ndarray
            Current fitted parameter values.
        target_values : np.ndarray
            Current target values.
        distribution : Distribution
            The distribution object for computing residuals.
        state : ARTermState
            State containing memory of previous values.

        Returns
        -------
        np.ndarray
            Lagged absolute residuals including memory.
        """
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
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        distribution : Distribution
            The distribution object for computing residuals.
        state : ARTermState
            State containing memory of fitted and target values.

        Returns
        -------
        np.ndarray
            Lagged absolute residuals for prediction.

        Raises
        ------
        ValueError
            If prediction is not one step ahead.
        """
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
        """Create design matrix for in-sample fitting.

        Parameters
        ----------
        fitted_values : np.ndarray
            Fitted parameter values.
        target_values : np.ndarray
            Target values.
        distribution : Distribution
            The distribution object for computing residuals.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged residuals.
        """
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
        """Create design matrix for in-sample updating.

        Parameters
        ----------
        fitted_values : np.ndarray
            Current fitted parameter values.
        target_values : np.ndarray
            Current target values.
        distribution : Distribution
            The distribution object for computing residuals.
        state : ARTermState
            State containing memory of previous values.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Lagged residuals including memory.
        """
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
        """Create design matrix for out-of-sample prediction.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (unused).
        distribution : Distribution
            The distribution object for computing residuals.
        state : ARTermState
            State containing memory of fitted and target values.

        Returns
        -------
        np.ndarray
            Lagged residuals for prediction.

        Raises
        ------
        ValueError
            If prediction is not one step ahead.
        """
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
