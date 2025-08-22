"""Sktime adapter for OnlineLinearModel estimator."""

import numpy as np
from sktime.regression.base import BaseRegressor

from .online_linear_model import OnlineLinearModel


class OnlineLinearModelSktime(BaseRegressor):
    """Sktime adapter for OnlineLinearModel regressor.
    
    This adapter wraps the OnlineLinearModel estimator to make it compatible
    with the sktime framework while preserving its online learning capabilities.
    
    Parameters
    ----------
    forget : float, default=0.0
        Exponential discounting of old observations.
    scale_inputs : bool or np.ndarray, default=True
        Whether to scale the X matrix.
    fit_intercept : bool, default=True
        Whether to add an intercept in the estimation.
    regularize_intercept : bool, default=False
        Whether to regularize the intercept.
    method : EstimationMethod or str, default="ols"
        The estimation method. Can be a string or EstimationMethod class.
    ic : str, default="bic"
        The information criteria for model selection.
        One of "aic", "bic", "hqc", "max".
    """
    
    def __init__(
        self,
        forget=0.0,
        scale_inputs=True,
        fit_intercept=True,
        regularize_intercept=False,
        method="ols",
        ic="bic",
    ):
        self.forget = forget
        self.scale_inputs = scale_inputs
        self.fit_intercept = fit_intercept
        self.regularize_intercept = regularize_intercept
        self.method = method
        self.ic = ic
        
        super().__init__()
        
        # Set sktime tags for regression
        self.set_tags(**{
            "capability:multivariate": True,
            "capability:univariate": True,
            "capability:train_estimate": True,
            "X_inner_mtype": "numpy3D",
            "y_inner_mtype": "numpy1D",
            "requires_y": True,
            "enforce_index_type": None,
        })

    def _fit(self, X, y):
        """Fit the OnlineLinearModel to training data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_features)
            Training data features.
        y : np.ndarray of shape (n_instances,)
            Training data targets.
            
        Returns
        -------
        self : object
            Reference to self.
        """
        # Convert sktime format to sklearn format if needed
        if X.ndim == 3:
            # Flatten multivariate time series to 2D for OnlineLinearModel
            n_instances, n_timepoints, n_features = X.shape
            X = X.reshape(n_instances, n_timepoints * n_features)
        
        # Initialize the underlying OnlineLinearModel
        self._estimator = OnlineLinearModel(
            forget=self.forget,
            scale_inputs=self.scale_inputs,
            fit_intercept=self.fit_intercept,
            regularize_intercept=self.regularize_intercept,
            method=self.method,
            ic=self.ic,
        )
        
        # Fit the underlying estimator
        self._estimator.fit(X, y)
        
        return self

    def _predict(self, X):
        """Predict using the fitted OnlineLinearModel.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_features)
            Data to predict on.
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_instances,)
            Predicted values.
        """
        # Convert sktime format to sklearn format if needed
        if X.ndim == 3:
            # Flatten multivariate time series to 2D for OnlineLinearModel
            n_instances, n_timepoints, n_features = X.shape
            X = X.reshape(n_instances, n_timepoints * n_features)
        
        # Make predictions using the underlying estimator
        return self._estimator.predict(X)
    
    def update(self, X, y, sample_weight=None):
        """Update the model with new data.
        
        This method enables online learning by updating the model
        with new observations without full retraining.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_features)
            New training data features.
        y : np.ndarray of shape (n_instances,)
            New training data targets.
        sample_weight : np.ndarray of shape (n_instances,), optional
            Sample weights for the new data.
            
        Returns
        -------
        self : object
            Reference to self.
        """
        # Convert sktime format to sklearn format if needed
        if X.ndim == 3:
            # Flatten multivariate time series to 2D for OnlineLinearModel
            n_instances, n_timepoints, n_features = X.shape
            X = X.reshape(n_instances, n_timepoints * n_features)
        
        # Update the underlying estimator
        self._estimator.update(X, y, sample_weight=sample_weight)
        
        return self
    
    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_features)
            Test samples.
        y : np.ndarray of shape (n_instances,)
            True values for X.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        # Convert sktime format to sklearn format if needed
        if X.ndim == 3:
            # Flatten multivariate time series to 2D for OnlineLinearModel
            n_instances, n_timepoints, n_features = X.shape
            X = X.reshape(n_instances, n_timepoints * n_features)
        
        return self._estimator.score(X, y)
    
    @property
    def coef_(self):
        """Get the model coefficients."""
        return self._estimator.coef_
    
    @property
    def coef_path_(self):
        """Get the full regularization path coefficients."""
        return self._estimator.coef_path_
    
    def get_fitted_params(self, deep=True):
        """Get fitted parameters as a dictionary.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to return fitted parameters of sub-estimators.
            
        Returns
        -------
        fitted_params : dict
            Dictionary of fitted parameters.
        """
        fitted_params = {}
        if hasattr(self, '_estimator'):
            fitted_params['coef_'] = self.coef_
            if hasattr(self._estimator, 'coef_path_'):
                fitted_params['coef_path_'] = self.coef_path_
            if hasattr(self._estimator, 'n_observations_'):
                fitted_params['n_observations_'] = self._estimator.n_observations_
            if hasattr(self._estimator, 'n_features_'):
                fitted_params['n_features_'] = self._estimator.n_features_
        return fitted_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.
        
        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.
            
        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "forget": 0.0,
            "scale_inputs": True,
            "fit_intercept": True,
            "method": "ols",
            "ic": "bic",
        }
        
        params2 = {
            "forget": 0.1,
            "scale_inputs": False,
            "fit_intercept": False,
            "method": "lasso",
            "ic": "aic",
        }
        
        return [params1, params2]