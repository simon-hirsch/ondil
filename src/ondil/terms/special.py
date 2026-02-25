from typing import Literal

import numpy as np

from ..base import Distribution, Term
from ..design_matrix import subset_array


class ScikitLearnEstimatorTerm(Term):
    """Term that wraps a scikit-learn estimator for use in distributional regression.

    This term allows using any scikit-learn compatible estimator as a component
    in structured additive distributional regression models.
    """

    allow_online_updates: bool = False

    def __init__(
        self,
        sklearn_estimator,
        features: np.ndarray | list[int] | Literal["all"],
    ):
        """Initialize the ScikitLearnEstimatorTerm.

        Parameters
        ----------
        sklearn_estimator : sklearn estimator
            A scikit-learn compatible estimator instance.
        features : np.ndarray | list[int] | Literal["all"]
            Indices of features to select for the estimator.
        """
        self.sklearn_estimator = sklearn_estimator
        self.features = features

    def _prepare_term(self):
        """Prepare the term by setting the method name.

        Returns
        -------
        ScikitLearnEstimatorTerm
            The prepared term instance.
        """
        self.method = self.sklearn_estimator.__class__.__name__
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        fitted_values: np.ndarray,
        distribution: Distribution,
        target_values: np.ndarray,
    ) -> "ScikitLearnEstimatorTerm":
        """Fit the scikit-learn estimator.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        y : np.ndarray
            Target values.
        sample_weight : np.ndarray
            Sample weights.
        fitted_values : np.ndarray
            Fitted values (unused).
        distribution : Distribution
            Distribution object (unused).
        target_values : np.ndarray
            Target values (unused).

        Returns
        -------
        ScikitLearnEstimatorTerm
            The fitted term.
        """
        X_mat = subset_array(X, self.features)

        if sample_weight is not None:
            self.sklearn_estimator.fit(X_mat, y, sample_weight=sample_weight)
        else:
            self.sklearn_estimator.fit(X_mat, y)
        return self

    def predict_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        """Predict out-of-sample values using the fitted estimator.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        distribution : Distribution
            Distribution object (unused).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        X_mat = subset_array(X, self.features)
        return self.sklearn_estimator.predict(X_mat)

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "ScikitLearnEstimatorTerm":
        """Update the term (not supported for scikit-learn estimators).

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        y : np.ndarray
            Target values.
        fitted_values : np.ndarray
            Fitted values.
        target_values : np.ndarray
            Target values.
        distribution : Distribution
            Distribution object.
        sample_weight : np.ndarray
            Sample weights.

        Raises
        ------
        NotImplementedError
            Online updates are not supported.
        """
        raise NotImplementedError(
            "Online updates are not supported for ScikitLearnEstimatorTerm."
        )
