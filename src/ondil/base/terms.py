from abc import ABC, abstractmethod

import numpy as np


class Term(ABC):
    """The base class for terms in structured additive distributional regression."""

    def _prepare_term(self) -> "Term":
        return self

    @property
    @abstractmethod
    def allow_online_updates(self) -> bool:
        return True

    def find_zero_variance_columns(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Find columns with variance 0 in the design matrix X.

        Args:
            X (np.ndarray): The design matrix.
        Returns:
            np.ndarray: Indices of columns with variance 0. columns.
        """

        # Remove the columns with zero standard deviation
        # but keep the first one (intercept) if present
        self._zero_std_cols = np.where(np.isclose(X.std(axis=0), 0))[0]
        if len(self._zero_std_cols) > int(self.fit_intercept):
            print(
                f"Found: {len(self._zero_std_cols)} columns with variance 0. Removing them."
            )

    def remove_zero_variance_columns(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Remove columns with variance 0from the design matrix X.

        Args:
            X (np.ndarray): The design matrix.

        Returns:
            np.ndarray: The design matrix without columns with variance 0.
        """
        if len(self._zero_std_cols) > int(self.fit_intercept):
            X = np.delete(X, self._zero_std_cols[int(self.fit_intercept) :], axis=1)
        return X

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ) -> "Term":
        raise NotImplementedError("Not implemented")

    def predict_in_sample_during_fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
    ) -> np.ndarray:
        return self.predict_out_of_sample(X=X)

    def predict_in_sample_during_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
    ):
        # For all terms that do not implement their own
        # predict_in_sample_during_update, we use the same
        # logic as in predict_in_sample_during_fit
        # This is valid for terms that do not consist of time series
        # components.
        return self.predict_in_sample_during_fit(
            X,
            y,
            fitted_values=fitted_values,
            target_values=target_values,
        )

    @abstractmethod
    def predict_out_of_sample(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray = None,
        target_values: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ) -> "Term":
        raise NotImplementedError("Not implemented")
