from abc import ABC, abstractmethod

import numpy as np

from .distribution import Distribution


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

        self.find_multicollinear_columns(X)

    def find_multicollinear_columns(
        self,
        X: np.ndarray,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Find multicollinear columns in the design matrix X.

        Args:
            X (np.ndarray): The design matrix.
            tol (float): Tolerance for detecting multicollinearity.
        Returns:
            np.ndarray: Indices of multicollinear columns.
        """
        n_features = X.shape[1]
        redundant = set()

        # Normalize columns to avoid scaling issues
        X_norm = X / (np.linalg.norm(X, axis=0) + 1e-12)

        # Keep track of independent columns so far
        independent_cols = []

        for j in range(n_features):
            if j in redundant:
                continue
            if not independent_cols:
                independent_cols.append(j)
                continue
            # Build matrix of current independent columns + new candidate column
            # Compute SVD once for small matrix
            _, s, _ = np.linalg.svd(
                X_norm[:, independent_cols + [j]], full_matrices=False
            )

            # If smallest singular value ~ 0 → column j is dependent
            if s[-1] < tol:
                redundant.add(j)
            else:
                independent_cols.append(j)

        self._colinear = set(redundant)

    def remove_zero_variance_columns(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Remove zero-variance columns from the design matrix X.

        Args:
            X (np.ndarray): The design matrix.

        Returns:
            np.ndarray: The design matrix without columns with variance 0.
        """
        remove = set()
        if len(self._zero_std_cols) > int(self.fit_intercept):
            remove = remove.union(set(self._zero_std_cols[int(self.fit_intercept) :]))
        if len(self._colinear) > 0:
            remove = remove.union(self._colinear)
        remove = sorted(list(remove))

        X = np.delete(X, remove, axis=1)
        return X

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "Term":
        raise NotImplementedError("Not implemented")

    def predict_in_sample_during_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        return self.predict_out_of_sample(
            X=X,
            distribution=distribution,
        )

    def predict_in_sample_during_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
    ):
        # For all terms that do not implement their own
        # predict_in_sample_during_update, we use the same
        # logic as in predict_in_sample_during_fit
        # This is valid for terms that do not consist of time series
        # components.
        return self.predict_in_sample_during_fit(
            X=X,
            y=y,
            fitted_values=fitted_values,
            target_values=target_values,
            distribution=distribution,
        )

    @abstractmethod
    def predict_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
    ) -> "Term":
        raise NotImplementedError("Not implemented")


class FeatureTransformation(ABC):
    """Base class for feature transformations."""

    @abstractmethod
    def make_design_matrix_in_sample_during_fit(
        self,
        X: np.ndarray,
        **kwargs,
    ):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def make_design_matrix_in_sample_during_update(
        self,
        X: np.ndarray,
        **kwargs,
    ):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def make_design_matrix_out_of_sample(
        self,
        X,
        **kwargs,
    ):
        raise NotImplementedError("Not implemented")
