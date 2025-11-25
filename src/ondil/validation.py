import numpy as np


def ensure_atleast_bivariate(y: np.ndarray) -> None:
    r"""Validates that the response variable is a multivariate array with at least two dimensions.

    Args:
        y (np.ndarray): The response variable to validate.
    Raises:
        ValueError: If `y` does not have at least two dimensions and has less than two columns.
    """
    if y.shape[1] < 2:
        raise ValueError(
            "Multivariate response must have at least two dimensions. "
            f"Got {y.shape[1]} dimensions."
        )


def validate_response_support(estimator, y: np.ndarray) -> None:
    r"""Validates that the response values are within the distribution support of the estimator.

    Args:
        estimator: The estimator with a distribution that defines the support.
        y (np.ndarray): The response variable to validate.
    Raises:
        ValueError: If any response values are outside the distribution support.

    """
    if np.any(y < estimator.distribution.distribution_support[0]):
        raise ValueError(
            f"Response values must be greater than or equal to {estimator.distribution.distribution_support[0]}. "
            f"Got values less than this: {y[y < estimator.distribution.distribution_support[0]]}."
        )
    if np.any(y > estimator.distribution.distribution_support[1]):
        raise ValueError(
            f"Response values must be less than or equal to {estimator.distribution.distribution_support[1]}. "
            f"Got values greater than this: {y[y > estimator.distribution.distribution_support[1]]}."
        )
