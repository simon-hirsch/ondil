import copy
from dataclasses import dataclass

from .gram import init_forget_vector
import numpy as np


@dataclass(frozen=True)
class IncrementalStatistics:
    mean: float | np.ndarray
    var: float | np.ndarray
    m: float | np.ndarray
    weight: float
    forget: float


def calculate_statistics(
    X: np.ndarray,
    forget: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> IncrementalStatistics:
    """Calculate the variance of X with optional sample weights.

    Args:
        X (np.ndarray): Input data.
        sample_weight (np.ndarray | None, optional): Sample weights. Defaults to None.

    Returns:
        np.ndarray: Variance of X.
    """
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])

    forget_weight = init_forget_vector(forget, X.shape[0])
    effective_weight = sample_weight * forget_weight

    mean = np.average(X, weights=sample_weight * effective_weight, axis=0)
    diff_sq = (X - mean) ** 2
    var = np.average(diff_sq, weights=effective_weight, axis=0)
    weight = np.sum(effective_weight)
    m = var * weight
    return IncrementalStatistics(
        mean=mean,
        var=var,
        weight=weight,
        forget=forget,
        m=m,
    )


def update_statistics(
    incremental_statistics: IncrementalStatistics,
    X: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> IncrementalStatistics:
    """Update the online variance with new data X and optional sample weights.

    Args:
        incremental_statistics (IncrementalStatistics): Current online variance state.
        X (np.ndarray): New input data.
        sample_weight (np.ndarray | None, optional): Sample weights. Defaults to None.

    Returns:
        OnlineVariance: Updated online variance state.
    """
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])

    for i in range(X.shape[0]):
        eff_old_w = incremental_statistics.weight * (1 - incremental_statistics.forget)
        eff_new_w = sample_weight[i] + eff_old_w
        diff_old = X[i] - incremental_statistics.mean

        mean = (
            incremental_statistics.mean * eff_old_w + X[i, :] * sample_weight[i]
        ) / eff_new_w
        diff_new = X[i, :] - mean

        m = (
            incremental_statistics.m * (1 - incremental_statistics.forget)
            + sample_weight[i] * diff_old * diff_new
        )

        incremental_statistics = IncrementalStatistics(
            mean=mean,
            var=m / eff_new_w,
            weight=eff_new_w,
            forget=incremental_statistics.forget,
            m=m,
        )
    return incremental_statistics


__all__ = [
    "IncrementalStatistics",
    "calculate_statistics",
    "update_statistics",
]
