import numpy as np

from ..base import LinkFunction


class Identity(LinkFunction):
    r"""
    The identity link function.

    The identity link is defined as \(g(x) = x\).
    """

    link_support = (-np.inf, np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class LowerTruncatedIdentity(LinkFunction):
    r"""
    The identity link function.

    The lower truncacted identity link is defined as $$g(x) = \max(x, value)$$.
    """

    def __init__(self, lower: float):
        self.lower = lower

    @property
    def link_support(self):
        return (self.lower, np.inf)

    def link(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, a_min=self.lower, a_max=None)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
