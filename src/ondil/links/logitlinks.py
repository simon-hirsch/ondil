import numpy as np

from ..base import LinkFunction
from .robust_math import robust_exp, robust_log


class Logit(LinkFunction):
    r"""The Logit Link function.

    The logit-link function is defined as \(g(x) = \log (x/ (1-x))\).
    """

    link_support = (np.nextafter(0, 1), np.nextafter(1, 0))

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x / (1 - x))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x) / (robust_exp(x) + 1)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x) / ((robust_exp(x) + 1) ** 2)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (x * (1 - x))

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -(1 - 2 * x) / ((x**2) * ((1 - x) ** 2))
