from typing import Tuple

import numpy as np

from ..base import LinkFunction
from .robust_math import SMALL_NUMBER, robust_exp, robust_log


class Log(LinkFunction):
    """
    The log-link function.

    The log-link function is defined as \(g(x) = \log(x)\).
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / x

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / x**2


class LogShiftValue(LinkFunction):
    """
    The Log-Link function shifted to a value \(v\).

    This link function is defined as \(g(x) = \log(x - v)\). It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.value + np.nextafter(0, 1), np.inf)

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x - self.value + SMALL_NUMBER)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + robust_exp(x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (x - self.value + SMALL_NUMBER)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / (x - self.value + SMALL_NUMBER) ** 2


class LogShiftTwo(LogShiftValue):
    """
    The Log-Link function shifted to 2.

    This link function is defined as \(g(x) = \log(x - 2)\). It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self):
        super().__init__(2)
        pass


class LogIdent(LinkFunction):
    """The Logident Link function.

    The LogIdent Link function has been introduced by [Narajewski & Ziel 2020](https://arxiv.org/pdf/2005.01365) and can be
    used to avoid the exponential inverse for large values while keeping the log-behaviour in small ranges. This can stabilize
    the estimation procedure.
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 1, robust_log(x), x - 1)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, robust_exp(x), x + 1)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, robust_exp(x), 1)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return super().link_derivative(x)

    def link_second_derivative(self, x) -> np.ndarray:
        return super().link_second_derivative(x)
