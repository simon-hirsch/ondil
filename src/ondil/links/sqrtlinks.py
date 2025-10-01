from typing import Tuple

import numpy as np

from ..base import LinkFunction
from .robust_math import SMALL_NUMBER


class Sqrt(LinkFunction):
    """The square root Link function.

    The square root link function is defined as $$g(x) = \sqrt(x)$$.
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.power(x, 2)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (2 * np.sqrt(x))

    def link_second_derivative(self, x) -> np.ndarray:
        return -1 / (4 * x ** (3 / 2))


class SqrtShiftValue(LinkFunction):
    """
    The Sqrt-Link function shifted to a value \(v\).

    This link function is defined as $$g(x) = \sqrt(x - v)$$. It can be used
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
        return np.sqrt(x - self.value + SMALL_NUMBER)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.power(x, 2)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (2 * np.sqrt(x - self.value + SMALL_NUMBER))

    def link_second_derivative(self, x) -> np.ndarray:
        return -1 / (4 * (x - self.value + SMALL_NUMBER) ** (3 / 2))


class SqrtShiftTwo(SqrtShiftValue):
    """
    The Sqrt-Link function shifted to 2.

    This link function is defined as $$g(x) = \sqrt(x - 2)$$. It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self):
        super().__init__(2)
        pass
