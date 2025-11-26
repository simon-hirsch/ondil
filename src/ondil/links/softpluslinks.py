from typing import Tuple

import numpy as np

from ..base import LinkFunction
from ..robust_math import (
    robust_exp,
    robust_softplus,
    robust_softplus_inverse,
    zero_safe_division,
)


class InverseSoftPlus(LinkFunction):
    r"""
    The softplus is defined as $$
        \operatorname{SoftPlus(x)} = \log(1 + \exp(x))
    $$ and hence the inverse is defined as $$
        \log(\exp(x) - 1)
    $$ which can be used as link function for the parameters on the
    positive real line. The behavior of the inverse softplus is more
    graceful on large values as it avoids the exp of the log-link and
    converges towards a linear behaviour.

    The softplus is the smooth approximation of $\max(x, 0)$.
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray):
        return robust_softplus_inverse(x)

    def inverse(self, x: np.ndarray):
        return robust_softplus(x)

    def link_derivative(self, x: np.ndarray):
        return zero_safe_division(1, (robust_exp(x) - 1)) + 1

    def link_second_derivative(self, x: np.ndarray):
        # return 1 / (2 - 2 * np.cosh(x)) # equivalent
        # return -1 / (np.exp(x) - 1) - 1 / (np.exp(x) - 1) ** 2
        return zero_safe_division(1, (2 - 2 * np.cosh(x)))

    def inverse_derivative(self, x: np.ndarray):
        return zero_safe_division(1, (robust_exp(-x) + 1))


class InverseSoftPlusShiftValue(LinkFunction):
    r"""
    The Inverse SoftPlus function shifted to a value \(v\).
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.value + np.nextafter(0, 1), np.inf)

    def link(self, x: np.ndarray):
        z = x - self.value
        # return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
        # return robust_log(robust_exp(z) - 1)
        return robust_softplus_inverse(z)

    def inverse(self, x: np.ndarray):
        return robust_softplus(x) + self.value

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (robust_exp(-x) + 1))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (1 - robust_exp(self.value - x)))

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (2 - 2 * np.cosh(self.value - x)))


class InverseSoftPlusShiftTwo(InverseSoftPlusShiftValue):
    r"""
    The Inverse SoftPlus function shifted to 2.
    """

    def __init__(self):
        super().__init__(2)
        pass
