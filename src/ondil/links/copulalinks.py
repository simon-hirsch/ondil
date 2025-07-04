from typing import Tuple

import numpy as np

from ..base import LinkFunction

class FisherZLink(LinkFunction):
    """
    The Fisher Z transform.

    The Fisher Z transform is defined as:
        $$ z = \frac{1}{2} \log\left(\frac{1 + r}{1 - r}\right) $$
    The inverse is defined as:
        $$ r = \frac{\exp(2z) - 1}{\exp(2z) + 1} $$
    This link function maps values from the range (-1, 1) to the real line and vice versa.

    Note:
        2 * atanh(x) = log((1 + x) / (1 - x)), so atanh(x) = 0.5 * log((1 + x) / (1 - x)).
        Thus, Fisher Z transform is exactly atanh(x), and 2 * atanh(x) = log((1 + x) / (1 - x)).
    """
    
    # The Fisher Z transform is defined for x in (-1, 1), exclusive.
    link_support = (np.nextafter(-1, 0), np.nextafter(1, 0))

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log((1+x)/(1-x))*(1 - 1e-5)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        # Ensure output is strictly within (-1, 1)
        out = np.tanh(x / 2)
        out = np.clip(out, -1 + 1e-5, 1 - 1e-5)
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # d1 = 1 / (1 + cosh(x))
        return 1.0 / (1.0 + np.cosh(x))

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        # d2 = -4 * sinh(x / 2)^4 * (1 / sinh(x))^3
        sinh_half = np.sinh(x / 2.0)
        sinh_x = np.sinh(x)
        # Avoid division by zero
        sinh_x_safe = np.where(np.abs(sinh_x) < 1e-10, 1e-10, sinh_x)
        return -4.0 * sinh_half**4 * (1.0 / sinh_x_safe)**3
    
    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # The derivative of the inverse Fisher Z transform (tanh(x/2)) is 0.5 * sech^2(x/2)
        return 0.5 * (1 / np.cosh(x / 2)) ** 2


class KendallsTauToParameter(LinkFunction):
    """
    Link function mapping Kendall's tau to the Gaussian copula correlation parameter rho.

    The relationship is:
        rho = sin(pi/2 * tau)
    The inverse is:
        tau = (2/pi) * arcsin(rho)
    """
    # The tau parameter is in (-1, 1), but for the Gaussian copula, rho is also in (-1, 1).
    # For practical numerical stability, avoid endpoints.
    link_support = (np.nextafter(-1, 0), np.nextafter(1, 0))

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        # Map tau to rho
        return (2 / np.pi) * np.arcsin(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        # Map rho to tau
        return np.sin((np.pi / 2) * x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of sin(pi/2 * x) w.r.t x
        return (np.pi / 2) * np.cos((np.pi / 2) * x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        # Second derivative of sin(pi/2 * x) w.r.t x
        return -((np.pi / 2) ** 2) * np.sin((np.pi / 2) * x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of (2/pi) * arcsin(x) w.r.t x
        return (2 / np.pi) / np.sqrt(1 - x ** 2)
