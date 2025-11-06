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
        return np.log((1 + x) / (1 - x)) * (1 - 1e-5)

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
        return -4.0 * sinh_half**4 * (1.0 / sinh_x_safe) ** 3

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # The derivative of the inverse Fisher Z transform (tanh(x/2)) is 0.5 * sech^2(x/2)
        return 0.5 * (1 / np.cosh(x / 2)) ** 2


class GumbelLink(LinkFunction):
    """
    Link function for the Gumbel copula parameter.

    The Gumbel copula parameter theta must be >= 1. This link function maps
    theta from [1, inf) to the real line and vice versa using:
        z = log(theta - 1)
    The inverse is:
        theta = exp(z) + 1
    """

    # The Gumbel parameter is defined for theta >= 1
    link_support = (1.0, np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        # Map theta to real line: log(theta - 1)
        return np.log(x - 1)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        # Map real line to theta: exp(z) + 1
        return np.exp(x) + 1

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of log(x - 1) w.r.t x
        return 1 / (x - 1)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        # Second derivative of log(x - 1) w.r.t x
        return -1 / (x - 1) ** 2

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of exp(x) + 1 w.r.t x
        return np.exp(x)


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
        return (2 / np.pi) / np.sqrt(1 - x**2)


class KendallsTauToParameterGumbel(LinkFunction):
    """
    The Gumbel copula link function.

    The Gumbel copula link function is defined as:
        $$ z = -\log\left(-\log\left(F_X(x)\right)\right) $$
    The inverse is defined as:
        $$ F_X(x) = \exp\left(-\exp\left(-z\right)\right) $$
    This link function maps values from the range (0, 1) to the real line and vice versa.
    """

    # The Gumbel link function is defined for x in (0, 1), exclusive.
    link_support = (np.nextafter(0, 1), np.nextafter(1, 0))

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return x / np.abs(x) - 1 / x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x / ((1 - np.abs(x)) * np.abs(x))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of x/|x| - 1/x w.r.t x
        # d/dx(x/|x|) = 0 for x != 0 (since x/|x| = sign(x))
        # d/dx(-1/x) = 1/x^2
        return 1 / (1 - np.abs(x)) ** 2

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        # Second derivative of x/|x| - 1/x w.r.t x
        # d²/dx²(-1/x) = -2/x^3
        return -2 * np.sign(x) / (np.abs(x) - 1) ** 3

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of x/((1-|x|)*|x|) w.r.t x
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        numerator = (1 - abs_x) * abs_x - x * sign_x * (1 - 2 * abs_x)
        denominator = ((1 - abs_x) * abs_x) ** 2
        return numerator / denominator


class KendallsTauToParameterClayton(LinkFunction):
    """
    The Clayton copula link function.

    The Clayton copula link function is defined as:
        $$ z = -\log\left(-\log\left(F_X(x)\right)\right) $$
    The inverse is defined as:
        $$ F_X(x) = \exp\left(-\exp\left(-z\right)\right) $$
    This link function maps values from the range (0, 1) to the real line and vice versa.
    """

    # The Gumbel link function is defined for x in (0, 1), exclusive.
    link_support = (np.nextafter(0, 1), np.nextafter(1, 0))

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return x / (2 + np.abs(x))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return 2 * x / (1 - np.abs(x))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of x/(2 + |x|) w.r.t x
        abs_x = np.abs(x)
        return 2 / (1 - abs_x) ** 2

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        # Second derivative of x/(2 + |x|) w.r.t x
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        return -4 * sign_x / (abs_x - 1) ** 3

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # Derivative of 2*x/(1 - |x|) w.r.t x
        abs_x = np.abs(x)
        return 2 / (1 - abs_x) ** 2
