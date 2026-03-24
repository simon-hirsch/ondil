import numpy as np

from ..base import LinkFunction


class FisherZLink(LinkFunction):
    """
    The Fisher Z transform.

    The relationship is:

        z = log((1 + r) / (1 - r))

    The inverse is:

        r = tanh(z / 2)

    This mapping transforms r in (-1, 1)
    to the real line and vice versa.
    """

    # r support (avoid ±1 exactly)
    link_support = (np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0))

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def link(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -1 + self.eps, 1 - self.eps)
        return np.log((1.0 + x) / (1.0 - x))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x / 2.0)
        return np.clip(out, -1 + self.eps, 1 - self.eps)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # derivative of log((1+x)/(1-x)) = 2 / (1 - x^2)
        x = np.clip(x, -1 + self.eps, 1 - self.eps)
        return 2.0 / (1.0 - x**2)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        # second derivative = 4x / (1 - x^2)^2
        x = np.clip(x, -1 + self.eps, 1 - self.eps)
        return 4.0 * x / (1.0 - x**2) ** 2

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        # derivative of tanh(z/2) = 0.5 * sech^2(z/2)
        return 0.5 / np.cosh(x / 2.0) ** 2

class GumbelLink(LinkFunction):
    """
    Link function for the Gumbel copula parameter.

    The Gumbel copula parameter satisfies theta >= 1.

    The relationship is:

        z = log(theta - 1)

    The inverse is:

        theta = exp(z) + 1

    This mapping transforms the parameter domain [1, ∞)
    to the real line and vice versa.
    """

    # theta support
    link_support = (1.0, np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(x - 1)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) + 1

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (x - 1.0)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1.0 / (x - 1.0) ** 2

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

class ParameterToKendallsTau(LinkFunction):
    """
    Link function mapping Gaussian copula parameter to Kendall's tau.

    The relationship is:
       tau = (2/pi) * arcsin(rho)
    The inverse is:
        rho = sin(pi/2 * tau)
    """

    # The tau parameter is in (-1, 1), but for the Gaussian copula, rho is also in (-1, 1).
    # For practical numerical stability, avoid endpoints.
    link_support = (np.nextafter(-1, 0), np.nextafter(1, 0))

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        # Map rho to tau
        return (2 / np.pi) * np.arcsin(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        # Map tau to rho
        return np.sin((np.pi / 2) * x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return (np.pi / 2) * np.cos((np.pi / 2) * x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -((np.pi / 2) ** 2) * np.sin((np.pi / 2) * x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return (2 / np.pi) / np.sqrt(1 - x**2)


class GumbelParameterToKendallsTau(LinkFunction):
    """
    Link function implementing the custom transform

        z = sign(x) - 1/x

    The inverse is given by

        x = z / (|z|(1 - |z|)).

    This mapping is defined for x ≠ 0 and |z| ∈ (0, 1).
    For numerical stability, values near 0 and ±1 are clipped.
    """

    link_support = (np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def link(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        # sign(0) -> +1 (same intent as before)
        sign_x = np.sign(x)
        sign_x = np.where(sign_x == 0.0, 1.0, sign_x)

        # protect 1/x near 0
        x_safe = np.where(np.abs(x) < self.eps, sign_x * self.eps, x)

        return sign_x - 1.0 / x_safe

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        # protect denom near |x|=0 and |x|=1
        ax = np.abs(x)
        ax_safe = np.clip(ax, self.eps, 1.0 - self.eps)

        denom = ax_safe * (1.0 - ax_safe)
        return x / denom

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 - np.abs(x)) ** 2

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -2.0 * np.sign(x) / (np.abs(x) - 1.0) ** 3

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        numerator = (1.0 - abs_x) * abs_x - x * sign_x * (1.0 - 2.0 * abs_x)
        denominator = ((1.0 - abs_x) * abs_x) ** 2
        return numerator / denominator

class ClaytonParameterToKendallsTau(LinkFunction):
    """
    Link function implementing the custom Clayton transform.

    The relationship is defined as:

        tau = theta / (2 + |theta|)

    The inverse is:

        theta = 2 * tau / (1 - |tau|)

    This mapping transforms the copula parameter theta to a bounded
    dependence measure tau in (-1, 1). For numerical stability,
    values near ±1 are clipped.
    """

    # tau support (avoid ±1 exactly)
    link_support = (np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0))

    def __init__(self):
        self.eps = 1e-12

    def link(self, x: np.ndarray) -> np.ndarray:
        return x / (2 + np.abs(x))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.clip(np.abs(x), 0, 1 - self.eps)
        return 2 * x / (1 - abs_x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.clip(np.abs(x), 0, 1 - self.eps)
        return 2 / (1 - abs_x) ** 2

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.clip(np.abs(x), 0, 1 - self.eps)
        sign_x = np.sign(x)
        return 4 * sign_x / (1 - abs_x) ** 3

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.clip(np.abs(x), 0, 1 - self.eps)
        return 2 / (1 - abs_x) ** 2