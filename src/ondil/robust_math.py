import numba as nb
import numpy as np

SMALL_NUMBER = 1e-10
LARGE_NUMBER = 1e15
LOG_SMALL_NUMBER = np.log(SMALL_NUMBER)
LOG_LARGE_NUMBER = np.log(LARGE_NUMBER)


@nb.vectorize(["float64(float64)", "float32(float32)"])
def robust_log(x: np.ndarray) -> np.ndarray:
    """
    A robust log function that handles negative and zero values.

    This function returns the logarithm of the input array, replacing
    negative and zero values with a small positive number to avoid
    undefined logarithm values.
    """
    if x > SMALL_NUMBER:
        return np.log(x)
    else:
        return LOG_SMALL_NUMBER


@nb.vectorize(["float64(float64)", "float32(float32)"])
def robust_exp(x: np.ndarray) -> np.ndarray:
    """
    A robust exponential function that handles large values.
    """

    if x > LOG_LARGE_NUMBER:
        return LARGE_NUMBER
    else:
        return np.exp(x)


@nb.vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def robust_sigmoid(value, k):
    if value / k > 50:
        return 1
    elif value / k < -50:
        return 0
    else:
        return 1 / (1 + robust_exp(-k * (value - 1)))


@nb.vectorize(["float32(float32)", "float64(float64)"])
def robust_softplus(value):
    if value > 40:
        return value
    else:
        return np.log(1 + np.exp(value))


@nb.vectorize(["float32(float32)", "float64(float64)"])
def robust_softplus_inverse(value):
    if value > 40:
        return value
    elif value < SMALL_NUMBER:
        return -30
    else:
        return np.log(np.exp(value) - 1)


@nb.vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def zero_safe_division(a, b):
    """Return the max float value for the current precision at the a / b if
    b becomes 0. Returns

    Args:
        a (float): Nominator.
        b (float): Denominator.

    Returns:
        float: Result of the division a / b.
    """
    if np.isclose(b, 0):
        return LARGE_NUMBER
    elif np.isclose(a, 0):
        return LARGE_NUMBER
    else:
        return a / b
