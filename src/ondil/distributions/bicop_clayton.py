import numpy as np
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import KendallsTauToParameterClayton, Log
from ..types import ParameterShapes


class BivariateCopulaClayton(BivariateCopulaMixin, CopulaMixin, Distribution):
    """
    Bivariate Clayton copula distribution class.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (1e-6, np.inf)}
    distribution_support = (0, 1)
    n_params = len(parameter_names)
    parameter_shape = {0: ParameterShapes.SCALAR}

    def __init__(
        self,
        link: LinkFunction = Log(),
        param_link: LinkFunction = KendallsTauToParameterClayton(),
        family_code: int = 31,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
        )
        self.family_code = family_code
        self.is_multivariate = True
        self._regularization_allowed = {0: False}

    @property
    def rotation(self):
        """Return the effective rotation based on family code in degrees."""
        return {
            301: 0,
            302: 90,
            303: 180,
            304: 270,
        }[self.family_code]

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1}

    def theta_to_params(self, theta):
        return theta[0]

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        return {"theta": theta}

    def set_initial_guess(self, theta, param):
        return theta

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Pearson correlation for each sample
        # y is expected to be (M, 2)
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        return np.full((M, 1), tau)

    def theta_to_scipy(self, theta: dict):
        return {"theta": theta}

    def logpdf(self, y, theta):
        theta_logpdf = self.theta_to_params(theta)
        result = _log_likelihood(y, theta_logpdf, self.family_code)
        return result

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def dl1_dp1(self, y, theta, param=0):
        theta = self.theta_to_params(theta)
        return _derivative_1st(y, theta, self.family_code)

    def dl2_dp2(self, y, theta, param=0, clip=False):
        """
        Second derivative with proper chain rule matching gamBiCopFit.R
        """
        theta = self.theta_to_params(theta)
        return _derivative_2nd(y, theta, self.family_code)

    def element_score(self, y, theta, param=0, k=0):
        return self.element_dl1_dp1(y, theta, param, k)

    def element_hessian(self, y, theta, param=0, k=0):
        return self.element_dl2_dp2(y, theta, param, k)

    def element_dl1_dp1(self, y, theta, param=0, k=0, clip=False):
        # Apply proper chain rule like dl1_dp1
        theta_params = self.theta_to_params(theta)
        deriv = _derivative_1st(y, chol=theta_params, family_code=self.family_code)
        return deriv

    def element_dl2_dp2(self, y, theta, param=0, k=0, clip=False):
        theta_params = self.theta_to_params(theta)
        deriv = _derivative_2nd(y, theta=theta_params, family_code=self.family_code)
        return deriv

    def dl2_dpp(self, y, theta, param=0):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def calculate_conditional_initial_values(self, y: np.ndarray, theta: dict) -> dict:
        raise NotImplementedError("Not implemented for Clayton copula.")

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def rvs(self, size, theta, family_code=None):
        """
        Generate random samples from the bivariate normal copula.

        Args:
            size (int): Number of samples to generate.
            theta (dict or np.ndarray): Correlation parameter(s).

        Returns:
            np.ndarray: Samples of shape (size, 2) in (0, 1).
        """
        # Generate standard normal samples

        z1 = np.random.uniform(size=size)
        z2 = np.random.uniform(size=size)

        x = self.hinv(z1, z2, theta, un=1, family_code=family_code)

        return x

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def get_regularization_size(self, dim: int) -> int:
        return dim

    def hfunc(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int
    ) -> np.ndarray:
        UMIN = 1e-12
        UMAX = 1 - 1e-12
        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX

        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)

        theta = theta.copy()  # <- prevents in-place mutation of caller's array
        u = u.reshape(-1, 1)
        v = v.reshape(-1, 1)

        # Get rotations for all samples
        rotation = get_effective_rotation(theta, family_code)
        # Apply rotation transformations vectorized
        u_rot, v_rot = u.copy(), v.copy()
        # 180° rotation (survival)
        mask_1 = (rotation == 1).squeeze()
        u_rot[mask_1] = 1 - u[mask_1]
        v_rot[mask_1] = 1 - v[mask_1]
        if un == 1:
            # 90° rotation
            mask_2 = (rotation == 2).squeeze()
            u_rot[mask_2] = 1 - u[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = (rotation == 3).squeeze()
            v_rot[mask_3] = 1 - v[mask_3]
            theta[mask_3] = -theta[mask_3]
        else:
            # 90° rotation
            mask_2 = (rotation == 2).squeeze()
            v_rot[mask_2] = 1 - v[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = (rotation == 3).squeeze()
            u_rot[mask_3] = 1 - u[mask_3]
            theta[mask_3] = -theta[mask_3]

        # Vectorized conditional distribution computation
        t1 = v_rot ** (-theta - 1)
        t2 = u_rot ** (-theta) + v_rot ** (-theta) - 1
        t1 = np.where(np.isinf(t1), 1e50, t1)
        t2 = np.where(np.isinf(t2), 1e50, t2)
        t2 = np.maximum(t2, UMIN)
        t3 = -1.0 - 1.0 / theta
        h = t1 * (t2**t3)
        mask_4 = (theta < 1e-4).squeeze()
        h[mask_4] = u_rot[mask_4]

        # Ensure results are in [0,1] using masks
        h_mask_low = h < 0
        h_mask_high = h > 1
        h = np.where(h_mask_low, 0, h)
        h = np.where(h_mask_high, 1, h)

        if un == 1:
            h[mask_1] = 1 - h[mask_1]  # 180° rotation
            h[mask_2] = 1 - h[mask_2]  # 270° rotation
        else:
            h[mask_1] = 1 - h[mask_1]  # 180° rotation
            h[mask_3] = 1 - h[mask_3]  # 270° rotation

        return h.squeeze()

    def hinv(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int
    ) -> np.ndarray:
        """
        Inverse conditional distribution function h^(-1)(u|v) for the bivariate normal copula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Correlation parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute.

        Returns:
            np.ndarray: Array of shape (n,) with inverse conditional probabilities.
        """

        UMIN = 1e-12
        UMAX = 1 - 1e-12

        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX

        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)
        u = u.reshape(-1, 1)
        v = v.reshape(-1, 1)
        theta = theta.copy()  # <- prevents in-place mutation of caller's array

        XEPS = 1e-4
        rotation = get_effective_rotation(theta, family_code)
        # Apply rotation transformations vectorized
        u_rot, v_rot = u.copy(), v.copy()
        # 180° rotation (survival)
        mask_1 = rotation == 1
        u_rot[mask_1] = 1 - u[mask_1]
        v_rot[mask_1] = 1 - v[mask_1]

        if un == 1:
            # 90° rotation
            mask_2 = rotation == 2
            u_rot[mask_2] = 1 - u_rot[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = rotation == 3
            v_rot[mask_3] = 1 - v_rot[mask_3]
            theta[mask_3] = -theta[mask_3]
        else:
            # 90° rotation
            mask_2 = rotation == 2
            v_rot[mask_2] = 1 - v_rot[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = rotation == 3
            u_rot[mask_3] = 1 - u_rot[mask_3]
            theta[mask_3] = -theta[mask_3]

        # Prepare output array
        hinv = np.zeros_like(u)

        # Case 1: theta < XEPS
        mask_small = np.abs(theta) < XEPS
        hinv[mask_small] = u_rot[mask_small]

        # Case 2: theta < 75
        mask_medium = (~mask_small) & (np.abs(theta) < 75)
        if np.any(mask_medium):
            u_med = u_rot[mask_medium]
            v_med = v_rot[mask_medium]
            theta_med = theta[mask_medium]

            term1 = u_med * np.power(v_med, theta_med + 1.0)
            term2 = np.power(term1, -theta_med / (theta_med + 1.0))
            term3 = 1.0 - np.power(v_med, -theta_med)
            hinv[mask_medium] = np.power(term2 + term3, -1.0 / theta_med)

        # Case 3: theta >= 75 (numerical inversion fallback)
        mask_large = (~mask_small) & (~mask_medium)
        if np.any(mask_large):
            u_large = u_rot[mask_large]
            v_large = v_rot[mask_large]
            theta_large = theta[mask_large]
            hinv[mask_large] = _hinv_numerical(
                u_large, v_large, theta_large, self.family_code, un=un
            )

        # Clip output for numerical stability
        # Ensure results are in [0,1] using masks

        h_mask_low = hinv < 0
        h_mask_high = hinv > 1
        hinv = np.where(h_mask_low, 0, hinv)
        hinv = np.where(h_mask_high, 1, hinv)

        if un == 1:
            hinv[mask_1] = 1 - hinv[mask_1]  # 180° rotation
            hinv[mask_2] = 1 - hinv[mask_2]  # 270° rotation
        else:
            hinv[mask_1] = 1 - hinv[mask_1]  # 180° rotation
            hinv[mask_3] = 1 - hinv[mask_3]  # 270° rotation

        return hinv.squeeze()


def _hinv_numerical(
    u: np.ndarray, v: np.ndarray, theta: np.ndarray, family_code: int, un: int
) -> np.ndarray:
    """
    Vectorized numerical inversion of h-function using bisection method.

    Args:
        u (np.ndarray): Target values in (0, 1)
        v (np.ndarray): Conditioning values in (0, 1)
        theta (np.ndarray): Copula parameters
        family_code (int): Family code
        un (int): Determines which conditional to compute

    Returns:
        np.ndarray: Inverse conditional probabilities
    """
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    tol = 1e-12
    max_iter = 50

    # Initialize bounds as arrays matching the shape of theta
    x0 = np.full_like(theta, UMIN, dtype=float)
    x1 = np.full_like(theta, UMAX, dtype=float)

    # Create dummy BivariateCopulaClayton instance to access hfunc
    temp_copula = BivariateCopulaClayton(family_code=family_code)
    # Evaluate at boundaries

    fl = (temp_copula.hfunc(x0, v, theta, un, family_code) - u).reshape(-1, 1)
    fh = (temp_copula.hfunc(x1, v, theta, un, family_code) - u).reshape(-1, 1)

    # Initialize result
    ans = (x0 + x1) / 2.0

    # Check if solution is at boundaries
    at_lower = np.abs(fl) <= tol
    at_upper = np.abs(fh) <= tol
    ans = np.where(at_lower, x0, ans)
    ans = np.where(at_upper, x1, ans)

    # Track which elements still need iteration
    active = ~(at_lower | at_upper).squeeze()

    # Bisection method
    for it in range(max_iter):
        if not np.any(active):
            break

        # Only update active elements
        ans[active] = (x0[active] + x1[active]) / 2.0

        val = temp_copula.hfunc(ans, v, theta, un, family_code) - u

        # Check convergence
        converged = (np.abs(val) <= tol) | (np.abs(x1 - x0) <= tol)
        active = active & ~converged

        # Update intervals for active elements
        update_upper = active & (val > 0.0)
        update_lower = active & (val <= 0.0)

        x1 = np.where(update_upper, ans, x1)
        fh = np.where(update_upper, val, fh)
        x0 = np.where(update_lower, ans, x0)
        fl = np.where(update_lower, val, fl)

    return ans.squeeze()


##########################################################
# Helper functions for the log-likelihood and derivatives #
##########################################################


def get_effective_rotation(theta_values: np.ndarray, family_code: int) -> np.ndarray:
    """
    Vectorized version of get_effective_rotation().
    Accepts an array of theta_values and returns corresponding rotations.

    Args:
        theta_values (np.ndarray): Copula parameter values (any shape)
        family_code (int): Family code (301–304)

    Returns:
        np.ndarray: Effective rotations (same shape as theta_values)
    """
    theta_values = np.asarray(theta_values)
    rot = np.empty_like(theta_values, dtype=int)

    if family_code == 31:
        rot[:] = np.where(theta_values > 0, 0, 2)
    elif family_code == 32:
        rot[:] = np.where(theta_values > 0, 0, 3)
    elif family_code == 33:
        rot[:] = np.where(theta_values > 0, 1, 2)
    elif family_code == 34:
        rot[:] = np.where(theta_values > 0, 1, 3)
    else:
        raise ValueError(
            f"Unsupported family code: {family_code}. Supported codes: 31, 32, 33, 34."
        )

    return rot


def _log_likelihood(y, theta, family_code=None):
    theta = np.asarray(theta).copy()  # <- prevents in-place mutation of caller's array

    u = y[:, 0].reshape(-1, 1)
    v = y[:, 1].reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)

    # Apply rotation transformations to data vectorized
    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1 - u[mask_1]
    v_rot[mask_1] = 1 - v[mask_1]

    # 90° rotation
    mask_2 = rotation == 2
    u_rot[mask_2] = 1 - u[mask_2]
    theta[mask_2] = -theta[mask_2]

    # 270° rotation
    mask_3 = rotation == 3
    v_rot[mask_3] = 1 - v[mask_3]
    theta[mask_3] = -theta[mask_3]

    # Vectorized computation for all samples at once
    valid_mask = (theta != 0) & (theta >= 1e-10)

    t5 = u_rot ** (-theta)
    t6 = v_rot ** (-theta)
    t7 = np.maximum(t5 + t6 - 1.0, 1e-12)

    f = np.where(
        valid_mask,
        np.log1p(theta)
        - (1.0 + theta) * np.log(np.maximum(u_rot * v_rot, 1e-12))
        - (2.0 + 1.0 / theta) * np.log(t7),
        0,
    )

    # Apply bounds
    XINFMAX = 700
    DBL_MIN = 2.2e-308
    f = np.clip(f, np.log(DBL_MIN), XINFMAX)
    return f.squeeze()  # Always 1D


def _derivative_1st(y, chol, family_code):
    """
    Computes the first derivative of the bivariate Clayton copula log-likelihood
    with respect to theta, supporting rotations via copula_code.

    Args:
        y (np.ndarray): Input data of shape (M, 2)
        theta (np.ndarray or float): Copula parameter(s), shape (M,) or scalar
        copula_code (int): Copula family code (3=Clayton, 13/23/33=rotated)

    Returns:
        np.ndarray: First derivative, shape (M,)
    """
    chol = np.asarray(chol).copy()  # <- prevents in-place mutation of caller's array
    sign = np.ones_like(chol)

    y = y.copy()

    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12).reshape(-1, 1)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12).reshape(-1, 1)

    rotation = get_effective_rotation(chol, family_code)

    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1 - u_rot[mask_1]
    v_rot[mask_1] = 1 - v_rot[mask_1]
    sign[mask_1] = 1.0

    mask_2 = rotation == 2
    u_rot[mask_2] = 1 - u_rot[mask_2]
    chol[mask_2] = -chol[mask_2]
    sign[mask_2] = -1.0

    mask_3 = rotation == 3
    v_rot[mask_3] = 1 - v_rot[mask_3]
    chol[mask_3] = -chol[mask_3]
    sign[mask_3] = -1.0

    t4 = np.log(u_rot * v_rot)
    t5 = u_rot ** (-chol)
    t6 = v_rot ** (-chol)
    t7 = t5 + t6 - 1.0
    t8 = np.log(t7)
    t9 = chol**2
    t14 = np.log(u_rot)
    t16 = np.log(v_rot)
    result = (
        1.0 / (1.0 + chol)
        - t4
        + t8 / t9
        + (1.0 / chol + 2.0) * (t5 * t14 + t6 * t16) / t7
    )
    deriv = sign * result

    return deriv.squeeze()


def _derivative_2nd(y, theta, family_code):
    """
    Second derivative of Clayton copula PDF w.r.t. parameter theta.
    Based on diff2PDF_mod from VineCopula C code.

    Args:
        y: array of shape (n, 2) - copula data [u, v]
        theta: array of shape (n,) or scalar - copula parameters
        copula: array of shape (n,) or scalar - copula family codes

    Returns:
        np.ndarray: second derivative values for each observation
    """

    # Constants for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    theta = np.asarray(theta).copy()  # <- prevents in-place mutation of caller's array

    # Extract u and v from the y matrix
    u = np.clip(y[:, 0], UMIN, UMAX).reshape(-1, 1)
    v = np.clip(y[:, 1], UMIN, UMAX).reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)
    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1 - u[mask_1]
    v_rot[mask_1] = 1 - v[mask_1]

    mask_2 = rotation == 2
    v_rot[mask_2] = 1 - v[mask_2]
    theta[mask_2] = -theta[mask_2]

    mask_3 = rotation == 3
    u_rot[mask_3] = 1 - u[mask_3]
    theta[mask_3] = -theta[mask_3]

    # Basic terms (matching C variable names)
    t1 = u_rot * v_rot
    t2 = -theta - 1.0
    t3 = np.power(t1, t2)
    t4 = np.log(t1)

    t6 = np.power(u_rot, -theta)
    t7 = np.power(v_rot, -theta)
    t8 = t6 + t7 - 1.0

    t10 = -2.0 - 1.0 / theta
    t11 = np.power(t8, t10)

    # Higher order terms
    t15 = theta * theta
    t16 = 1.0 / t15
    t17 = np.log(t8)

    t19 = np.log(u_rot)
    t21 = np.log(v_rot)

    t24 = -t6 * t19 - t7 * t21

    t26 = 1.0 / t8
    t27 = t16 * t17 + t10 * t24 * t26

    t30 = -t2 * t3
    t32 = t4 * t4
    t14 = t27 * t27
    t13 = t19 * t19
    t12 = t21 * t21
    t9 = t24 * t24
    t5 = t8 * t8

    # The complete second derivative expression from C code
    term1 = -2.0 * t3 * t4 * t11
    term2 = 2.0 * t3 * t11 * t27
    term3 = t30 * t32 * t11
    term4 = -2.0 * t30 * t4 * t11 * t27
    term5 = t30 * t11 * t14

    # Additional correction terms
    t67 = 2.0 / (t15 * theta)  # Triple derivative term
    t70 = t6 * t13 + t7 * t12  # Second log derivative terms
    t74 = t9 / t5  # Ratio correction

    correction = (
        t30 * t11 * (-t67 * t17 + 2.0 * t16 * t24 * t26 + t10 * t70 * t26 - t10 * t74)
    )

    deriv = term1 + term2 + term3 + term4 + term5 + correction

    return deriv.squeeze()
