# Author: Your Name
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import LogShiftValue,Log, KendallsTauToParameterClayton,Identity
from ..types import ParameterShapes

class BivariateCopulaClayton(CopulaMixin, Distribution):
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
        family_code: int = 301,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
            rotation=0
        )
        self.family_code = family_code
        self.is_multivariate = True
        self._adr_lower_diag = {0: False}
        self._regularization_allowed = {0: False}
        self._regularization = ""
        self._scoring = "fisher"

    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2)}

    @property
    def param_structure(self):
        return self._param_structure

    @staticmethod
    def set_theta_element(theta: dict, value: np.ndarray, param: int, k: int) -> dict:
        theta[param] = value
        return theta

    def theta_to_params(self, theta):
        if isinstance(theta, dict):
            theta = theta[0]
        theta_array = np.asarray(theta)
        sign = np.sign(theta_array)
        abs_theta = np.abs(theta_array)
        abs_theta_clipped = np.clip(abs_theta, 1e-6, 200)  # Match R bounds
        return sign * abs_theta_clipped

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        return {"theta": theta}

    def set_initial_guess(self, theta, param):
        return theta

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Pearson correlation for each sample
        # y is expected to be (M, 2)
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        chol = np.full((M, 1), tau)
        return chol.reshape(-1)

    def cube_to_flat(self, x: np.ndarray, param: int):
        return x

    def flat_to_cube(self, x: np.ndarray, param: int):
        return x

    def param_conditional_likelihood(
        self, y: np.ndarray, theta: dict, eta: np.ndarray, param: int
    ) -> np.ndarray:
        fitted = self.flat_to_cube(eta, param=param)
        fitted = self.link_inverse(fitted, param=param)
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: dict):
        return {"theta": theta}

    def logpdf(self, y, theta):
        theta = self.theta_to_params(theta)
        result = _clayton_logpdf(y, theta, self.family_code)
        #return np.asarray(result).reshape(-1)  # Always 1D
        return result
    
    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def dl1_dp1(self, y, theta, param=0):
        theta = self.theta_to_params(theta)
        return _clayton_derivative_1st(y, theta, self.family_code)

    def dl2_dp2(self, y, theta, param=0, clip=False):
        """
        Second derivative with proper chain rule matching gamBiCopFit.R
        """
        theta = self.theta_to_params(theta)
        return _clayton_derivative_2nd(y, theta, self.family_code)
        

    def element_score(self, y, theta, param=0, k=0):
        return self.element_dl1_dp1(y, theta, param, k)

    def element_hessian(self, y, theta, param=0, k=0):
        return self.element_dl2_dp2(y, theta, param, k)

    def element_dl1_dp1(self, y, theta, param=0, k=0, clip=False):
    # Apply proper chain rule like dl1_dp1
        theta_params = self.theta_to_params(theta)
        raw_d1 = _clayton_derivative_1st(y, theta = theta_params, family_code=self.family_code)
        
        # Apply link derivative
        # eta = self.links[param].link(theta_params)
        # dpar_deta = self.links[param].link_derivative(eta)
        
        return raw_d1 

    def element_dl2_dp2(self, y, theta, param=0, k=0, clip=False):
        # Apply proper chain rule like dl2_dp2
        theta_params = self.theta_to_params(theta)

        raw_d2 = _clayton_derivative_2nd(y, theta_params, self.family_code)
        return raw_d2

    def dl2_dpp(self, y, theta, param=0):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_link(y)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_derivative(y)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_link_second_derivative(y)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].inverse(y)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_inverse_derivative(y)

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: dict
    ) -> dict:
        raise NotImplementedError("Not implemented for Clayton copula.")

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def rvs(self, size, theta):
        theta = self.theta_to_params(theta)
        u = np.random.uniform(size=size)
        w = np.random.uniform(size=size)
        v = (w ** (-theta / (1 + theta)) * (u ** (-theta) - 1) + 1) ** (-1 / theta)
        return np.column_stack((u, v))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def get_effective_rotation(theta_value, family_code):
        """
        Get the effective rotation based on family code and parameter sign.
        This mimics the gamCopula logic from getFams() and bicoppd1d2().
        
        Args:
            theta_value: The copula parameter value
            
        Returns:
            int: The effective rotation (0, 1, 2, 3)
        """
        # Map gamCopula family codes to VineCopula rotations
        # Based on getFams() function from utilsFamilies.R
        if family_code == 301:
            # Double Clayton type I (standard and rotated 90 degrees)
            if theta_value > 0:
                return 0  # Standard Clayton (3)
            else:
                return 2  # 90° rotation (23)
        elif family_code == 302:
            # Double Clayton type II (standard and rotated 270 degrees)
            if theta_value > 0:
                return 0  # Standard Clayton (3)
            else:
                return 3  # 270° rotation (33)
        elif family_code == 303:
            # Double Clayton type III (survival and rotated 90 degrees)
            if theta_value > 0:
                return 1  # 180° rotation (13) - survival
            else:
                return 2  # 90° rotation (23)
        elif family_code == 304:
            # Double Clayton type IV (survival and rotated 270 degrees)
            if theta_value > 0:
                return 1  # 180° rotation (13) - survival
            else:
                return 3  # 270° rotation (33)
        else:
            # Default to 301 behavior if invalid family code
            raise ValueError(f"Unsupported family code: {family_code}. Supported codes are 301, 302, 303, 304.")

    def hfunc(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int) -> np.ndarray:
        M = u.shape[0]
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        u = np.clip(u, UMIN, UMAX)
        v = np.clip(v, UMIN, UMAX)

        # Swap u and v if un == 2
        if un == 2:
            u, v = v, u

        h = np.empty(M)
        for m in range(M):
            th = theta[m] if hasattr(theta, "__len__") else theta

            rotation = get_effective_rotation(th, family_code)

            # Apply rotation transformations to data
            u_rot, v_rot = u[m], v[m]
            if rotation == 1:  # 180° rotation (survival)
                u_rot = 1 - u[m]
                v_rot = 1 - v[m]
            elif rotation == 2:  # 90° rotation
                v_rot = 1 - v[m]
                th = -th
            elif rotation == 3:  # 270° rotation
                # u_rot stays the same
                v_rot = 1 - v[m]


            # Conditional distribution function for Clayton copula
            # h(v|u) = ∂C(u,v)/∂v = (u^{-θ-1}) * (u^{-θ} + v^{-θ} - 1)^{-1/θ - 1}
            t1 = v_rot ** (-th - 1)
            t2 = v_rot ** (-th) + u_rot ** (-th) - 1
            t2 = np.maximum(t2, UMIN)
            t3 = -1.0 / th - 1.0
            h[m] = t1 * (t2 ** t3)
        h = np.clip(h, UMIN, UMAX)
        return h
    
    def get_regularization_size(self, dim: int) -> int:
        return dim
    
def get_effective_rotation(theta_value, family_code):
    """
    Get the effective rotation based on family code and parameter sign.
    This mimics the gamCopula logic from getFams() and bicoppd1d2().
    
    Args:
        theta_value: The copula parameter value
        
    Returns:
        int: The effective rotation (0, 1, 2, 3)
    """
    # Map gamCopula family codes to VineCopula rotations
    # Based on getFams() function from utilsFamilies.R
    if family_code == 301:
        # Double Clayton type I (standard and rotated 90 degrees)
        if theta_value > 0:
            return 0  # Standard Clayton (3)
        else:
            return 2  # 90° rotation (23)
    elif family_code == 302:
        # Double Clayton type II (standard and rotated 270 degrees)
        if theta_value > 0:
            return 0  # Standard Clayton (3)
        else:
            return 3  # 270° rotation (33)
    elif family_code == 303:
        # Double Clayton type III (survival and rotated 90 degrees)
        if theta_value > 0:
            return 1  # 180° rotation (13) - survival
        else:
            return 2  # 90° rotation (23)
    elif family_code == 304:
        # Double Clayton type IV (survival and rotated 270 degrees)
        if theta_value > 0:
            return 1  # 180° rotation (13) - survival
        else:
            return 3  # 270° rotation (33)
    else:
        # Default to 301 behavior if invalid family code
        raise ValueError(f"Unsupported family code: {family_code}. Supported codes are 301, 302, 303, 304.")

##########################################################
# Helper functions for the log-likelihood and derivatives #
##########################################################

def _clayton_logpdf(y, theta, family_code=None):
    M = y.shape[0]
    f = np.empty(M)
    u = y[:, 0]
    v = y[:, 1]
    
    # Handle edge cases like in C code
    
    u_valid = u if hasattr(u, '__len__') else u
    v_valid = v if hasattr(v, '__len__') else v


    for m in range(M):
        if M == 1:
            theta_valid = theta
        else: 
            theta_valid = theta[m]
        
        rotation = get_effective_rotation(theta_valid, family_code)

        # Apply rotation transformations to data
        u_rot, v_rot = u_valid[m], v_valid[m]
        if rotation == 1:  # 180° rotation (survival)
            u_rot = 1 - u_valid[m]
            v_rot = 1 - v_valid[m]
        elif rotation == 2:  # 90° rotation
            u_rot = 1 - u_valid[m]
            theta_valid = -theta_valid
        elif rotation == 3:  # 270° rotation
            # u_rot stays the same
            v_rot = 1 - v_valid[m]

        if theta_valid == 0:
            f[m] = 0
        elif theta_valid < 1e-10:
            f[m] = 0
        else: 
            t5 = u_rot ** (-theta_valid)
            t6 = v_rot ** (-theta_valid)
            t7 = t5 + t6 - 1.0
            t7 = np.maximum(t7, 1e-12)
            
            f[m] = (
                + np.log1p(theta_valid)  # log1p instead of log(theta + 1)
                            - (1.0 + theta_valid) * np.log(np.maximum(u_rot * v_rot, 1e-12))  # log(u*v) with numerical protection
                - (2.0 + 1.0 / theta_valid) * np.log(t7)
            )
            
            # Apply bounds like C code
            XINFMAX = 700  # Approximate value for log overflow protection
            DBL_MIN = 2.2e-308
            f[m] = np.where(f[m] > XINFMAX, XINFMAX, f[m])
            f[m] = np.where(f[m] < np.log(DBL_MIN), np.log(DBL_MIN), f[m])
            

    return f  # Always 1D

def _clayton_derivative_1st(y, theta, family_code):
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
    M = y.shape[0]
    deriv = np.empty((M,), dtype=np.float64)
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)

    for m in range(M):
        th = theta[m] if hasattr(theta, "__len__") else theta

        rotation = get_effective_rotation(th, family_code)

        # Handle rotations
        if rotation == 0:  # Standard Clayton
            uu, vv = u[m], v[m]
            sign = 1
        elif rotation == 1:  # 180° rotated Clayton (survival)
            uu, vv, tth = 1 - u[m], 1 - v[m], abs(th)
            sign = 1.0 if th >= 0 else -1.0
        elif rotation == 2:  # 90° rotated Clayton
            uu, vv = 1 - u[m], v[m]
            th = -th
            sign = -1.0 
        elif rotation == 3:  # 270° rotated Clayton
            uu, vv, tth = u[m], 1 - u[m], abs(th)
            sign = -1.0 if th >= 0 else 1.0
        else:
            raise NotImplementedError("Copula family not implemented.")

        t4 = np.log(uu * vv)
        t5 = uu ** (-th)
        t6 = vv ** (-th)
        t7 = t5 + t6 - 1.0
        t8 = np.log(t7)
        t9 = th ** 2
        t14 = np.log(uu)
        t16 = np.log(vv)
        result = 1.0 / (1.0 + th) - t4 + t8 / t9 + (1.0 / th + 2.0) * (t5 * t14 + t6 * t16) / t7
        deriv[m] = sign*result

    return deriv

def _clayton_derivative_2nd(y, theta, family_code):
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
    
    # Extract u and v from the y matrix
    u = np.clip(y[:, 0], UMIN, UMAX)
    v = np.clip(y[:, 1], UMIN, UMAX)
    
    # Ensure arrays - FIX: Handle scalar theta properly
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    
    # Fix the theta handling
    if np.isscalar(theta):
        theta = np.full(len(u), theta)
    else:
        theta = np.atleast_1d(theta)
    
    n = len(u)
    out = np.zeros(n)
    
    for i in range(n):
        theta_i = theta[i]  # Now this will work since theta is always an array
        rotation = get_effective_rotation(theta_i, family_code)
        
        if rotation == 0:  # Standard Clayton
            uu, vv = u[i], v[i]
            th = theta_i
        elif rotation == 1:  # 180° rotated Clayton (survival)
            uu, vv, tth = 1 - u[m], 1 - v[m], abs(th)
            sign = 1.0 if th >= 0 else -1.0
        elif rotation == 2:  # 90° rotated Clayton
            uu, vv = u[i], 1 - v[i]
            th = -theta_i
        elif rotation == 3:  # 270° rotated Clayton
            uu, vv, tth = u[m], 1 - u[m], abs(th)
            sign = -1.0 if th >= 0 else 1.0
        else:
            raise NotImplementedError("Copula family not implemented.")
        
     
        # Basic terms (matching C variable names)
        t1 = uu * vv
        t2 = -th - 1.0
        t3 = np.power(t1, t2)
        t4 = np.log(t1)
        
        t6 = np.power(uu, -th)
        t7 = np.power(vv, -th)
        t8 = t6 + t7 - 1.0
        
        t10 = -2.0 - 1.0/th
        t11 = np.power(t8, t10)
        
        # Higher order terms
        t15 = th * th
        t16 = 1.0 / t15
        t17 = np.log(t8)
        
        t19 = np.log(uu)
        t21 = np.log(vv)
        
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
        t67 = 2.0 / (t15 * th)  # Triple derivative term
        t70 = t6 * t13 + t7 * t12  # Second log derivative terms
        t74 = t9 / t5  # Ratio correction
        
        correction = t30 * t11 * (-t67 * t17 + 2.0 * t16 * t24 * t26 + 
                                    t10 * t70 * t26 - t10 * t74)
        
        result = term1 + term2 + term3 + term4 + term5 + correction
        out[i] = result

    
    return out