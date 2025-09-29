# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st


from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import  FisherZLink, KendallsTauToParameter
from ..types import ParameterShapes


class BivariateCopulaStudentT(CopulaMixin, Distribution):

    corresponding_gamlss: str = None
    parameter_names = {0: "rho", 1: "nu"}
    parameter_support = {0: (-1, 1), 1: (2, np.inf)}
    distribution_support = (-1, 1) 
    n_params = len(parameter_names)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    def __init__(
        self, 
        link_1: LinkFunction = FisherZLink(), 
        link_2: LinkFunction = FisherZLink(), 
        param_link_1: LinkFunction = KendallsTauToParameter(),
        param_link_2: LinkFunction = KendallsTauToParameter(),
    ):
        super().__init__(
            links={0: link_1, 1: link_2},
            param_links={0: param_link_1, 1: param_link_2},
        )
        self.is_multivariate = True
        self._adr_lower_diag = {0: False, 1: False}
        self._regularization_allowed = {0: False, 1: False}
        self._regularization = ""  # or adr
        self._scoring = "fisher"


    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2), 1: 1} 
    
    @property
    def param_structure(self):
        return self._param_structure


    @staticmethod
    def set_theta_element(theta: Dict, value: np.ndarray, param: int, k: int) -> Dict:
        """Sets an element of theta for parameter param and place k.

        !!! Note
            This will mutate `theta`!

        Args:
            theta (Dict): Current fitted $\theta$
            value (np.ndarray): Value to set
            param (int): Distribution parameter
            k (int): Flat element index $k$

        Returns:
            Dict: Theta where element (param, k) is set to value.
        """
        theta[param] = value
        return theta
    
  
    
    def theta_to_params(self, theta):
        if len(theta) > 1:
            return theta[0], theta[1]  # rho, nu
        else:
            return theta[0], 4.0  # default nu = 4

    
    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters

        Returns:
            dict: Dict of parameters for scipy.stats.t
        """
        rho = theta[:, 0]
        nu = theta[:, 1] if theta.shape[1] > 1 else np.full_like(rho, 4.0)
        params = {"rho": rho, "df": nu}
        return params

    def set_initial_guess(self, theta, param):
        return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with parameters
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 1st derivatives.
        """
        rho, nu = self.theta_to_params(theta)

        if param == 0:  # derivative wrt rho
            deriv = _derivative_1st_rho(y=y, rho=rho, nu=nu)
        else:  # derivative wrt nu
            deriv = _derivative_1st_nu(y=y, rho=rho, nu=nu)

        return deriv

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Return the second derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with parameters
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 2nd derivatives.
        """
        rho, nu = self.theta_to_params(theta)

        if param == 0:  # derivative wrt rho
            deriv = _derivative_2nd_rho(y=y, rho=rho, nu=nu)
        else:  # derivative wrt nu
            deriv = _derivative_2nd_nu(y=y, rho=rho, nu=nu)
        
        return deriv

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        rho, nu = self.theta_to_params(theta)
        
        if param == 0:
            deriv = _derivative_1st_rho(y, rho, nu)
        else:
            deriv = _derivative_1st_nu(y, rho, nu)
        return deriv

    def element_dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        rho, nu = self.theta_to_params(theta)
        
        if param == 0:
            deriv = _derivative_2nd_rho(y, rho, nu)
        else:
            deriv = _derivative_2nd_nu(y, rho, nu)
        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")


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

    def initial_values(self, y, param=0):
        M = y.shape[0]
        if param == 0:  # rho
            tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
            rho = np.full((M, 1), tau)
            return rho
        else:  # nu
            nu = np.full((M, 1), 4.0)  # default degrees of freedom
            return nu
    
    def cube_to_flat(self, x: np.ndarray, param: int):
        out = x
        return out

    def flat_to_cube(self, x: np.ndarray, param: int):
        out = x
        return out

    def param_conditional_likelihood(
        self, y: np.ndarray, theta: Dict, eta: np.ndarray, param: int
    ) -> np.ndarray:
        """Calulate the log-likelihood for (flat) eta for parameter (param)
        and theta for all other parameters.

        Args:
            y (np.ndarray): True values
            theta (Dict): Fitted theta.
            eta (np.ndarray): Fitted eta.
            param (int): Param for which we take eta.

        Returns:
            np.ndarray: Log-likelihood.
        """
        fitted = self.flat_to_cube(eta, param=param)
        fitted = self.link_inverse(fitted, param=param)
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        rho, nu = self.theta_to_params(theta)
        out = {
            "rho": rho,
            "df": nu,
        }
        return out


    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        """
        Generate random samples from the bivariate t copula.

        Args:
            size (int): Number of samples to generate.
            theta (dict or tuple): (rho, nu) parameters.

        Returns:
            np.ndarray: Samples of shape (size, 2) in (0, 1).
        """
        rho, nu = self.theta_to_params(theta)
        
        # Generate from bivariate t distribution
        z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)
        w = np.random.gamma(nu/2, 2/nu, size=size)
        
        # Scale by chi-squared random variable
        t_samples = z / np.sqrt(w[:, np.newaxis])
        
        # Transform to uniform marginals using the t CDF
        u = st.t.cdf(t_samples[:, 0], df=nu)
        v = st.t.cdf(t_samples[:, 1], df=nu)
        
        return np.column_stack((u, v))


    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        rho, nu = self.theta_to_params(theta)
        return np.log(_log_likelihood_t(y, rho, nu))

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")

    def hfunc(self, u: np.ndarray, v: np.ndarray, theta: Dict, un: int) -> np.ndarray:
        """
        Conditional distribution function h(u|v) for the bivariate t copula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (Dict): Parameters {0: rho, 1: nu}.
            un (int): Conditioning variable (1 or 2).

        Returns:
            np.ndarray: Array of shape (n,) with conditional probabilities.
        """
        rho, nu = self.theta_to_params(theta)
        M = u.shape[0]
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        u = np.clip(u, UMIN, UMAX)
        v = np.clip(v, UMIN, UMAX)

        # Swap u and v if un == 2
        if un == 2:
            u, v = v, u

        # Handle edge cases
        h = np.where((v == 0) | (u == 0), 0, np.nan)
        h = np.where(v == 1, u, h)
        
        qt_u = st.t.ppf(u, df=nu)
        qt_v = st.t.ppf(v, df=nu)

        for m in range(M):
            rho_m = rho[m] if hasattr(rho, '__len__') else rho
            nu_m = nu[m] if hasattr(nu, '__len__') else nu
            
            denom = np.sqrt((nu_m + qt_v[m]**2) * (1 - rho_m**2) / (nu_m + 1))
            x = (qt_u[m] - rho_m * qt_v[m]) / denom
            
            if np.isfinite(x):
                h[m] = st.t.cdf(x, df=nu_m + 1)
            elif (qt_u[m] - rho_m * qt_v[m]) < 0:
                h[m] = 0
            else:
                h[m] = 1

        # Clip output for numerical stability
        h = np.clip(h, UMIN, UMAX)
        return h

##########################################################
### Functions for the Student-t copula derivatives #####
##########################################################


def _log_likelihood_t(y, rho, nu):
    """Log-likelihood for bivariate t copula"""
    M = y.shape[0]
    f = np.empty(M)
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    
    u = st.t.ppf(y_clipped[:, 0], df=nu)
    v = st.t.ppf(y_clipped[:, 1], df=nu)
    
    for m in range(M):
        rho_m = rho[m] if hasattr(rho, '__len__') else rho
        nu_m = nu[m] if hasattr(nu, '__len__') else nu
        
        t1 = u[m]
        t2 = v[m]
        
        # Bivariate t copula density
        R = np.array([[1, rho_m], [rho_m, 1]])
        R_inv = np.linalg.inv(R)
        det_R = np.linalg.det(R)
        
        marginal1 = st.t.pdf(t1, df=nu_m)
        marginal2 = st.t.pdf(t2, df=nu_m)
        
        quad_form = np.array([t1, t2]) @ R_inv @ np.array([t1, t2])
        
        joint = (
            st.gamma((nu_m + 2) / 2) * st.gamma(nu_m / 2) /
            (st.gamma((nu_m + 1) / 2)**2 * np.sqrt(det_R)) *
            (1 + quad_form / nu_m)**(-(nu_m + 2) / 2) /
            ((1 + t1**2 / nu_m)**(-(nu_m + 1) / 2) * (1 + t2**2 / nu_m)**(-(nu_m + 1) / 2))
        )
        
        f[m] = joint / (marginal1 * marginal2)
    
    f[f <= 0] = 1e-16
    return f

def _derivative_1st_rho(y, rho, nu):
    """First derivative wrt rho for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    
    u = st.t.ppf(y[:, 0], df=nu)
    v = st.t.ppf(y[:, 1], df=nu)
    
    for m in range(M):
        rho_m = rho[m] if hasattr(rho, '__len__') else rho
        nu_m = nu[m] if hasattr(nu, '__len__') else nu
        
        t1, t2 = u[m], v[m]
        
        # Numerical derivative (simplified)
        quad_form = (t1**2 + t2**2 - 2*rho_m*t1*t2) / (1 - rho_m**2)
        
        deriv[m] = (
            (nu_m + 2) * (t1*t2*(1-rho_m**2) - rho_m*(t1**2 + t2**2 - 2*rho_m*t1*t2)) /
            ((1 - rho_m**2)**2 * (nu_m + quad_form))
        ) - rho_m / (1 - rho_m**2)
    
    return deriv.squeeze()

def _derivative_1st_nu(y, rho, nu):
    """First derivative wrt nu for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    
    # This is a complex derivative - simplified approximation
    for m in range(M):
        # Numerical approximation or simplified analytical form
        deriv[m] = 0.0  # Placeholder - implement proper derivative
    
    return deriv.squeeze()

def _derivative_2nd_rho(y, rho, nu):
    """Second derivative wrt rho for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    
    # Simplified second derivative - implement full analytical form
    for m in range(M):
        deriv[m] = -1.0  # Placeholder
    
    return deriv.squeeze()

def _derivative_2nd_nu(y, rho, nu):
    """Second derivative wrt nu for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    
    # Simplified second derivative - implement full analytical form
    for m in range(M):
        deriv[m] = -1.0  # Placeholder
    
    return deriv.squeeze()