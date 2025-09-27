from typing import Dict

import numpy as np
import scipy.linalg as la
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction, MultivariateDistributionMixin
from ..links import Identity, Log, LogShiftTwo, MatrixDiag, MatrixDiagTril
from ..types import ParameterShapes
from .mv_normal_modchol import (
    _deriv1_mu as _deriv1_mu_normal,
    _deriv2_mu as _deriv2_mu_normal,
)

# The MV gaussian distribution with modified Cholesky decomposition
# We use the notation T' D T = PRECISION
# Where T is a lower diagonal matrix and D is a diagonal matrix


class MultivariateStudentTInverseModifiedCholesky(
    MultivariateDistributionMixin, Distribution
):
    """
    Multivariate Student's t distribution with modified Cholesky decomposition.

    The PDF of the multivariate \\( t \\)-distribution with precision matrix parameterized as \\( T^T D T \\) is:
    $$
    p(y \mid \\mu, D, T, \\nu) =
    \\frac{\\Gamma\\left(\\frac{\\nu + k}{2}\\right)}
         {\\Gamma\\left(\\frac{\\nu}{2}\\right) \\left(\\pi \\nu\\right)^{k/2}}
    \\cdot \\sqrt{\\det(D)}
    \\left(1 + \\frac{1}{\\nu} (y - \\mu)^T T^T D T (y - \\mu)\\right)^{-\\frac{\\nu + k}{2}}
    $$
    where \\( k \\) is the dimensionality, \\( \\mu \\) is the location, \\( D \\) is a diagonal matrix, \\( T \\) is a lower triangular matrix, and \\( \\nu \\) is the degrees of freedom.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "mu", 1: "D", 2: "T", 3: "dof"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (-np.inf, np.inf),
        3: (1, np.inf),
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.multivariate_normal
    scipy_names = {"mu": "loc", "sigma": "shape", "nu": "df"}
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.DIAGONAL_MATRIX,
        2: ParameterShapes.LOWER_TRIANGULAR_MATRIX,
        3: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link_1: LinkFunction = MatrixDiag(Log()),
        scale_link_2: LinkFunction = MatrixDiagTril(Identity(), Identity()),
        tail_link: LinkFunction = LogShiftTwo(),
        dof_guesstimate: float = 10,
        use_gaussian_for_location: bool = False,
    ):
        """
        Initializes the distribution with specified link functions and parameters.
        Args:
            loc_link (LinkFunction, optional): Link function for the location parameter. Defaults to Identity().
            scale_link_1 (LinkFunction, optional): Link function for the first scale parameter (diagonal). Defaults to MatrixDiag(Log()).
            scale_link_2 (LinkFunction, optional): Link function for the second scale parameter (lower-triangular). Defaults to MatrixDiagTril(Identity(), Identity()).
            tail_link (LinkFunction, optional): Link function for the tail (degrees of freedom) parameter. Defaults to LogShiftTwo().
            dof_guesstimate (float, optional): Initial guess for the degrees of freedom. Defaults to 10.
            use_gaussian_for_location (bool, optional): Whether to use a Gaussian approximation for the location parameter. Defaults to False.
        Attributes:
            dof_guesstimate (float): Stores the initial guess for the degrees of freedom.
            dof_independence (float): Large value to represent independence in degrees of freedom.
            use_gaussian_for_location (bool): Indicates if a Gaussian is used for the location parameter.
            _regularization_allowed (dict): Specifies which parameters allow regularization.
        """
        super().__init__(
            links={
                0: loc_link,
                1: scale_link_1,
                2: scale_link_2,
                3: tail_link,
            }
        )
        self.dof_guesstimate = dof_guesstimate
        self.dof_independence = 1e6
        self.use_gaussian_for_location = use_gaussian_for_location
        self._regularization_allowed = {0: False, 1: False, 2: True, 3: False}

    def get_adr_regularization_distance(self, dim: int, param: int) -> float:
        if param in (0, 1, 3):
            return None
        if param == 2:
            i, j = np.tril_indices(dim, k=-1)
            return np.abs(i - j)

    def get_regularization_size(self, dim: int) -> float:
        return dim - 1

    @staticmethod
    def fitted_elements(dim: int):
        return {0: dim, 1: dim, 2: int(dim * (dim - 1) // 2), 3: 1}

    @staticmethod
    def index_flat_to_cube(k: int, d: int, param: int):
        if param in (0, 3):
            return k
        if param == 1:
            return k, k
        if param == 2:
            # tril_indicies is row-wise
            # "inverted" triu_indicies is column-wise
            i, j = np.tril_indices(d, k=-1)
            return i[k], j[k]

    def set_theta_element(
        self, theta: Dict, value: np.ndarray, param: int, k: int
    ) -> Dict:
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
        if param in (0, 3):
            theta[param][:, k] = value
        if param == 1:
            theta[param][:, k, k] = value
        if param == 2:
            d = theta[0].shape[1]
            i, j = self.index_flat_to_cube(k, d, param)
            theta[param][:, i, j] = value
        return theta

    def theta_to_params(self, theta):
        mu = theta[0]
        d_mat = theta[1]
        t_mat = theta[2]
        dof = theta[3]
        return mu, d_mat, t_mat, dof

    def set_initial_guess(
        self,
        y: np.ndarray,
        theta: Dict[int, np.ndarray],
        param: int,
    ) -> Dict[int, np.ndarray]:
        if param in (0, 2):
            return theta
        if param == 1:
            M = y.shape[0]
            residual = y - theta[0]
            var = np.var(residual, axis=0)
            shape = np.diag(var * (self.dof_guesstimate - 2) / self.dof_guesstimate)
            _, d_inv, _ = la.ldl(shape, lower=True)
            d_mat = np.linalg.inv(d_inv)
            theta[1] = np.tile(d_mat, (M, 1, 1))
            return theta
        if param == 3:
            theta[3] = np.full_like(theta[3], self.dof_guesstimate)
            return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        mu, d_mat, t_mat, dof = self.theta_to_params(theta)
        if (param == 0) and not self.use_gaussian_for_location:
            deriv = _deriv1_mu(y, mu, d_mat, t_mat, dof, i=k)
        elif (param == 0) and self.use_gaussian_for_location:
            deriv = _deriv1_mu_normal(y, mu, d_mat, t_mat, i=k)
        if param == 1:
            deriv = _deriv1_dmat(y, mu, d_mat, t_mat, dof, i=k)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            deriv = _deriv1_tmat(y, mu, d_mat, t_mat, dof, i=i, j=j)
        if param == 3:
            deriv = _deriv1_dof(y, mu, d_mat, t_mat, dof)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        mu, d_mat, t_mat, dof = self.theta_to_params(theta)
        if (param == 0) and not self.use_gaussian_for_location:
            deriv = _deriv2_mu(y, mu, d_mat, t_mat, dof, i=k)
        if (param == 0) and self.use_gaussian_for_location:
            deriv = _deriv2_mu_normal(y, mu, d_mat, t_mat, i=k)
        if param == 1:
            deriv = _deriv2_dmat(y, mu, d_mat, t_mat, dof, i=k)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            deriv = _deriv2_tmat(y, mu, d_mat, t_mat, dof, i=i, j=j)
        if param == 3:
            deriv = _deriv2_dof(y, mu, d_mat, t_mat, dof)
        return deriv

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param in (0, 3):
            return self.links[param].link(y)
        if param in (1, 2):
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            return self.links[param].element_link(y, i=i, j=j)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param in (0, 3):
            return self.links[param].link_derivative(y)
        if param in (1, 2):
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            return self.links[param].element_derivative(y, i=i, j=j)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param in (0, 3):
            return self.links[param].link_second_derivative(y)
        if param in (1, 2):
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            return self.links[param].element_link_second_derivative(y, i=i, j=j)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param in (0, 3):
            return self.links[param].inverse(y)
        if param in (1, 2):
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            return self.links[param].element_inverse(y, i=i, j=j)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param in (0, 3):
            return self.links[param].inverse_derivative(y)
        if param in (1, 2):
            i, j = self.index_flat_to_cube(k=k, d=y.shape[1], param=param)
            return self.links[param].element_inverse_derivative(y, i=i, j=j)

    def initial_values(self, y, param=0):
        M = y.shape[0]
        mu = np.mean(y, axis=0)
        var = np.var(y, axis=0)
        shape = np.diag(var * (self.dof_guesstimate - 2) / self.dof_guesstimate)
        t_inv, d_inv, _ = la.ldl(shape, lower=True)
        t_mat = np.linalg.inv(t_inv)
        d_mat = np.linalg.inv(d_inv)

        if param == 0:
            return np.tile(mu, (M, 1))
        if param == 1:
            return np.tile(d_mat, (M, 1, 1))
        if param == 2:
            return np.tile(t_mat, (M, 1, 1))
        if param == 3:
            return np.full((M, 1), self.dof_independence)

    def cube_to_flat(self, x: np.ndarray, param: int):
        if param in (0, 3):
            return x
        if param == 1:
            k = x.shape[1]
            return x[:, range(k), range(k)]
        if param == 2:
            k = x.shape[1]
            # column wise indices
            i, j = np.tril_indices(k, k=-1)
            return x[:, i, j]

    def flat_to_cube(self, x: np.ndarray, param: int):
        if param in (0, 3):
            return x
        if param == 1:
            k = x.shape[1]
            out = np.zeros((x.shape[0], k, k))
            out[:, range(k), range(k)] = x
            return out

        if param == 2:
            n, k = x.shape
            # The following conversion holds for upper diagonal matrices
            # We INCLUDE the diagonal!!
            # (D + 1) * D // 2 = k
            # (D + 1) * D = 2 * k
            # D^2 + D = 2 * k
            # ... Wolfram Alpha ...
            # D = 0.5 * (sqrt(8k + 1) - 1)
            d = int(1 / 2 * (np.sqrt(8 * k + 1) - 1))
            i, j = np.tril_indices(d + 1, k=-1)
            out = np.zeros((n, d + 1, d + 1))
            out[:, range(d + 1), range(d + 1)] = 1
            out[:, i, j] = x
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
        # fitted_theta = {**theta, param: fitted_eta}
        return self.logpdf(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        mu, d_mat, t_mat, dof = self.theta_to_params(theta)
        cov = np.linalg.inv(t_mat.swapaxes(-1, -2) @ d_mat @ t_mat)
        out = {
            "loc": mu,
            "shape": cov,
            "df": dof,
        }
        return out

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        return _loglikelihood_t_modchol(y, theta[0], theta[1], theta[2], theta[3])

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")


def _loglikelihood_t_modchol(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
) -> Dict[str, float]:
    k = y.shape[1]
    y_centered = y - mu
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    A = np.squeeze(sp.gammaln((dof + k) / 2))
    B = np.squeeze(sp.gammaln(dof / 2))
    C = 1 / 2 * k * np.squeeze(np.log(np.pi * dof))
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)
    part1 = A - B - C
    # part2 = 1 / 2 * np.log(np.linalg.det(precision))
    part2 = 1 / 2 * np.log(np.linalg.det(D_inv))
    part3 = np.squeeze((dof + k) / 2) * np.log((1 + np.squeeze((1 / dof)) * Z))
    return part1 + part2 - part3


def _deriv1_mu(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
    i: int = 0,
) -> np.ndarray:
    k = y.shape[1]
    y_centered = y - mu
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)
    part1 = (k + dof.squeeze()) / (2 * (Z + dof.squeeze()))
    part2 = 2 * np.sum(precision[:, i, :] * y_centered, axis=1)
    return part1 * part2


def _deriv2_mu(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
    i: int = 0,
) -> np.ndarray:
    k = y.shape[1]
    y_centered = y - mu
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)
    deriv_1 = 2 * np.sum(precision[:, i, :] * y_centered, axis=1)
    deriv_2 = 2 * precision[:, i, i]
    part1 = k + dof.squeeze()
    part2 = (Z + dof.squeeze()) * deriv_2 - deriv_1**2
    part3 = 2 * (Z + dof.squeeze()) ** 2
    return -(part1 * part2) / part3


def _deriv1_dmat(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
    i: int = 0,
) -> np.ndarray:
    k = y.shape[1]
    y_centered = y - mu
    Ty = (T_mat @ y_centered[..., None]).squeeze(-1)
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)

    deriv_logdet = 1 / 2 * 1 / D_inv[:, i, i]

    part2 = np.squeeze(k + dof) / (2 * (Z + dof.squeeze()))
    part3 = Ty[:, i] ** 2
    deriv_Z = part2 * part3
    return deriv_logdet - deriv_Z


def _deriv2_dmat(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
    i: int = 0,
) -> np.ndarray:
    k = y.shape[1]
    y_centered = y - mu
    Ty = (T_mat @ y_centered[..., None]).squeeze(-1)
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)

    # Derivative of logdet part
    part1 = -1 / 2 * 1 / D_inv[:, i, i] ** 2

    # Derivatives of Z
    deriv1 = Ty[:, i] ** 2
    deriv2 = 0

    part2 = k + dof.squeeze()
    part3 = (Z + dof.squeeze()) * deriv2 - deriv1**2
    part4 = 2 * (Z + dof.squeeze()) ** 2
    return part1 - (part2 * part3) / part4


def _deriv1_tmat(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
    i: int = 0,
    j: int = 0,
) -> np.ndarray:
    k = y.shape[1]
    y_centered = y - mu
    Ty = (T_mat @ y_centered[..., None]).squeeze(-1)
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)

    part1 = 0  # logdet part
    part2 = np.squeeze(k + dof) / (2 * (Z + dof.squeeze()))
    part3 = 2 * D_inv[:, i, i] * Ty[:, i] * y_centered[:, j]

    return part1 - part2 * part3


def _deriv2_tmat(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
    i: int = 0,
    j: int = 0,
) -> np.ndarray:
    k = y.shape[1]
    y_centered = y - mu
    Ty = (T_mat @ y_centered[..., None]).squeeze(-1)
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)

    part1 = 0  # Derivative of logdet part
    # first and second
    # deriv_Z = (-1 / 2) * 2 * D_inv[:, i, i] * Ty[:, i] * y_centered[:, j]
    # deriv_Z = (-1 / 2) * 2 * D_inv[:, i, i] * y_centered[:, j] ** 2
    deriv1 = 2 * D_inv[:, i, i] * Ty[:, i] * y_centered[:, j]
    deriv2 = 2 * D_inv[:, i, i] * y_centered[:, j] ** 2

    part2 = k + dof.squeeze()
    part3 = (Z + dof.squeeze()) * deriv2 - deriv1**2
    part4 = 2 * (Z + dof.squeeze()) ** 2
    return part1 - (part2 * part3) / part4


def _deriv1_dof(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
):
    k = y.shape[1]
    y_centered = y - mu
    dof = np.squeeze(dof)
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)

    part1 = -(-dof * sp.digamma((k + dof) / 2) + k + dof * sp.digamma(dof / 2)) / (
        2 * dof
    )
    part2 = (Z * (k + dof)) / (2 * dof * (dof + Z)) - 1 / 2 * np.log((dof + Z) / dof)
    return part1 + part2


def _deriv2_dof(
    y: np.ndarray,
    mu: np.ndarray,
    D_inv: np.ndarray,
    T_mat: np.ndarray,
    dof: np.ndarray,
):
    k = y.shape[1]
    y_centered = y - mu
    dof = np.squeeze(dof)
    precision = np.swapaxes(T_mat, -2, -1) @ D_inv @ T_mat
    Z = np.sum(y_centered * (precision @ y_centered[..., None]).squeeze(), 1)
    part1 = (
        1
        / 4
        * (
            (2 * k) / (dof**2)
            + sp.polygamma(1, (dof + k) / 2)
            - sp.polygamma(1, dof / 2)
        )
    )
    part2 = (Z * (dof * Z - k * (2 * dof + Z))) / (2 * dof**2 * (dof + Z) ** 2)
    return part1 + part2
