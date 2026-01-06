# %%
import numpy as np
import pytest
import rpy2.robjects as robjects

from .utils import get_distributions_with_gamlss

N = 100
CLIP_BOUNDS = (-1e5, 1e5)

SPECIAL_BOUNDS_DISTRIBUTIONS = {"PowerExponential": (-1e2, 1e2)}


@pytest.mark.parametrize(
    "distribution",
    get_distributions_with_gamlss(),
    ids=lambda dist: dist.__class__.__name__,
)
def test_distribution_derivatives(distribution):
    """Test that Python derivatives match R GAMLSS derivatives for different distributions."""

    # Set seed for reproducibility
    np.random.seed(42)

    clip_bounds = SPECIAL_BOUNDS_DISTRIBUTIONS.get(
        distribution.__class__.__name__,
        CLIP_BOUNDS,  # default np.allclose tolerance
    )

    # Generate random data within distribution support
    y = np.random.uniform(
        np.clip(distribution.distribution_support[0], *clip_bounds),
        np.clip(distribution.distribution_support[1], *clip_bounds),
        N,
    )
    robjects.globalenv["y"] = robjects.FloatVector(y)

    # Generate random parameters within support bounds
    theta = np.array([
        np.random.uniform(
            np.clip(distribution.parameter_support[i][0], *clip_bounds),
            np.clip(distribution.parameter_support[i][1], *clip_bounds),
            N,
        )
        for i in range(distribution.n_params)
    ]).T

    # Assign R variables programmatically
    for i, param_name in distribution.parameter_names.items():
        robjects.globalenv[param_name] = robjects.FloatVector(theta[:, i])

    # Obtain derivatives from R
    code = f"""
        dist <- gamlss.dist::{distribution.corresponding_gamlss}()
        # all_derivative_funcs <- c("dldm", "dldd", "d2ldm2", "d2ldd2", "d2ldmdd")
        available_funcs <- names(dist)[startsWith(names(dist), "d2l") | startsWith(names(dist), "dl")]
        setNames(lapply(available_funcs, function(f) {{
          do.call(dist[[f]], mget(names(formals(dist[[f]])), envir = .GlobalEnv))
        }}), available_funcs)
        """

    R_list = robjects.r(code)

    # Map R derivative names to Python method calls
    # First derivatives
    derivative_mapping = {
        "dldm": lambda: distribution.dl1_dp1(y, theta=theta, param=0),
        "dldd": lambda: distribution.dl1_dp1(y, theta=theta, param=1),
        "dldv": lambda: distribution.dl1_dp1(y, theta=theta, param=2),
        "dldt": lambda: distribution.dl1_dp1(y, theta=theta, param=3),
    }

    # Second derivatives (diagonal)
    for i in range(distribution.n_params):
        param_names = ["m", "d", "v", "t"]
        if i < len(param_names):
            derivative_mapping[f"d2ld{param_names[i]}2"] = lambda i=i: (
                distribution.dl2_dp2(y, theta=theta, param=i)
            )

    # Second derivatives (cross terms)
    param_names = ["m", "d", "v", "t"]
    cross_derivatives = [
        ("m", "d", 0, 1),
        ("m", "v", 0, 2),
        ("m", "t", 0, 3),
        ("d", "v", 1, 2),
        ("d", "t", 1, 3),
        ("v", "t", 2, 3),
    ]

    for name1, name2, idx1, idx2 in cross_derivatives:
        if idx1 < distribution.n_params and idx2 < distribution.n_params:
            derivative_mapping[f"d2ld{name1}d{name2}"] = lambda idx1=idx1, idx2=idx2: (
                distribution.dl2_dpp(y, theta=theta, params=(idx1, idx2))
            )

    # Compare R and Python derivatives - only for available derivatives
    available_derivatives = R_list.names
    for key in available_derivatives:
        if key in derivative_mapping:
            assert np.allclose(derivative_mapping[key](), R_list.rx2(key)), (
                f"Derivative {key} doesn't match for {distribution.__class__.__name__}"
            )
