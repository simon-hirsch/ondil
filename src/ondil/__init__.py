# ruff: noqa: E402

from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False
HAS_SKTIME = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

if find_spec("sktime") is not None:
    HAS_SKTIME = True

from .information_criteria import InformationCriterion
from .scaler import OnlineScaler

# Import estimators conditionally
try:
    from . import estimators
    if HAS_SKTIME:
        from .estimators import OnlineLinearModelSktime
        __all__ = [
            "OnlineScaler",
            "InformationCriterion",
            "estimators",
            "OnlineLinearModelSktime",
        ]
    else:
        __all__ = [
            "OnlineScaler", 
            "InformationCriterion",
            "estimators",
        ]
except ImportError:
    # If estimators can't be imported (e.g., missing numba), just expose basic functionality
    __all__ = [
        "OnlineScaler",
        "InformationCriterion",
    ]

__version__ = version("ondil")
