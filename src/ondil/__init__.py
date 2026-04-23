from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False
HAS_MPL = False
HAS_SCORINGRULES = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

if find_spec("matplotlib") is not None:
    HAS_MPL = True

if find_spec("scoringrules") is not None:
    HAS_SCORINGRULES = True

from . import (  # noqa: E402, F401
    base,
    diagnostics,
    distributions,
    error,
    estimators,
    incremental_statistics,
    information_criteria,
    links,
    logging,
    methods,
    pipeline,
    scaler,
    terms,
    utils,
    warnings,
)

logging.set_log_level("INFO")

__version__ = version("ondil")

__all__ = [
    "base",
    "terms",
    "diagnostics",
    "distributions",
    "error",
    "estimators",
    "information_criteria",
    "links",
    "methods",
    "pipeline",
    "scaler",
    "utils",
    "warnings",
    "incremental_statistics",
]
