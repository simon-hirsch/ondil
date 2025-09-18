# ruff: noqa: E402

from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False
HAS_MPL = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

if find_spec("matplotlib") is not None:
    HAS_MPL = True

from . import (
    base,
    diagnostics,
    distributions,
    error,
    estimators,
    information_criteria,
    links,
    methods,
    scaler,
    utils,
    warnings,
)

__version__ = version("ondil")

__all__ = [
    "base",
    "diagnostics",
    "error",
    "information_criteria",
    "links",
    "methods",
    "distributions",
    "estimators",
    "utils",
    "scaler",
    "warnings",
]
