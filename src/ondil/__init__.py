# ruff: noqa: E402

from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

from . import (
    base,
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
