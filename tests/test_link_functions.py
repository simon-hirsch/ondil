# Author: Simon Hirsch
# Licence: MIT

import numpy as np
import pytest

from ondil.links import (
    Identity,
    InverseSoftPlus,
    InverseSoftPlusShiftValue,
    LogIdent,
    Logit,
    Log,
    LogShiftValue,
    Sqrt,
    SqrtShiftValue,
)

# We don't test
# - LogShiftTwo
# - SqrtShiftTwo
# at the moment since they derive from the ShiftValue

REAL_LINE_LINKS = [Identity]
POSITIVE_LINE_LINKS = [Log, Sqrt, LogIdent, InverseSoftPlus]
POSITIVE_RESTRICTED_LINE_LINKS = [Logit]  ##domain (0,1)
SHIFTED_LINKS = [LogShiftValue, SqrtShiftValue, InverseSoftPlusShiftValue]
VALUES = np.array([2, 5, 10, 25, 100])
M = 10000


@pytest.mark.parametrize("linkfun", REAL_LINE_LINKS)
def test_link_real_line(linkfun):
    """Test links that are defined on the real line (-inf, inf)"""
    instance = linkfun()
    x = np.linspace(-100, 100, M)
    y = instance.inverse(instance.link(x))
    print(y)
    assert np.allclose(x, y), "Links don't match"


@pytest.mark.parametrize("linkfun", [Logit])
def test_link_zero_one_domain(linkfun):
    """Test links that are defined on the (0, 1) domain"""
    instance = linkfun()
    # Avoid exact 0 and 1 to stay within the domain
    x = np.linspace(1e-10, 1 - 1e-10, M)
    y = instance.inverse(instance.link(x))
    assert np.allclose(x, y), "Links don't match"


@pytest.mark.parametrize("linkfun", POSITIVE_LINE_LINKS)
def test_link_positive_line(linkfun):
    """Test links that are defined on the positive real line (eps, inf)"""
    instance = linkfun()
    x = np.linspace(1e-10, 100, M)
    y = instance.inverse(instance.link(x))
    assert np.allclose(x, y), "Links don't match"


@pytest.mark.parametrize("linkfun", POSITIVE_RESTRICTED_LINE_LINKS)
def test_link_positive_restricted_line(linkfun):
    """Test links that are defined on (eps, 1-eps)"""
    instance = linkfun()
    eps = 1e-10
    x = np.linspace(eps, 1 - eps, M)
    y = instance.inverse(instance.link(x))
    assert np.allclose(x, y), "Links don't match"


@pytest.mark.parametrize("linkfun", SHIFTED_LINKS)
@pytest.mark.parametrize("value", VALUES)
def test_link_positive_shifted_line(linkfun, value):
    """Test links that are shifted. This changes the domain of the links."""
    instance = linkfun(value)
    x = np.linspace(value, 100 + value, M)
    y = instance.inverse(instance.link(x))
    assert np.allclose(x, y), "Links don't match"
