import numpy as np
import pytest

from rolch.gram import init_forget_vector
from rolch.stats import IncrementalWeightedMean

M = [100, 1000, 10000]
D = [1, 10, 100]
forget = [0, 0.1, 0.001, 0.0001]
loc = [0, 10, -10]
scale = [1, 10]


@pytest.mark.parametrize("M", M)
@pytest.mark.parametrize("D", D)
@pytest.mark.parametrize("forget", forget)
@pytest.mark.parametrize("loc", loc)
@pytest.mark.parametrize("scale", scale)
def test_incremental_weighted_mean(M, D, forget, loc, scale):

    rvs = np.random.normal(loc=loc, scale=scale, size=(M, D))
    w = np.random.uniform(size=(M, D))

    out_online = np.empty_like(rvs)
    out_numpy = np.empty_like(rvs)

    om = IncrementalWeightedMean(forget=forget, axis=0)

    for i in range(M):
        om.update_save(rvs[[i]], w[[i]])
        forget_vector = init_forget_vector(forget, i + 1)
        out_numpy[i] = np.average(
            rvs[: (i + 1)], weights=w[: (i + 1)] * forget_vector[:, None], axis=0
        )
        out_online[i] = om.avg

    assert np.allclose(
        out_online, out_numpy
    ), f"Error happened for forget={forget}, D={D}, M={M}, loc={loc}, scale={scale}"
