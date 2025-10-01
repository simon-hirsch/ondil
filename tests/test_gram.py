import numpy as np
import pytest
import scipy.stats as st
from ondil.gram import (
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)


def make_x_y_w(N, D, random_weights=True):
    X = st.multivariate_normal().rvs((N, D))
    if D == 1:
        X = X.reshape(-1, 1)
    y = st.multivariate_normal().rvs((N, 1))
    if random_weights:
        w = st.uniform().rvs(N)
    else:
        w = np.ones(N)
    return X, y, w


N = [100, 1000]
D = [1, 2, 10]
RANDOM_WEIGHTS = [True, False]
FORGET = [0, 0.0001, 0.001, 0.01, 0.1]
BATCH_SIZE = [10, 25]


@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
def test_inverse_rank_deficit(N, D, random_weights, forget):
    X, _, w = make_x_y_w(N, D, random_weights=random_weights)
    for d in range(1, D + 1):
        choice = np.random.choice(np.arange(D), d)
        XX = np.hstack((X, X[:, choice]))
        with pytest.raises(ValueError):
            _ = init_inverted_gram(XX[:-1], w[:-1], forget)


@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
def test_single_update_x_gram(N, D, random_weights, forget):
    X, _, w = make_x_y_w(N, D, random_weights=random_weights)
    gram_start = init_gram(X[:-1], w[:-1], forget)
    gram_final = init_gram(X, w, forget)
    assert np.allclose(
        gram_final, update_gram(gram_start, X[[-1]], forget=forget, w=w[-1])
    )


@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
@pytest.mark.parametrize("batchsize", BATCH_SIZE, ids=lambda x: f"batchsize_{x}")
def test_batch_update_x_gram(N, D, random_weights, forget, batchsize):
    X, _, w = make_x_y_w(N, D, random_weights=random_weights)
    gram_start = init_gram(X[:-batchsize], w[:-batchsize], forget)
    gram_final = init_gram(X, w, forget)
    assert np.allclose(
        gram_final,
        update_gram(gram_start, X[-batchsize:, :], forget=forget, w=w[-batchsize:]),
    )


# INVERTED GRAM
@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
def test_single_update_inv_gram(N, D, random_weights, forget):
    X, _, w = make_x_y_w(N, D, random_weights=random_weights)
    gram_start = init_inverted_gram(X[:-1], w[:-1], forget)
    gram_final = init_inverted_gram(X, w, forget)
    assert np.allclose(
        gram_final, update_inverted_gram(gram_start, X[[-1]], forget=forget, w=w[-1])
    )


@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
@pytest.mark.parametrize("batchsize", BATCH_SIZE, ids=lambda x: f"batchsize_{x}")
def test_batch_update_inv_gram(N, D, random_weights, forget, batchsize):
    X, _, w = make_x_y_w(N, D, random_weights=random_weights)
    gram_start = init_inverted_gram(X[:-batchsize], w[:-batchsize], forget)
    gram_final = init_inverted_gram(X, w, forget)
    assert np.allclose(
        gram_final,
        update_inverted_gram(
            gram_start, X[-batchsize:, :], forget=forget, w=w[-batchsize:]
        ),
    )


# Y-GRAM
@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
def test_single_update_y_gram(N, D, random_weights, forget):
    X, y, w = make_x_y_w(N, D, random_weights=random_weights)
    gram_start = init_y_gram(X[:-1], y[:-1], w[:-1], forget)
    gram_final = init_y_gram(X, y, w, forget)
    assert np.allclose(
        gram_final, update_y_gram(gram_start, X[[-1]], y[[-1]], forget=forget, w=w[-1])
    )


@pytest.mark.parametrize("N", N, ids=lambda x: f"N_{x}")
@pytest.mark.parametrize("D", D, ids=lambda x: f"D_{x}")
@pytest.mark.parametrize(
    "random_weights", RANDOM_WEIGHTS, ids=lambda x: f"random_weights_{x}"
)
@pytest.mark.parametrize("forget", FORGET, ids=lambda x: f"forget_{x}")
@pytest.mark.parametrize("batchsize", BATCH_SIZE, ids=lambda x: f"batchsize_{x}")
def test_batch_update_y_gram(N, D, random_weights, forget, batchsize):
    X, y, w = make_x_y_w(N, D, random_weights=random_weights)
    gram_start = init_y_gram(X[:-batchsize], y[:-batchsize], w[:-batchsize], forget)
    gram_final = init_y_gram(X, y, w, forget)
    assert np.allclose(
        gram_final,
        update_y_gram(
            gram_start,
            X[-batchsize:, :],
            y[-batchsize:],
            forget=forget,
            w=w[-batchsize:],
        ),
    )
