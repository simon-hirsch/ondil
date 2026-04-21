import numpy as np
import pytest

from ondil.gram import init_forget_vector
from ondil.scaler import OnlineMeanAbsoluteDeviationScaler, OnlineScaler


@pytest.mark.parametrize("N_init", [5, 10, 100], ids=["init_1", "init_10", "init_100"])
@pytest.mark.parametrize("D", [1, 10], ids=["vars_1", "vars_10"])
@pytest.mark.parametrize(
    "selection_dtype", [bool, int], ids=["sel_dtype_bool", "sel_dtype_int"]
)
@pytest.mark.parametrize(
    "forget",
    [0, 0.01, 0.1],
    ids=["no_forgetting", "forget_0.01", "forget_0.1"],
)
@pytest.mark.parametrize(
    "sample_weight",
    [True, False],
    ids=["sample_weight", "no_sample_weight"],
)
def test_online_scaler(N_init, D, forget, selection_dtype, sample_weight):
    """Test OnlineScaler with various combinations of parameters."""
    # Skip invalid combinations
    N = 100

    X = np.random.uniform(0, 10, N * D).reshape(-1, D)
    X_init = X[0:N_init,]

    # Calculate effective weights for comparison
    W_forget = init_forget_vector(forget, N).reshape(-1, 1)

    if sample_weight:
        W_sample = np.random.uniform(0.1, 1.0, (N, 1)).reshape(-1, 1)
        W_effective = W_forget * W_sample
    else:
        W_sample = None
        W_effective = W_forget

    if selection_dtype is bool:
        to_scale = np.random.choice([True, False], D)
    if selection_dtype is int:
        to_scale = np.random.choice(
            np.arange(D), np.random.randint(0, D + 1), replace=False
        )

    # Calculate true mean and variance for all columns after initial fit
    true_mean_init = np.array([
        np.average(X_init[:, d], weights=W_effective[0:N_init].flatten())
        for d in range(D)
    ])
    true_var_init = np.array([
        np.average(
            (X_init[:, d] - true_mean_init[d]) ** 2,
            weights=W_effective[0:N_init].flatten(),
        )
        for d in range(D)
    ])

    # Setup and fit the OnlineScaler
    os = OnlineScaler(forget=forget, to_scale=to_scale)
    if sample_weight:
        os.fit(X_init, sample_weight=W_sample[0:N_init,].flatten())
    else:
        os.fit(X_init)

    # # Assert initial fit is correct
    assert np.allclose(os.mean_, true_mean_init[to_scale]), (
        f"Initial mean mismatch: {os.mean_} vs {true_mean_init[to_scale]}"
    )
    assert np.allclose(os.var_, true_var_init[to_scale]), (
        f"Initial variance mismatch: {os.var_} vs {true_var_init[to_scale]}"
    )

    # For N_init = 1, var = 0, so scaling is not defined
    if N_init > 1:
        expected_out = np.copy(X_init)
        scaled_out = (X_init - true_mean_init) / np.sqrt(true_var_init)
        expected_out[:, to_scale] = scaled_out[:, to_scale]
        out = os.transform(X=X_init)
        assert np.allclose(out, expected_out), (
            f"Initial scaled X mismatch: {out} vs {expected_out}"
        )

    # Test partial fit updates
    for i in range(N_init, N):
        if sample_weight:
            os.update(X[i : i + 1,], sample_weight=W_sample[i,])
        else:
            os.update(X[i : i + 1,])

        # Calculate true mean and variance for all columns
        true_mean = np.array([
            np.average(X[0 : (i + 1), d], weights=W_effective[0 : (i + 1)].flatten())
            for d in range(D)
        ])
        true_var = np.array([
            np.average(
                (X[0 : (i + 1), d] - true_mean[d]) ** 2,
                weights=W_effective[0 : (i + 1)].flatten(),
            )
            for d in range(D)
        ])
        expected_out = np.copy(X[0 : (i + 1),])
        scaled_out = (X[0 : (i + 1), :] - true_mean) / np.sqrt(true_var)
        expected_out[:, to_scale] = scaled_out[:, to_scale]
        out = os.transform(X=X[0 : (i + 1), :])

        assert np.allclose(os.mean_, true_mean[to_scale]), (
            f"Step {i - N_init + 1}: Mean mismatch: {os.mean_} vs {true_mean[to_scale]}"
        )
        assert np.allclose(os.var_, true_var[to_scale]), (
            f"Step {i - N_init + 1}: Variance mismatch: {os.var_} vs {true_var[to_scale]}"
        )
        assert np.allclose(out, expected_out), (
            f"Scaled X mismatch: {out} vs {expected_out}"
        )


@pytest.mark.parametrize("D", [1, 10], ids=["features_1", "features_10"])
def test_standard_scaling_dont_scale(D):
    N = 100
    X = np.random.uniform(0, 10, N * D).reshape(-1, D)
    expected_out = X

    scaler = OnlineScaler(forget=0, to_scale=False)
    scaler.fit(X=X)
    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)


@pytest.mark.parametrize("N_init", [5, 10], ids=["init_5", "init_10"])
@pytest.mark.parametrize("D", [1, 5], ids=["vars_1", "vars_5"])
@pytest.mark.parametrize(
    "selection_dtype", [bool, int], ids=["sel_dtype_bool", "sel_dtype_int"]
)
@pytest.mark.parametrize(
    "forget",
    [0, 0.1],
    ids=["no_forgetting", "forget_0.1"],
)
@pytest.mark.parametrize(
    "sample_weight",
    [True, False],
    ids=["sample_weight", "no_sample_weight"],
)
def test_online_mad_scaler(N_init, D, forget, selection_dtype, sample_weight):
    """Test OnlineMeanAbsoluteDeviationScaler with various parameter combinations."""
    N = 80

    X = np.random.uniform(0, 10, N * D).reshape(-1, D)
    X_init = X[0:N_init,]

    W_forget = init_forget_vector(forget, N).reshape(-1, 1)

    if sample_weight:
        W_sample = np.random.uniform(0.1, 1.0, (N, 1)).reshape(-1, 1)
        W_effective = W_forget * W_sample
    else:
        W_sample = None
        W_effective = W_forget

    if selection_dtype is bool:
        to_scale = np.random.choice([True, False], D)
    if selection_dtype is int:
        to_scale = np.random.choice(
            np.arange(D), np.random.randint(0, D + 1), replace=False
        )

    true_mean_init = np.array([
        np.average(X_init[:, d], weights=W_effective[0:N_init].flatten())
        for d in range(D)
    ])
    true_mad_init = np.array([
        np.average(
            np.abs(X_init[:, d] - true_mean_init[d]),
            weights=W_effective[0:N_init].flatten(),
        )
        for d in range(D)
    ])

    os = OnlineMeanAbsoluteDeviationScaler(forget=forget, to_scale=to_scale)
    if sample_weight:
        os.fit(X_init, sample_weight=W_sample[0:N_init,].flatten())
    else:
        os.fit(X_init)

    assert np.allclose(os.mean_, true_mean_init[to_scale]), (
        f"Initial mean mismatch: {os.mean_} vs {true_mean_init[to_scale]}"
    )
    assert np.allclose(os.scale_, true_mad_init[to_scale]), (
        f"Initial MAD mismatch: {os.scale_} vs {true_mad_init[to_scale]}"
    )

    expected_out = np.copy(X_init)
    expected_out[:, to_scale] = (X_init[:, to_scale] - true_mean_init[to_scale]) / (
        true_mad_init[to_scale]
    )
    out = os.transform(X=X_init)
    assert np.allclose(out, expected_out), (
        f"Initial scaled X mismatch: {out} vs {expected_out}"
    )

    expected_mean = np.copy(true_mean_init[to_scale])
    expected_mad = np.copy(true_mad_init[to_scale])
    W_forget_init = init_forget_vector(forget, N_init)
    if sample_weight:
        expected_cumulative_w = np.sum(W_sample[0:N_init,].flatten() * W_forget_init)
    else:
        expected_cumulative_w = np.sum(W_forget_init)
    expected_A = expected_mad * expected_cumulative_w

    for i in range(N_init, N):
        if sample_weight:
            os.update(X[i : i + 1,], sample_weight=W_sample[i,])
            w_i = W_sample[i,].item()
        else:
            os.update(X[i : i + 1,])
            w_i = 1.0

        eff_old_w = expected_cumulative_w * (1 - forget)
        expected_cumulative_w = eff_old_w + w_i
        x_new = X[i, to_scale]
        expected_mean = (
            expected_mean * eff_old_w + x_new * w_i
        ) / expected_cumulative_w
        diff_new = x_new - expected_mean
        expected_A = expected_A * (1 - forget) + w_i * np.abs(diff_new)
        expected_mad = expected_A / expected_cumulative_w

        expected_out = np.copy(X[0 : (i + 1),])
        expected_out[:, to_scale] = (
            X[0 : (i + 1), :][:, to_scale] - expected_mean
        ) / expected_mad
        out = os.transform(X=X[0 : (i + 1), :])

        assert np.allclose(os.mean_, expected_mean), (
            f"Step {i - N_init + 1}: Mean mismatch: {os.mean_} vs {expected_mean}"
        )
        assert np.allclose(os.scale_, expected_mad), (
            f"Step {i - N_init + 1}: MAD mismatch: {os.scale_} vs {expected_mad}"
        )
        assert np.allclose(out, expected_out), (
            f"Scaled X mismatch: {out} vs {expected_out}"
        )


@pytest.mark.parametrize("D", [1, 10], ids=["features_1", "features_10"])
def test_mad_scaling_dont_scale(D):
    N = 100
    X = np.random.uniform(0, 10, N * D).reshape(-1, D)
    expected_out = X

    scaler = OnlineMeanAbsoluteDeviationScaler(forget=0, to_scale=False)
    scaler.fit(X=X)
    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)
