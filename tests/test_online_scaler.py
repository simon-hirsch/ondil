import numpy as np
import pytest

from ondil.gram import init_forget_vector
from ondil.scaler import OnlineScaler


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
    true_mean_init = np.array(
        [
            np.average(X_init[:, d], weights=W_effective[0:N_init].flatten())
            for d in range(D)
        ]
    )
    true_var_init = np.array(
        [
            np.average(
                (X_init[:, d] - true_mean_init[d]) ** 2,
                weights=W_effective[0:N_init].flatten(),
            )
            for d in range(D)
        ]
    )

    # Setup and fit the OnlineScaler
    os = OnlineScaler(forget=forget, to_scale=to_scale)
    if sample_weight:
        os.fit(X_init, sample_weight=W_sample[0:N_init,].flatten())
    else:
        os.fit(X_init)

    # # Assert initial fit is correct
    assert np.allclose(
        os.mean_, true_mean_init[to_scale]
    ), f"Initial mean mismatch: {os.mean_} vs {true_mean_init[to_scale]}"
    assert np.allclose(
        os.var_, true_var_init[to_scale]
    ), f"Initial variance mismatch: {os.var_} vs {true_var_init[to_scale]}"

    # For N_init = 1, var = 0, so scaling is not defined
    if N_init > 1:
        expected_out = np.copy(X_init)
        scaled_out = (X_init - true_mean_init) / np.sqrt(true_var_init)
        expected_out[:, to_scale] = scaled_out[:, to_scale]
        out = os.transform(X=X_init)
        assert np.allclose(
            out, expected_out
        ), f"Initial scaled X mismatch: {out} vs {expected_out}"

    # Test partial fit updates
    for i in range(N_init, N):
        if sample_weight:
            os.update(X[i : i + 1,], sample_weight=W_sample[i,])
        else:
            os.update(X[i : i + 1,])

        # Calculate true mean and variance for all columns
        true_mean = np.array(
            [
                np.average(
                    X[0 : (i + 1), d], weights=W_effective[0 : (i + 1)].flatten()
                )
                for d in range(D)
            ]
        )
        true_var = np.array(
            [
                np.average(
                    (X[0 : (i + 1), d] - true_mean[d]) ** 2,
                    weights=W_effective[0 : (i + 1)].flatten(),
                )
                for d in range(D)
            ]
        )
        expected_out = np.copy(X[0 : (i + 1),])
        scaled_out = (X[0 : (i + 1), :] - true_mean) / np.sqrt(true_var)
        expected_out[:, to_scale] = scaled_out[:, to_scale]
        out = os.transform(X=X[0 : (i + 1), :])

        assert np.allclose(
            os.mean_, true_mean[to_scale]
        ), f"Step {i - N_init + 1}: Mean mismatch: {os.mean_} vs {true_mean[to_scale]}"
        assert np.allclose(
            os.var_, true_var[to_scale]
        ), f"Step {i - N_init + 1}: Variance mismatch: {os.var_} vs {true_var[to_scale]}"
        assert np.allclose(
            out, expected_out
        ), f"Scaled X mismatch: {out} vs {expected_out}"


@pytest.mark.parametrize("D", [1, 10], ids=["features_1", "features_10"])
def test_standard_scaling_dont_scale(D):
    N = 100
    X = np.random.uniform(0, 10, N * D).reshape(-1, D)
    expected_out = X

    scaler = OnlineScaler(forget=0, to_scale=False)
    scaler.fit(X=X)
    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)
