import numba as nb
import numpy as np
import pandas as pd
import scipy.stats as st
import scoringrules as sr
from tqdm import tqdm



def dawid_sebastiani_score(obs: np.ndarray, fct: np.ndarray):
    M = fct.shape[0]
    bias = obs - (np.sum(fct, axis=0) / M)
    cov = np.cov(fct, rowvar=False).astype(bias.dtype)
    prec = np.linalg.inv(cov).astype(bias.dtype)
    log_det = np.log(np.linalg.det(cov))
    bias_precision = bias.T @ prec @ bias
    return log_det + bias_precision


def energy_score_parallel(
    obs: np.ndarray,
    fct: np.ndarray,
):
    """Compute the Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        e_2_inner = np.zeros(M)
        for j in nb.prange(M):
            e_2_inner[j] = float(np.linalg.norm(fct[i] - fct[j]))
        e_2 += np.sum(e_2_inner)

    return e_1 / M - 0.5 / (M**2) * e_2



def energy_score_fast(
    obs: np.ndarray,
    fct: np.ndarray,
):
    """Compute the Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        for j in range(i, M):
            e_2 += float(np.linalg.norm(fct[i] - fct[j]))

    return e_1 / M - 1 / (M**2) * e_2


# Copula likelihood
def gaussian_copula_log_likelihood(
    uniform: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    # Naive estimation as in the paper by, see pages 106/107
    # @article{arbenz2013bayesian,
    #     title={Bayesian copulae distributions, with application to operational risk management—some comments},
    #     author={Arbenz, Philipp},
    #     journal={Methodology and computing in applied probability},
    #     volume={15},
    #     pages={105--108},
    #     year={2013},
    #     publisher={Springer}
    # }

    H = uniform.shape[1]
    std = cov[:, range(H), range(H)] ** 0.5
    corr = cov / (std[..., None] @ std[:, None, :])
    prec = np.linalg.inv(corr) - np.diag(np.ones(H))[None, ...]
    transf_unif = st.norm().ppf(np.clip(uniform, 1e-10, 1 - 1e-10))
    term_1 = (
        -1 / 2 * (transf_unif[:, None, :] @ prec @ transf_unif[:, :, None]).squeeze()
    )
    term_2 = 1 / np.sqrt(np.linalg.det(corr))
    return np.log(term_2) + term_1




results_df = pd.read_csv("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/results_df.csv")

y_numpy = results_df[["y1", "y2"]].to_numpy()
X_numpy = results_df.drop(columns=["y1", "y2"]).to_numpy()

# Flag the last 1/5 as test and the first 4/5 as train in an extra column
split_idx = int(X_numpy.shape[0] * 4/ 5)
flags = np.array(['train'] * split_idx + ['test'] * (X_numpy.shape[0] - split_idx))

# Optionally, add the flag as a column to DataFrames for easier inspection
X_df = pd.DataFrame(X_numpy, columns=[f"x_{i}" for i in range(X_numpy.shape[1])])
y_df = pd.DataFrame(y_numpy, columns=[f"y_{i}" for i in range(y_numpy.shape[1])])
X_df['set_flag'] = flags
y_df['set_flag'] = flags

IDX_TRAIN = X_df.set_flag.to_numpy() == "train"
IDX_TEST = X_df.set_flag.to_numpy() == "test"
N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)

X = X_df.drop(["set_flag"], axis=1)
y = y_df


N = X_numpy.shape[0]
H = y_numpy.shape[1]

# Takes roughly 60 mins to calculate the scores
#sims_univariate = np.load("results/sims_univariate_benchmark.npz")[KEY]
#sims_copula = np.load("results/sims_copula.npz")[KEY]
sims_multivariate = np.load("results/sims_multivariate.npz")[KEY]


sims = np.concatenate(
    (
        #sims_univariate,
        #sims_copula,
        sims_multivariate,
        
    ),
    axis=1,
)

N_MODELS = sims.shape[1]
N_SIMS = sims.shape[-2]

scores = np.zeros([N_TEST, N_MODELS, 9])
error_mean = np.expand_dims(y_numpy[N_TRAIN:, :], 1) - sims.mean(-2)
error_med = np.expand_dims(y_numpy[N_TRAIN:, :], 1) - np.median(sims, axis=-2)

# RMSE
# MAE
# This is a bit of a curious definition, might change it later
# We should make a distinction between the average RMSE per hour (averaged over time)
# and the actual RMSE across all hours, all time squared afterwards
scores[:, :, 0] = np.mean(error_mean**2, axis=2)  # Need to take the sqrt at the end
scores[:, :, 1] = np.mean(np.abs(error_med), axis=2)
# L2/Frobenius Norm on the Error Vector
scores[:, :, 2] = np.linalg.norm(error_mean, ord=2, axis=2)


# Variogram Score and Energy Score
for m in tqdm(range(scores.shape[1])):
    try:
        print("CRPS")
        # Ensemble CRPS
        scores[:, m, 3] = sr.crps_ensemble(
            y_numpy[N_TRAIN:, :],
            sims[:, m, ...],
            axis=-2,  # Ensemble dimension
        ).mean(1)
        print("VS")
        scores[:, m, 4] = sr.variogram_score(
            # y_numpy[IDX_TEST], sims[:, m, np.arange(N_SIMS // 2), :], p=1
            y_numpy[IDX_TEST],
            sims[:, m, :, :],
            p=1,
        )
        scores[:, m, 5] = sr.variogram_score(
            # y_numpy[IDX_TEST], sims[:, m, np.arange(N_SIMS // 2), :], p=0.5
            y_numpy[IDX_TEST],
            sims[:, m, :, :],
            p=0.5,
        )
        print("ES")
        scores[:, m, 6] = energy_score_fast(y_numpy[N_TRAIN:, :], sims[:, m, ...])
        print("DSS")
        scores[:, m, 7] = dawid_sebastiani_score(y_numpy[IDX_TEST, :], sims[:, m, ...])
    except Exception as _:
        print(f"Could not calculate score for model {m}.")

print(np.round(scores.mean(0), 2))

############### Log-Likelihood scores #####################
file_univariate = np.load(file="results/pred_univariate_benchmark.npz")
file_univariate_distreg = np.load("results/univariate_predictions_distreg.npz")
file_copula = np.load(file="results/pred_copula.npz")
file_multivariate = np.load(file="results/pred_multivariate.npz")

## Predictions univariate
predictions_loc_ar = file_univariate["predictions_loc"]
predictions_cov_ar = file_univariate["predictions_cov"]
for m in range(2):
    # Gaussian ARX Models here
    for t in range(N_TEST):
        scores[t, m, 8] = -st.multivariate_normal(
            mean=predictions_loc_ar[t],
            cov=predictions_cov_ar[t] if m == 1 else np.diag(predictions_cov_ar[t]),
        ).logpdf(y_numpy[N_TRAIN + t, :])

## Predictions Copula
pred_uni_copula = file_copula["predictions_uni"]
pred_cov_copula = file_copula["predictions_cov"]

marginal_loglikelihood = np.zeros((N_TEST, H))
marginal_predictions = file_univariate_distreg["predictions_outofsample"]
for h in range(H):
    marginal_loglikelihood[:, h] = rolch.DistributionT().pdf(
        y_numpy[N_TRAIN:, h], marginal_predictions[:, h, :]
    )

scores[:, 2, 8] = -np.log(marginal_loglikelihood).sum(1)

for c, m in enumerate(range(3, 5)):
    # Copula log score here!!
    scores[:, m, 8] = -(
        gaussian_copula_log_likelihood(pred_uni_copula, pred_cov_copula[:, c])
        + np.log(marginal_loglikelihood).sum(1)
    )

## Predictions Mutlviariate model
predictions_loc_mv = file_multivariate["predictions_loc"]
predictions_cov_mv = file_multivariate["predictions_cov"]
predictions_dof_mv = file_multivariate["predictions_dof"]

for m, idx in enumerate(range(5, 11)):
    # MV GAMLSS Models here
    for t in range(N_TEST):
        scores[t, idx, 8] = -st.multivariate_t(
            loc=predictions_loc_mv[t, m],
            shape=predictions_cov_mv[t, m],
            df=predictions_dof_mv[t, m],
        ).logpdf(y_numpy[N_TRAIN + t, :])

np.save(file="results/scores", arr=scores)

print(np.round(scores.mean(0), 2))








