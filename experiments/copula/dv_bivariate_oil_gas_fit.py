import sys
import os
import importlib

#  Add your local package to the path
ondil_path = r"C:\Users\OEK-admin\OneDrive\Arbeit_Uni\Uni_Due\ProjectII\ondil"
if ondil_path not in sys.path:
    sys.path.insert(0, ondil_path)

print("Python Path:", sys.path)
print("Current Working Directory:", os.getcwd())

#  HOT RELOAD: Clear old ondil modules from cache
for name in list(sys.modules):
    if "src.ondil" in name:  # adjust if your package is imported as ondil.* instead
        del sys.modules[name]
importlib.invalidate_caches()

#  Import ondil classes
from src.ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from src.ondil.links import FisherZLink, KendallsTauToParameter
from src.ondil.distributions import BivariateCopulaNormal, MarginalCopula, Normal

#  Other imports
import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
from joblib import Parallel, delayed

np.set_printoptions(precision=3, suppress=True)



##############################################
# Oil and Gas
##############################################

fuels = pd.read_csv("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/fuels.csv")
fuels = fuels.dropna()
fuels = fuels[["Date", "oil_fM_01", "gas_fM_01"]]

# Compute log returns for oil and gas, multiplied by 100
fuels["oil_fM_01"] = 100 * (np.log(fuels["oil_fM_01"]) - np.log(fuels["oil_fM_01"].shift(1)))
fuels["gas_fM_01"] = 100 * (np.log(fuels["gas_fM_01"]) - np.log(fuels["gas_fM_01"].shift(1)))
fuels = fuels.dropna()

y_numpy = fuels[["oil_fM_01", "gas_fM_01"]].to_numpy()

# Create lagged features for both oil and gas series
fuels_lagged = fuels[["oil_fM_01", "gas_fM_01"]].shift(1)
fuels_lagged.columns = ["oil_fM_01_lag1", "gas_fM_01_lag1"]
fuels_with_lags = pd.concat([fuels, fuels_lagged], axis=1).dropna()

y_numpy = fuels_with_lags[["oil_fM_01", "gas_fM_01"]].to_numpy()
X_numpy = fuels_with_lags[["oil_fM_01_lag1", "gas_fM_01_lag1"]].to_numpy()

# Flag the last 1/5 as test and the first 4/5 as train in an extra column
split_idx = int(X_numpy.shape[0] * 4/ 5)
flags = np.array(['train'] * split_idx + ['test'] * (X_numpy.shape[0] - split_idx))

# Optionally, add the flag as a column to DataFrames for easier inspection
X_df = pd.DataFrame(X_numpy, columns=[f"x_{i}" for i in range(X_numpy.shape[1])])
y_df = pd.DataFrame(y_numpy, columns=[f"y_{i}" for i in range(y_numpy.shape[1])])
X_df['set_flag'] = flags
y_df['set_flag'] = flags

X_df = X_df.drop(["set_flag"], axis=1)

N_TRAIN = split_idx
N_TEST = X_numpy.shape[0] - N_TRAIN
N = X_numpy.shape[0]
H = y_numpy.shape[1]

# Example: Select only the first two covariates for each equation
equation = {
    0: {0: "all", 1: "intercept"},
    1: {0: "all", 1: "intercept"},
    2: {0: "all"}
}

distribution_marg_cop = MarginalCopula(
    marginal_0 = Normal(),
    marginal_1 = Normal(),
    dependence = BivariateCopulaNormal(
        link=FisherZLink(),
        param_link=KendallsTauToParameter(),
    ),
)

estimator = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_marg_cop,
    equation=equation,
    method= "lasso",
    early_stopping=False,
    early_stopping_criteria="bic",
    iteration_along_diagonal=False,
    verbose=3,
    max_iterations_inner =  5,
    max_iterations_outer =  50,
)

estimator.fit(X=X_numpy, y=y_numpy)

estimator.fit(X=X_numpy[:N_TRAIN, :], y=y_numpy[:N_TRAIN, :])

estimator.update(X=X_numpy[(N_TRAIN - 1):, :], y=y_numpy[(N_TRAIN - 1):, :])

estimator.coef_





