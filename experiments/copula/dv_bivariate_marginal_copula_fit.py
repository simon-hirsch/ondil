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


##########################################
# Marginal Copula 
##########################################

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

X_df = X_df.drop(["set_flag"], axis=1)


N_TRAIN = split_idx
N_TEST = X_numpy.shape[0] - N_TRAIN
N = X_numpy.shape[0]
H = y_numpy.shape[1]

# Example: Select only the first two covariates for each equation
equation = {
    0: {0: "all", 1: "all"},
    1: {0: "all", 1: "all"},
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
    method= "ols",
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




# RUN THE SIMULATION STUDY!!
N_MODELS = 1

timings = np.zeros([N_MODELS, N_TEST])
predictions_cor = np.zeros((N_TEST, N_MODELS, 1))
predictions_loc = np.zeros((N_TEST, N_MODELS, H))
predictions_cov = np.zeros((N_TEST, N_MODELS, H, H))
predictions_dof = np.zeros((N_TEST, N_MODELS, 1))

import time


for m in range(N_MODELS):
    try:
        print("###############################################################")
        print(f"Fitting Model {m}", N_TRAIN, N)
        print("###############################################################")
        for k, i in tqdm(enumerate(range(N_TRAIN, N))):
            if k == 0:
                start = time.time()
                estimator.fit(X=X_numpy[:i, :], y=y_numpy[:i, :])
                stop = time.time()
                timings[m, k] = stop - start

                # Silence estimators after first initial fit
                estimator.verbose = 0
            else:
                start = time.time()
                estimator.update(X=X_numpy[[i - 1], :], y=y_numpy[[i - 1], :])
                stop = time.time()
                timings[m, k] = stop - start
            

            pred = estimator.predict(X=X_numpy[[i], :])
            pred = estimator.distribution.theta_to_scipy(pred)

            predictions_cor[k, m, ...] = pred['cor'][0].squeeze()
       
            np.savez_compressed(
                file="results/pred_copula.npz",
                timings=timings,
                predictions_cor=predictions_cor,
  
            )
            
    except Exception as e:
        print("###############################################################")
        print(f"Model {m}, step {k, i}, failed with exception", e)
        print("###############################################################")




predictions_cov[k, m, ...] = pred["marginal_2"].squeeze()
predictions_dof[k, m, ...] = pred["dependence"].squeeze()


from statsmodels.distributions.copula.api import GaussianCopula
import numpy as np

N_SIMS = 1000
RANDOM_STATE = 123


####
# Joint
####

# Create the simulation
RANDOM_STATE = 123
simulations = np.empty((N_TEST, N_MODELS, N_SIMS, 2))

for m in range(N_MODELS):
    for t in range(N_TEST):
        try:
            simulations[t, m, ...] = CopulaDistribution(GaussianCopula(corr=[[1, predictions_cor[t, m]], [predictions_cor[t, m], 1]]), 
                                [norm(predictions_marginal_1[t,m]), norm(predictions_marginal_2[t,m])]).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)
        except Exception as _:
            pass

samples = joint_dist.rvs(1000)

# Evaluate the joint PDF at a point
pdf_value = joint_dist.pdf([0, 0])
print("PDF at [0,0]:", pdf_value)





# # Save everything to the
np.savez_compressed(
    file="results/pred_multivariate.npz",
    timings=timings,
    predictions_loc=predictions_loc,
    predictions_cov=predictions_cov,
    predictions_dof=predictions_dof,
)