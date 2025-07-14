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
# Copula 
##########################################


# Read the merged data from the CSV file
merged_data = pd.read_csv("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/ondil/experiments/copula/merged_data_1.csv")

y_numpy = merged_data[["u1", "u2"]].to_numpy()
X_numpy = merged_data.drop(columns=["u1", "u2"]).to_numpy()


H = 2
equation = {
    0: {
        h: np.arange(X_numpy.shape[1])
        for h in range(H)
    }
}

# Flag the last 1/5 as test and the first 4/5 as train in an extra column
split_idx = int(X_numpy.shape[0] * 4/ 5)
flags = np.array(['train'] * split_idx + ['test'] * (X_numpy.shape[0] - split_idx))

# Optionally, add the flag as a column to DataFrames for easier inspection
X_df = pd.DataFrame(X_numpy, columns=[f"x_{i}" for i in range(X_numpy.shape[1])])
y_df = pd.DataFrame(y_numpy, columns=[f"y_{i}" for i in range(y_numpy.shape[1])])
X_df['set_flag'] = flags
y_df['set_flag'] = flags


N_TRAIN = split_idx
N_TEST = X_numpy.shape[0] - N_TRAIN
N = X_numpy.shape[0]


distribution = BivariateCopulaNormal(
    link=FisherZLink(),
    param_link=KendallsTauToParameter()
)

estimator = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution,
    equation=equation,
    method= "ols",
    early_stopping=False,
    early_stopping_criteria="bic",
    iteration_along_diagonal=False,
    verbose=3,
    max_iterations_inner =  5,
    max_iterations_outer =  100,
    scale_inputs =False,
)

estimator.fit(X_numpy, y_numpy)

estimator.coef_ 

estimator.fit(X=X_numpy[:N_TRAIN, :], y=y_numpy[:N_TRAIN, :])

estimator.update(X=X_numpy[(N_TRAIN - 1):, :], y=y_numpy[(N_TRAIN - 1):, :])



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





##########################################
# Univariate
##########################################


from src.ondil.estimators import OnlineGamlss

# Model coefficients 
equation = {
    0 : "all", # Can also use "intercept" or np.ndarray of integers / booleans
    1 : "intercept",
}

# Create the estimator
estimator  = OnlineGamlss(
    distribution=DistributionNormal(),
    method="ols",
    equation=equation,
    fit_intercept=True,
    ic="bic",
)


estimator.fit(
    X=X_numpy, 
    y=y_numpy[:,0]
)

estimator.beta




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


ecdf outside of the function 
and then when using the predictions transfrom back 


predictions_cov[k, m, ...] = pred["marginal_2"].squeeze()
predictions_dof[k, m, ...] = pred["dependence"].squeeze()


from statsmodels.distributions.copula.api import GaussianCopula
import numpy as np

N_SIMS = 1000
RANDOM_STATE = 123

# Initialize simulation array
simulations = np.empty((N_TEST, N_MODELS, N_SIMS, 2))
for m in range(N_MODELS):
    for t in range(N_TEST):
        try:
            # Clamp correlation to avoid singular matrix
            rho = float(predictions_cor[t, m])
            corr_matrix = [[1, rho], [rho, 1]]

            # Create copula and draw samples
            copula = GaussianCopula(corr=corr_matrix)
            samples = np.asarray(copula.rvs(N_SIMS, random_state=RANDOM_STATE + t))

            # Assign samples
            simulations[t, m, :, :] = samples

        except Exception as e:
            print(f"Skipped t={t}, m={m} due to error: {e}")
            continue



np.savez_compressed(
    file="results/sims_multivariate.npz",
    simulations=simulations,
)






####
# Joint
####

N_SIMS = 1000
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











































merged_data = pd.read_csv("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/sim_data_26052025.csv")

L = 100
core_max = 10
n_jobs = min(os.cpu_count() - 1, L, core_max)

start_time = time.time()

def run_simulation(ll, merged_data):
    # You can add your simulation logic here
    # For now, just create a DataFrame with ll and merged_data
    # If merged_data is a DataFrame, you may want to add a column 'll'
    df = merged_data.copy()
    df_sub = df[df['ll'] == ll]
    y_numpy = df_sub[["merged_data.u1", "merged_data.u2"]].to_numpy()
    X_numpy = df_sub.drop(columns=["ll","merged_data.u1", "merged_data.u2"]).to_numpy()
    estimator[0].fit(X=X_numpy, y=y_numpy)
    beta_hat = estimator[0].beta[0][0][0]
    # Store results in a DataFrame (beta_hat is a 1x4 array, flatten for columns)
    results = pd.DataFrame({
        'll': [ll],
        'beta_hat_0': [beta_hat[0]],
        'beta_hat_1': [beta_hat[1]],
        'beta_hat_2': [beta_hat[2]],
        'beta_hat_3': [beta_hat[3]],
    })
    return results

results = Parallel(n_jobs=n_jobs)(
    delayed(run_simulation)(ll, merged_data) for ll in range(1, L + 1)
)

# Combine all results into a single DataFrame
res_df_MC = pd.concat(results, ignore_index=True)

end_time = time.time()
run_time = end_time - start_time
print(f"Run time: {run_time:.2f} seconds")


# Calculate and print the mean of each beta_hat variable in the results DataFrame
beta_cols = [col for col in res_df_MC.columns if col.startswith('beta_hat_')]
beta_means = res_df_MC[beta_cols].mean()
print("Mean of beta_hat variables:")
print(beta_means)



# Save as pickle file
res_df_MC.to_pickle("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/sim_data_26052025.pkl")




                      











# Create the simulation
RANDOM_STATE = 123
simulations = np.empty((N_TEST, N_MODELS, N_SIMS, H))

for m in range(N_MODELS):
    for t in range(N_TEST):
        try:
            simulations[t, m, ...] = st.multivariate_t(
                loc=predictions_loc[t, m],
                shape=predictions_cov[t, m],
                df=predictions_dof[t, m],
            ).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)
        except Exception as _:
            pass



# # Save everything to the
np.savez_compressed(
    file="results/pred_multivariate.npz",
    timings=timings,
    predictions_loc=predictions_loc,
    predictions_cov=predictions_cov,
    predictions_dof=predictions_dof,
)

np.savez_compressed(
    file="results/sims_multivariate.npz",
    simulations=simulations,
)


