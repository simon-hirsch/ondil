# %%

import sys
import os
import importlib

from ondil.src.ondil.links.loglinks import LogShiftTwo, LogShiftValue

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
from ondil.src.ondil.distributions.bicop_studentt import BivariateCopulaStudentT
from ondil.src.ondil.links.copulalinks import KendallsTauToParameterClayton
from src.ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from src.ondil.links import Log, GumbelLink, KendallsTauToParameter, FisherZLink, KendallsTauToParameterGumbel
from src.ondil.distributions import BivariateCopulaNormal, BivariateCopulaGumbel, BivariateCopulaClayton, BivariateCopulaStudentT, MarginalCopula, Normal

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
merged_data = pd.read_csv("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/merged_data_1.csv")

y_numpy = merged_data[["u1", "u2"]].to_numpy()
X_numpy = merged_data.drop(columns=["u1", "u2"]).to_numpy()

H = 2
equation = {
    0: {
        h: np.arange(X_numpy.shape[1])
        for h in range(H)
    },  

   

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


#distribution = BivariateCopulaGumbel(
#    link= GumbelLink(),
#    param_link=KendallsTauToParameterGumbel()
#)

#distribution = BivariateCopulaClayton(
#    link= Log(),
#    param_link=KendallsTauToParameterClayton(),
#    rotation = 0,
#)

#distribution = BivariateCopulaNormal(
#    link= FisherZLink(),
#   param_link=KendallsTauToParameter()
#)

distribution = BivariateCopulaStudentT(
    link_1= FisherZLink(),
    param_link_1=KendallsTauToParameter(),
    link_2= LogShiftValue(value=2),
    param_link_2=KendallsTauToParameter(),
    rotation = 0,
)

estimator = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution,
    equation=equation,
    method= "ols",
    early_stopping=False,
    early_stopping_criteria="bic",
    iteration_along_diagonal=False,
    verbose=3,
    max_iterations_inner =  1,
    max_iterations_outer =  2,
    scale_inputs =False,
)


estimator.fit(X_numpy, y_numpy)

estimator.coef_ 

estimator.fit(X=X_numpy[:N_TRAIN, :], y=y_numpy[:N_TRAIN, :])

estimator.update(X=X_numpy[(N_TRAIN - 1):, :], y=y_numpy[(N_TRAIN - 1):, :])







##########################################
# Simulation Study
##########################################

import glob
import pickle
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
# Path to simulation data files
sim_data_folder = "C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/simulations/"
sim_files = sorted(glob.glob(sim_data_folder + "sim_data_*.csv"))

coef_results = []

# Use the first 100 datasets (or fewer if less available)
for sim_file in sim_files[:250]:
    print(f"Processing {sim_file}")
    merged_data = pd.read_csv(sim_file)

    y_numpy = merged_data[["u1", "u2"]].to_numpy()
    X_numpy = merged_data.drop(columns=["u1", "u2"]).to_numpy()

    H = 2
    equation = {
        0: {
            h: np.arange(X_numpy.shape[1])
            for h in range(H)
        }
    }

    distribution = BivariateCopulaNormal(
        link=FisherZLink(),
        param_link=KendallsTauToParameter()
    )

    estimator = MultivariateOnlineDistributionalRegressionPath(
        distribution=distribution,
        equation=equation,
        method="ols",
        early_stopping=False,
        early_stopping_criteria="bic",
        iteration_along_diagonal=False,
        verbose=0,
        max_iterations_inner=5,
        max_iterations_outer=50,
        scale_inputs=False,
    )

    estimator.fit(X_numpy, y_numpy)
    coef = estimator.coef_[0][0][0][0]

    coef_results.append(coef)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(coef_results, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Coefficients (coef_results)")
plt.xlabel("Coefficient Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()





##########################################
# Test on one dataset
##########################################


import glob
import pickle
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
# Path to simulation data files
sim_data_folder = "C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/simulations/"
sim_files = sorted(glob.glob(sim_data_folder + "sim_data_*.csv"))


print(f"Processing {sim_files[0]}")
merged_data = pd.read_csv(sim_files[0])

y_numpy = merged_data[["u1", "u2"]].to_numpy()
X_numpy = merged_data.drop(columns=["u1", "u2"]).to_numpy()

y_numpy = Y
X_numpy = X
combined_numpy = np.hstack([y_numpy['Copula'], X_numpy['Copula']])
combined_df = pd.DataFrame(combined_numpy)
combined_df.to_csv("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/combined_data.csv", index=False)

H = 2
equation = {
    0: {
        h: np.arange(X_numpy['Copula'].shape[1])
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
    method="ols",
    early_stopping=False,
    early_stopping_criteria="bic",
    iteration_along_diagonal=False,
    verbose=3,
    max_iterations_inner=5,
    max_iterations_outer=25,
    scale_inputs=False,
    fit_intercept=False,
)

estimator.fit(X_numpy['Copula'], y_numpy['Copula'])

estimator.fit(X=X_numpy[:N_TRAIN, :], y=y_numpy[:N_TRAIN, :])

estimator.update(X=X_numpy[(N_TRAIN - 1):, :], y=y_numpy[(N_TRAIN - 1):, :])

estimator.coef_






print("Coefficient from first dataset:", coef)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist([coef], bins=1, color='skyblue', edgecolor='black')
plt.title("Coefficient from First Dataset")
plt.xlabel("Coefficient Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()





















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



from statsmodels.distributions.copula.api import GaussianCopula
import numpy as np
import glob
import pickle

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




                      
