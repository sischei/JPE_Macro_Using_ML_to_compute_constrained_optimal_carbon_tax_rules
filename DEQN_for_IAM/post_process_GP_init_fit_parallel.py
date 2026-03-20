import importlib
import pandas as pd
import numpy as np
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions

import sys, shutil, os
import time
import sampling_module as sm
import tensorflow as tf

from itertools import product
import pickle

import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch.linalg import vector_norm, norm

from scipy.stats.qmc import LatinHypercube

from scipy.optimize import minimize, basinhopping, fmin, LinearConstraint

import post_process_GP_module_tensorflow as GPtf # tf functions to generate training data
import post_process_GP_module_64 as GPmodule
import warnings
from joblib import Parallel, delayed
import multiprocessing
import cloudpickle

tf.get_logger().setLevel('CRITICAL')
import functools



Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# we use double precision for the GP
torch.set_default_dtype(torch.float64)

# Create a folder to save models
os.makedirs(Parameters.LOG_DIR + "/individual_GP", exist_ok=True)
os.makedirs(Parameters.LOG_DIR + "/individual_GP/DK", exist_ok=True)
os.makedirs(Parameters.LOG_DIR + "/individual_GP/MA52", exist_ok=True)
os.makedirs(Parameters.LOG_DIR + "/individual_GP/SE", exist_ok=True)
os.makedirs(Parameters.LOG_DIR + "/individual_GP/SE/reinforced", exist_ok=True)
os.makedirs(Parameters.LOG_DIR + "/individual_GP/DK/reinforced", exist_ok=True)
# --------------------------------------------------------------------------- #
# MANUAL INPUTS


initial_samples = 2000 # N points in initial sample
if Parameters.MODEL_NAME == 'jpe_pseudostate_linear_transfers_risk_test_implied':
    Npoints = 800 
else:
    Npoints = 500
N_test_points = 500 # number of test points for evaluation

# Optimizer
learning_rate = 0.1 
# --------------------------------------------------------------------------- #
# rng for sampling
seed = 9534
# set torch seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# set numpy seed
np.random.seed(seed)

# set python random seed
import random
random.seed(seed)



rng = np.random.default_rng(seed)

# function to create welfare weights
def generate_weights(growth_rate,normalization = 1):
    # Generate exponential growth or shrinkage factor
    factors = (1 + growth_rate) ** np.arange(40)
    # Normalize the weights to sum to 1
    normalized_weights = factors / np.sum(factors) * normalization
    return normalized_weights

# defifne some welfare weights
welfare_weights = generate_weights(growth_rate= 0.0,normalization = 1)

if Parameters.MODEL_NAME in ['jpe_pseudostate_S_transfers_risk_test', 'jpe_pseudostate_S_transfers_risk_test_kappa']:
    # Constants for the problem
    S0 = Parameters.S0
    ST_range = Parameters.ST_range
    param_bounds = Parameters.param_bounds
    tax_range = Parameters.tax_range
    param_names = ['tau_const_x', 'tau_S_x', 'tshare1_x', 'tshare2_x', 'tshare3_x', 'tshare4_x', 'tshare5_x', 'tshare6_x', 'tshare7_x', 'tshare8_x', 'tshare9_x', 'tshare10_x', 'tshare11_x', 'tshare12_x']

    N_tax_params = 2  # number of tax parameters
    
    # Define the initial guess (0, 0.2 and 12 times 1/12)
    x0 = np.zeros(14)
    x0[0] = 0.0  # first parameter
    x0[1] = 0.2  # second parameter
    x0[2:] = 1/12  # remaining parameters set to 1/12
    
    def constraint_tax_0(params):
        # unnormalize here
        params_unnorm = params * std_X_init_np + mean_X_init_np
        tax_params = params_unnorm[:2]  # first two parameters are tax parameters
        tax_0 = sm.tax_function_lin_1(S0, tax_params)
        return tax_0
    
    def constraint_tax_T(params):
        # unnormalize params here
        params_unnorm = params * std_X_init_np + mean_X_init_np
        tax_params = params_unnorm[:2]  # first two parameters are tax parameters
        ST = ST_range[0]
        tax_T = sm.tax_function_lin_1(ST, tax_params)
        return tax_T

elif Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_test']:
    S0 = Parameters.S0
    ST_range = Parameters.ST_range
    param_bounds = Parameters.param_bounds
    tax_range = Parameters.tax_range
    param_names = ['tau_const_x', 'tau_S_x']
    x0 = np.zeros(2)
    x0[0] = 0.0  # first parameter
    x0[1] = 0.2  # second parameter

    # Constraint functions
    def constraint_tax_0(x):
        params_unnorm = x * std_X_init_np + mean_X_init_np
        tax_params = params_unnorm[:2]  # first four parameters are tax parameters
        tax_0 = sm.tax_function_lin_1(S0, tax_params)
        return tax_0  # Must be in [0, 1.5]

    def constraint_tax_T(x):
        params_unnorm = x * std_X_init_np + mean_X_init_np
        tax_params = params_unnorm[:2]  # first four parameters are tax parameters
        tax_T = sm.tax_function_lin_1(ST_range[0], tax_params)
        return tax_T  # Must be in [-0.3, 1.5]

elif Parameters.MODEL_NAME in ['jpe_pseudostate_linear_transfers_risk_test_implied']:
    # Constants for the problem
    S0, kappa0, tp0 = Parameters.S0, Parameters.kappa0, Parameters.TP0  # First period values
    ST_range, kappaT_range, tpT_range = Parameters.ST_range, Parameters.kappaT_range, Parameters.tpT_range  # Approximate upper bound from BAU scenario
    param_bounds = Parameters.param_bounds  # Parameter bounds
    tax_range = Parameters.tax_range  # Tax range

    # Now we need to normalize the values to match the tax function
    S0_norm = S0 / S0
    ST_range_norm = tuple(s/S0 for s in ST_range)  # Normalized upper bound
    kappa0_norm = kappa0 / kappa0
    kappaT_norm = tuple(kappaT / kappa0 for kappaT in kappaT_range)  # Normalized kappaT
    tp0_norm = (1.0-(tp0- 1.4467) / (3.5 - 1.4467))  # Normalized 1 - distance to tipping point
    tpT_range_norm = tuple((1.0-(tp - 3.0) / (3.5 - 1.4467)) for tp in tpT_range)  # Normalized 1 - distance to tipping assuming BAU average warming (3 degrees) and average tipping point (3.0)

    param_names = ['tau_const_x', 'tau_S_x', 'tau_kappa_x', 'tau_TP_x','tshare1_x', 'tshare2_x', 'tshare3_x', 'tshare4_x', 'tshare5_x', 'tshare6_x', 'tshare7_x', 'tshare8_x', 'tshare9_x', 'tshare10_x', 'tshare11_x', 'tshare12_x']

    # Define the initial guess (0, 0.2 and 12 times 1/12)
    x0 = np.zeros(16)
    x0[0] = 0.0  # first parameter
    x0[1] = 0.2  # second parameter
    x0[2] = 0.1  # third parameter (kappa)
    x0[3] = 0.0  # fourth parameter (tipping point)
    x0[4:] = 1/12  # remaining parameters set to 1/12

    # tax constraints
    # Constraint functions
    def constraint_tax_0(x):
        params_unnorm = x * std_X_init_np + mean_X_init_np
        tax_params = params_unnorm[:4]  # first four parameters are tax parameters
        tax_0 = sm.tax_function_lin_3(S0_norm, kappa0_norm, tp0_norm, tax_params)
        return tax_0  # Must be in [0, 1.5]

    def constraint_tax_T(x):
        params_unnorm = x * std_X_init_np + mean_X_init_np
        tax_params = params_unnorm[:4]  # first four parameters are tax parameters
        ST = ST_range_norm[0]
        tpT = tpT_range_norm[0]
        kappaT = kappaT_norm[0]
        tax_T = sm.tax_function_lin_3(ST, kappaT, tpT, tax_params)
        return tax_T  # Must be in [-0.3, 1.5]



else:
    raise ValueError("MODEL_NAME not implemented")



# --------------------------------------------------------------------------- #
# Training of the GP surrogate
# --------------------------------------------------------------------------- #
# First generate the training data for the GP surrogate and calculate mean and variance for normalization

# load training data
training_set = pd.read_csv(Parameters.LOG_DIR + "/GP_train_data_raw.csv").values


# select subset of training data
training_set = training_set[:initial_samples]



X_train = torch.tensor(training_set[:,:-40], dtype=torch.float64) # parameters, but we exclude the 12th transfer (redundant)
Y_train = torch.tensor(training_set[:,-40:], dtype=torch.float64) # welfares

# normalize features
mean_x = X_train.mean(dim=0, keepdim=True)
std_x = X_train.std(dim=0, keepdim=True) + 1e-6 # prevent dividing by 0
X_train = (X_train - mean_x) / std_x

# normalize labels
mean_y = Y_train.mean(dim=0, keepdim=True)
std_y = Y_train.std(dim=0, keepdim=True) + 1e-6 # prevent dividing by 0
Y_train = (Y_train - mean_y) / std_y

print("std_y: ", std_y.shape)


Nparams = X_train.shape[1]

mean_X_init_np = mean_x.numpy().squeeze()
std_X_init_np = std_x.numpy().squeeze()
mean_Y_init_np = mean_y.numpy().squeeze()
std_Y_init_np = std_y.numpy().squeeze()

# Split the data into training and validation sets
X_test = X_train[N_test_points:]
Y_test = Y_train[N_test_points:]
X_train = X_train[:Npoints]
Y_train = Y_train[:Npoints]

print("X_train_shape: ",X_train.shape)
print("Y_train_shape: ",Y_train.shape)
# create copy to use it for all models:
X_train_base = X_train
Y_train_base = Y_train

X_test_base = X_test
Y_test_base = Y_test

# Define the Deep Kernel
class DeepKernel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(DeepKernel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 32, dtype=torch.float64)
        self.fc3 = nn.Linear(32, 16, dtype=torch.float64)
        self.fc4 = nn.Linear(16, output_dim, dtype=torch.float64)
        self.activation = nn.SELU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Tanh()
        self.activation4 = nn.Sigmoid()
        self.activation5 = nn.GELU()



    def forward(self, x):
        x = x.to(torch.float64)  # Ensure input is float64
        x = self.activation5(self.fc1(x))
        x = self.activation5(self.fc2(x))
        x = self.activation5(self.fc3(x))
        x = self.fc4(x)
        return x

# ------------------------------------------------ 
# Model Training Loop: Train 40 models for each welfare value
# ------------------------------------------------
N_welfare_functions = 40  # Number of welfare functions to train on
X_train_base = X_train_base[:Npoints]  # Use the first Npoints for training
Y_train_base = Y_train_base[:Npoints]  # Use the first Npoints for training
X_train = X_train_base[:Npoints]
N_inputs = X_train_base.shape[1]  # Number of input features
print(f"Number of input features: {N_inputs}")
std_y_results = std_y[0,:]  # Assuming we want to use the first 10 welfare values for std calculation
X_test = X_test[:, :N_inputs]  # Print summary of results


# ------------------------------------------------------------- #
# Parallel Training Function
# ------------------------------------------------------------- #
def train_single_gp(welfare_idx, X_train_base, Y_train_base, Y_test_base, X_test, Npoints, Nparams, N_inputs, learning_rate, std_y_results, worker_seed):
    
    # Set seeds for this worker
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    torch.set_num_threads(1)
    
    # Extract training data
    X_train = X_train_base[:Npoints, :N_inputs]
    Y_train = Y_train_base[:Npoints, welfare_idx]
    Y_test = Y_test_base[:, welfare_idx]
    

    # Initialize likelihood and model
    # Use partial to bind N_inputs to the model class constructor
    model_cls = functools.partial(GPmodule.ExactGPMatern52, N_inputs=Nparams)
    
    # Train with restarts
    model, likelihood = GPmodule.train_with_restarts(
        model_cls=model_cls,
        likelihood_cls=gpytorch.likelihoods.GaussianLikelihood,
        mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood,
        X=X_train,
        F=Y_train,
        selected_criterion=1,
        num_restarts=10, 
        verbose=True
    )

    # Save best model
    output_file = Parameters.LOG_DIR + f"/individual_GP/MA52/welfare_surrogate{welfare_idx}.pcl"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'train_x': model.train_inputs[0],
        'train_y': model.train_targets,
        'N_inputs': Nparams
    }, output_file)

    
    model.eval()
    likelihood.eval()
    
    # metrics
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model(X_test)
        pred_mean = preds.mean
        squared_errors = (pred_mean - Y_test) ** 2
        rmse = torch.sqrt(torch.mean(squared_errors)).item()
        mse = torch.mean(squared_errors).item()
        mae = torch.mean(torch.abs(pred_mean - Y_test)).item()
        
    std_val = std_y_results[welfare_idx].item() if welfare_idx < len(std_y_results) else None
    
    print(f"[{welfare_idx}] Finished. RMSE: {rmse:.6f}")
    
    return {
        'welfare_idx': welfare_idx,
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'model_file': output_file,
        'std_original': std_val
    }

# ----------------------------------------------------------------- #
# Parallel Execution
# ----------------------------------------------------------------- #
print(f"\n{'='*60}")
print("Starting Parallel GP Training (12 jobs)")
print(f"{'='*60}")

# Use joblib to run in parallel
results_list = Parallel(n_jobs=12)(
    delayed(train_single_gp)(
        idx, X_train_base, Y_train_base, Y_test_base, X_test, Npoints, Nparams, N_inputs, learning_rate, std_y_results, worker_seed=seed + idx
    ) for idx in range(N_welfare_functions)

)

# Convert list to dict
results = {r['welfare_idx']: r for r in results_list}

print(f"\n{'='*60}")
print("TRAINING COMPLETE - SUMMARY OF ALL MODELS")
print(f"{'='*60}")

rows = []
print(f"{'Welfare':<8} {'RMSE':<12} {'MSE':<12} {'MAE':<12} {'Std_Y':<12}")
print("-" * 60)
for welfare_idx in range(N_welfare_functions):
    if welfare_idx in results:
        result = results[welfare_idx]
        std_val = result['std_original'] if result['std_original'] is not None else 'N/A'
        print(f"{welfare_idx:<8} {result['rmse']:<12.6f} {result['mse']:<12.6f} {result['mae']:<12.6f} {std_val}")
        rows.append({
            'Welfare': welfare_idx,
            'RMSE': result['rmse'],
            'MSE': result['mse'],
            'MAE': result['mae'],
            'Std_Y': std_val
        })
    else:
        print(f"Warning: Result for welfare index {welfare_idx} not found.")

df = pd.DataFrame(rows)
df.to_csv('welfare_model_summary.csv', index=False)

# Save summary results
import json
# convert numpy types to python types for json serialization
def convert_to_builtin_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

results_json = {k: {k2: convert_to_builtin_type(v2) for k2, v2 in v.items()} for k, v in results.items()}

summary_file = Parameters.LOG_DIR + "/individual_GP/training_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"\nSummary results saved to: {summary_file}")



