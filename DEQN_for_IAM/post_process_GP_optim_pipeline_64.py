import importlib
import pandas as pd
import numpy as np
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode, do_random_step
import sys, shutil, os
import time
import sampling_module as sm
import post_process_GP_module_64 as GPmodule
import pickle

import torch
import gpytorch
from scipy.stats.qmc import LatinHypercube
from smt.sampling_methods import LHS

from scipy.optimize import minimize, basinhopping, fmin, LinearConstraint

"""
Note: This is the main script for the optimization problem to find the optimal tax policy.
There are 3 parts which can be run independently:
1. The training: trains the GP surrogate
2. The optimization: uses the GP surrogate to find the optimal tax policy
3. The LOO error calculation: calculates the leave-one-out error of the GP surrogate (extremely costly!)
"""

tf.get_logger().setLevel('CRITICAL')


Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# we use double precision for the GP
torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------- #
# MANUAL INPUT
# --------------------------------------------------------------------------- #
# Baseline
path_welfare_BAU = 'runs/jpe_bau_final/final/sim_mean_welfare.csv'


# Run settings
Fit_GP = False
Optimize_Planner = True
Calculate_LOO = False # Attention: This is very expensive!

# --------------------------------------------------------------------------- #
# More settings
# --------------------------------------------------------------------------- #
# Sampling
initial_samples = 450 # if not sampled from scratch use subset
N_BAL_steps = 50
N_candidates_BAL = 1000 # number of candidates in each active learning step

# Optimizer
learning_rate = 0.1

# rng for sampling
seed = 9534

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)

# function to create welfare weights
def generate_welfare_weights(growth_rate: float, n_generations: int = 40, first_n_equal: int = 12):
    """
    Generate welfare weights:
    - First `first_n_equal` generations have equal weights summing to 1.
    - Remaining generations grow by `growth_rate` per generation.

    Args:
        growth_rate (float): Growth rate per generation (e.g., 0.02 for 2%).
        n_generations (int): Total number of generations (default 40).
        first_n_equal (int): Generations with equal weight (default 12).

    Returns:
        np.array: Welfare weights summing to 1.
    """
    # Equal weights for first generations
    equal_weight = 1 / first_n_equal
    weights = [equal_weight] * first_n_equal

    # Grow the rest geometrically
    for i in range(first_n_equal, n_generations):
        weight = weights[-1] * (1 + growth_rate)
        weights.append(weight)

    # Normalize to sum to 1
    weights = np.array(weights)
    weights /= weights.sum()

    return weights

welfare_weights = generate_welfare_weights(growth_rate=0.0, n_generations=40, first_n_equal=12)


# --------------------------------------------------------------------------- #
# house keeping and setup
# --------------------------------------------------------------------------- #




if Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_scratch']:
    S0 = Parameters.S0
    ST_range = Parameters.ST_range
    param_bounds = Parameters.param_bounds
    tax_range = Parameters.tax_range
    param_names = ['tau_const_x', 'tau_S_x']



else:
    raise ValueError("MODEL_NAME not implemented")

# load welfare values from BAU scenario
df_BAU = pd.read_csv(path_welfare_BAU)
welfares_BAU = np.squeeze(df_BAU.values[:40])



# # --------------------------------------------------------------------------- #
# # Training of the GP surrogate
# # --------------------------------------------------------------------------- #

# load training data
training_set = pd.read_csv(Parameters.LOG_DIR + "/GP_train_data_raw.csv").values


# select subset of training data
training_set = training_set[:initial_samples]

print("Welfares BAU shape", welfares_BAU.shape)


print("training_set welfares", training_set[:,-40:])

# define training data for the GP
X_train = torch.tensor(training_set[:,:-40], dtype=torch.float64)
Y_train = torch.tensor(training_set[:,-40:] @ welfare_weights, dtype=torch.float64)

print("X_train_shape",X_train.shape)

# normalize features
mean_X_init = X_train.mean(dim=0, keepdim=True)
std_X_init = X_train.std(dim=0, keepdim=True) 
X_train = (X_train - mean_X_init) / std_X_init

# normalize labels
mean_Y_init, std_Y_init = Y_train.mean(), Y_train.std()
Y_train = (Y_train - mean_Y_init) / std_Y_init

# create copy to use same initial conditions for all models:
X_train_base = X_train
Y_train_base = Y_train

Nparams = X_train.shape[1]

mean_X_init_np = mean_X_init.numpy()
std_X_init_np = std_X_init.numpy()
mean_Y_init_np = mean_Y_init.numpy()
std_Y_init_np = std_Y_init.numpy()

# ------------------------------------------------ 
# Model 1: Squared exponential kernel
# ------------------------------------------------
if Fit_GP:
    # create likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = GPmodule.ExactGPModel(X_train, Y_train, likelihood, N_inputs=Nparams)
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    print("-------------------")
    print("training")
    print("-------------------")
    
    model, likelihood = GPmodule.train_with_restarts(model_cls = lambda X, F, lik: GPmodule.ExactGPModel(X, F, lik, N_inputs=Nparams), likelihood_cls=gpytorch.likelihoods.GaussianLikelihood, mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood, X=X_train, F=Y_train, selected_criterion=1, num_restarts=10, verbose=True)

    # Re-initialize MLL and optimizer for the new model for active learning
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("-------------------")
    print("Active Learning")
    print("-------------------")

    # active learning

    for i in range(N_BAL_steps):
        print('iteration ', i)
        
        
        if Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_scratch']:
            candidates = sm.sample_params_lin1(num_samples=N_candidates_BAL, x0=S0, xT_range=ST_range, param_bounds=param_bounds, tax_range=tax_range, rng=rng, batch_size=1000)
            X_train, Y_train = GPmodule.active_learning_step_NEW(model=model, likelihood=likelihood, X=X_train, F=Y_train, candidates=candidates, welfare_weights=welfare_weights, X_mean = mean_X_init, X_std = std_X_init, F_mean = mean_Y_init, F_std = std_Y_init,param_names=param_names, num_new_points=1, N_generations=40)

        
        # train model
        model.set_train_data(inputs=X_train, targets=Y_train, strict=False)
        GPmodule.train(model, likelihood, optimizer, mll, X_train, Y_train, selected_criterion=1, verbose=True)


    # save model
    output_file = Parameters.LOG_DIR +"/welfare_surrogate_SE" + ".pcl"
    print(output_file )
    with open(output_file, 'wb') as fd:
        pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print("GP model written to disk")
        print(" -------------------------------------------")
    fd.close()

# End of GP Fit

# --------------------------------------------------------------------------- #
# Calculate the LOO error (extremely costly!)
# --------------------------------------------------------------------------- #
if Calculate_LOO:
    # load model
    output_file = Parameters.LOG_DIR +"/welfare_surrogate_SE" + ".pcl"

    print(output_file )
    with open(output_file, 'rb') as fd:
        model = pickle.load(fd)
        print("GP model loaded from disk")
        print("-------------------------------------------")
    fd.close()

    # LOO error 
    print("-------------------")
    print("Calculating LOO error")

   

    loo_error, loo_list = GPmodule.compute_loo_error(model, criterion=1, learning_rate=learning_rate)
    print(f"LOO error: {loo_error}")
    print("loo_list: ", loo_list)
    print("---------------------------------")



    # save loo_list as dataframe
    loo_list_df = pd.DataFrame(loo_list)
    output_file = Parameters.LOG_DIR +"/loo_list_SE" + ".csv"
    loo_list_df.to_csv(output_file, index=False)

# End of LOO error calculation

# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #

if Optimize_Planner: 

    output_file = Parameters.LOG_DIR +"/welfare_surrogate_SE" + ".pcl"
    print(output_file )
    with open(output_file, 'rb') as fd:
        model_SE = pickle.load(fd)
        print("GP model loaded from disk")
        print(" -------------------------------------------")
    fd.close()



    # -------------------------------------------------------------------------- #
    # tax linear in S
    # -------------------------------------------------------------------------- #


    if Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_scratch']:
        def constraint_tax_0(params):
            # unnormalize here
            params_unnorm = params * std_X_init_np + mean_X_init_np
            tax_0 = sm.tax_function_lin_1(S0, params_unnorm)
            return tax_0

       
        
        def constraint_tax_T(params):
            # unnormalize params here
            params_unnorm = params * std_X_init_np + mean_X_init_np
            S_min = ST_range
            tax_T_min = sm.tax_function_lin_1(S_min, params_unnorm)
            return tax_T_min
        
        
        # Constraints as dictionaries for scipy
        constraints = [
            {'type': 'ineq', 'fun': lambda params: constraint_tax_0(params) - Parameters.tax_range[0]},  # tax_0 >= lb
            {'type': 'ineq', 'fun': lambda params: Parameters.tax_range[1] - constraint_tax_0(params)},  # tax_0 <= ub
            {'type': 'ineq', 'fun': lambda params: constraint_tax_T(params) - Parameters.tax_range[0]},  # tax_T_min >= lb
            {'type': 'ineq', 'fun': lambda params: Parameters.tax_range[1] - constraint_tax_T(params)},  # tax_T_min <= ub
        ]
        print(" ------------------------------------------------ ")
        print("Optimize with basin hopping ")


        # Basinhopping
        initial_guess = [-0.2,0.5]   # Initial guess
        bounds_optim = np.array([[-2., 2.],[-2.,2.]])
        bounds_optim = (bounds_optim - mean_X_init_np.T)/std_X_init_np.T
        bounds_optim = tuple(map(tuple, bounds_optim))

        result_SE = basinhopping(
            GPmodule.objective,
            initial_guess,
            minimizer_kwargs={
                'method': 'SLSQP',
                'bounds': bounds_optim,
                'constraints': constraints,
                'args': (model_SE),
                'options': {'ftol': 1e-5}
            },
            niter=500,
            seed=seed
        )

    # -------------------------------------------------------------------------- #
    # optimization result
    # -------------------------------------------------------------------------- #


    print(" ------------------------------------------------")
    print(" Result squared exponential")
    print("result normalized", result_SE.x)
    print("Parameters: ")
    print(result_SE.x * std_X_init_np + mean_X_init_np)
    print("Best value: ")
    print(GPmodule.evaluate_model(result_SE.x,model=model_SE) * std_Y_init_np + mean_Y_init_np)
    print(" ------------------------------------------------")
    print("Constraints check:")
    print("tax_0: ", constraint_tax_0(result_SE.x))
    print(" ------------------------------------------------")

    # -------------------------------------------------------------------------- #
    # Save results to disk
    # -------------------------------------------------------------------------- #

    # Extract unnormalized parameters and best function value
    best_params_unnorm = result_SE.x * std_X_init_np + mean_X_init_np
    best_val_unnorm = GPmodule.evaluate_model(result_SE.x, model=model_SE) * std_Y_init_np + mean_Y_init_np

    # 1. Save summary to TXT file
    txt_output_path = os.path.join(Parameters.LOG_DIR, "optimization_results_summary.txt")
    with open(txt_output_path, "w") as f:
        f.write("Optimization Results Summary (Squared Exponential Surrogate)\n")
        f.write("="*60 + "\n")
        f.write(f"Model Name        : {Parameters.MODEL_NAME}\n")
        f.write(f"Best Welfare Value: {best_val_unnorm.item():.6f}\n")
        f.write("\nUnnormalized Optimal Parameters:\n")
        for i, p in enumerate(best_params_unnorm.flatten()):
            f.write(f"  Param {i}: {p:.6f}\n")
        f.write("\nConstraints Check:\n")
        f.write(f"  tax_0: {constraint_tax_0(result_SE.x)}\n")
    print(f"-> Text summary saved to {txt_output_path}")

    # 2. Save optimal parameters to CSV file
    csv_output_path = os.path.join(Parameters.LOG_DIR, "optimal_parameters.csv")
    # Use param_names if it's defined in the script
    try:
        cols = param_names
    except NameError:
        cols = [f"Param_{i}" for i in range(len(best_params_unnorm.flatten()))]

    df_params = pd.DataFrame([best_params_unnorm.flatten()], columns=cols)
    df_params.to_csv(csv_output_path, index=False)
    print(f"-> Optimal parameters saved to {csv_output_path}")

# End of planner problem