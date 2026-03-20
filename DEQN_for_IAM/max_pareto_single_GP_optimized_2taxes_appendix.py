
import importlib
import pandas as pd
import numpy as np
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions

import sys, shutil, os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
import psutil

import time
import sampling_module as sm
import tensorflow as tf

from itertools import product
import pickle

import torch
import torch.nn as nn

torch.set_num_threads(1)  

import gpytorch
from torch.linalg import vector_norm, norm

from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize, basinhopping, fmin, LinearConstraint

import multiprocessing as mp
# Force Linux to act like Mac
try:
    mp.set_start_method('forkserver', force=True)
except RuntimeError:
    pass

from joblib import Parallel, delayed



import post_process_GP_module_tensorflow as GPtf # tf functions to generate training data
import post_process_GP_module_64 as GPmodule

import tempfile, gc

from post_process_GP_module_64_worker import train_gp_worker_v2

tf.get_logger().setLevel('CRITICAL')


Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")




# we use double precision for the GP
torch.set_default_dtype(torch.float64)

seed = 92
# set torch seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# set numpy seed
np.random.seed(seed)

# set python random seed
import random
random.seed(seed)

# set tensorflow seed
tf.random.set_seed(seed)

# rng for sampling (new numpy API)
rng = np.random.default_rng(seed)
# --------------------------------------------------------------------------- #
# MANUAL INPUT

# path BAU welfares
path_welfare_BAU = 'runs/jpe_bau_final/final/sim_mean_welfare.csv'
# path_welfare_BAU = 'runs/jpe_bau_final/high_kappa/sim_mean_welfare.csv'



START_BAL = False
FINAL_OPTIMIZATION = True
KERNEL_CHOICE = 'MA52' 
# --------------------------------------------------------------------- #

N_WORKERS = 4

if Parameters.MODEL_NAME in ['jpe_pseudostate_S_transfers_risk_test','jpe_pseudostate_S_transfers_risk_test_kappa']:
    initial_samples = 2000 # if not sampled from scratch use subset
    Npoints = 500 
    N_test_points = 500 
    N_BAL_STEPS = 500 # number of pareto trials
    learning_rate = 0.1 
else:
    raise ValueError("MODEL_NAME not implemented")




CHECKPOINT_DIR_BASE = Parameters.LOG_DIR + f"/individual_GP/{KERNEL_CHOICE}/"

# Create a folder to save models
os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_BASE + "reinforced", exist_ok=True)



# function to create welfare weights
def generate_weights(growth_rate,normalization = 1):
    # Generate exponential growth or shrinkage factor
    factors = (1 + growth_rate) ** np.arange(40)
    # Normalize the weights to sum to 1
    normalized_weights = factors / np.sum(factors) * normalization
    return normalized_weights

# defifne some welfare weights
welfare_weights = generate_weights(growth_rate= 0.0,normalization = 1)

if Parameters.MODEL_NAME in ['jpe_pseudostate_S_transfers_risk_test']:
    # Constants for the problem
    S0 = Parameters.S0
    ST_range = Parameters.ST_range
    param_bounds = Parameters.param_bounds
    tax_range = Parameters.tax_range
    param_names = ['tau_const_x', 'tau_S_x', 'tshare1_x', 'tshare2_x', 'tshare3_x', 'tshare4_x', 'tshare5_x', 'tshare6_x', 'tshare7_x', 'tshare8_x', 'tshare9_x', 'tshare10_x', 'tshare11_x', 'tshare12_x']

    N_tax_params = 2  # number of tax parameters
    
    # Define the initial guess (0, 0.2 and 12 times 1/12)
    x0 = np.zeros(14)
    x0[0] = -0.2  # first parameter
    x0[1] = 0.35  # second parameter
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

elif Parameters.MODEL_NAME in ['jpe_pseudostate_S_transfers_risk_test_kappa']:
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


else:
    raise ValueError("MODEL_NAME not implemented")

# load welfare values from BAU scenario
df_BAU = pd.read_csv(path_welfare_BAU)
welfares_BAU = np.squeeze(df_BAU.values[:40])

# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #

# load training data
training_set = pd.read_csv(Parameters.LOG_DIR + "/GP_train_data_raw.csv").values

# select subset of training data
training_set = training_set[:initial_samples]


X_train = torch.tensor(training_set[:,:-40], dtype=torch.float64) # parameters
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


# parallel training
def run_parallel_training(models, likelihoods, X_train_base, Y_train_base,
                           learning_rate, N_WORKERS):
    shared_path = os.path.join(tempfile.gettempdir(), "gp_shared_train_4fast.pt")
    torch.save({
        'X_train_base': X_train_base,
        'Y_train_base': Y_train_base,
    }, shared_path)

    worker_args = []
    for w_idx, (m, l) in enumerate(zip(models, likelihoods)):
        worker_args.append({
            'welfare_idx': w_idx,
            'model_state': {k: v.cpu().clone() for k, v in m.state_dict().items()},
            'likelihood_state': {k: v.cpu().clone() for k, v in l.state_dict().items()},
        })

    results = Parallel(n_jobs=N_WORKERS, backend='loky')(
        delayed(train_gp_worker_v2)(
            a['welfare_idx'], a['model_state'], a['likelihood_state'],
            shared_path, learning_rate, KERNEL_CHOICE, False
        )
        for a in worker_args
    )
    os.remove(shared_path)
    return results




def clear_gp_caches(models):
    """Force GPyTorch to drop cached kernel matrices."""
    for m in models:
        m.train()   # Entering train mode clears prediction caches
        m.eval()    # Back to eval for the next optimization
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def check_memory(threshold_gb=55):
    """Warn and trigger GC if memory usage is high."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    print(f"Memory: {used_gb:.1f} / {total_gb:.1f} GB ({mem.percent}%)")
    if used_gb > threshold_gb:
        print("WARNING: Memory pressure detected, running GC...")
        gc.collect()
        return True
    return False


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
X_test = X_test[:, :N_inputs]  # Ensure X_test has the same number of features as X_train





# --------------------------------------------------------------------------- #
# Optimization of the welfare functions
# --------------------------------------------------------------------------- #

# load GP models
def load_all_models(checkpoint_dir=CHECKPOINT_DIR_BASE, load_reinforced=False):
    models = []
    likelihoods = []
    
    for i in range(N_welfare_functions):
        # Construct the full path to the saved checkpoint
        if load_reinforced:
            checkpoint_path = checkpoint_dir + f"reinforced/welfare_surrogate_{i}.pth"
        else:
            checkpoint_path = checkpoint_dir + f"welfare_surrogate{i}.pcl"
        

        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Reconstruct model and likelihood (architecture must match original!)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()  # Or your custom likelihood

        if KERNEL_CHOICE == 'SE':
             model = GPmodule.ExactGPModel(
                checkpoint['train_x'], 
                checkpoint['train_y'], 
                likelihood,
                N_inputs=N_inputs
            )

        elif KERNEL_CHOICE == 'MA52':
            # Matern 5/2
            model = GPmodule.ExactGPMatern52(
                checkpoint['train_x'], 
                checkpoint['train_y'], 
                likelihood,
                N_inputs=N_inputs
            )

        
        else:
            raise ValueError(f"Unknown KERNEL_CHOICE: {KERNEL_CHOICE}")
        
        # Load the saved states
        model.load_state_dict(checkpoint['model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        likelihood.eval()
        
        models.append(model)
        likelihoods.append(likelihood)
    
    return models, likelihoods

def evaluate_all_gps(models, x):
    # x should have shape (1, d)
    x = torch.tensor(x, dtype=torch.float64)  # ensure x is a tensor
    if x.dim() == 1:
        x = x.unsqueeze(0)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = model(x)
            preds.append(pred.mean.item())
    return np.array(preds)  # shape (40,)



# ------------------------------------------------------------- #
# GPOptimizationWrapper
# ------------------------------------------------------------- #
class GPOptimizationWrapper:
    def __init__(self, models, welfare_weights, mean_y, std_y, BAU_welfares):
        self.models = models
        self.welfare_weights = torch.tensor(welfare_weights, dtype=torch.float64)
        self.mean_y = torch.tensor(mean_y, dtype=torch.float64)
        self.std_y = torch.tensor(std_y, dtype=torch.float64)
        self.BAU_welfares = torch.tensor(BAU_welfares, dtype=torch.float64)
        
        # Caches
        self.last_x = None
        self.cached_result = None
        self.eval_count = 0 
        self.cache_hit_count = 0
        
    def _ensure_eval_mode(self):
        for model in self.models:
            model.eval()

    
    def compute_all_fast(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.cached_result

        self._ensure_eval_mode()

        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        x_batch = x_tensor.unsqueeze(0) if x_tensor.dim() == 1 else x_tensor

        mus_list = []
        with gpytorch.settings.max_cholesky_size(2000), \
             gpytorch.settings.skip_posterior_variances(True), \
             gpytorch.settings.debug(False):
            for model in self.models:
                pred = model(x_batch).mean
                mus_list.append(pred)

        mus_cat = torch.cat(mus_list)
        obj = -torch.dot(self.welfare_weights, mus_cat)

        # --- SINGLE backward for obj_grad (replaces 40 retain_graph calls) ---
        obj.backward(retain_graph=True)  # retain_graph needed only for jac_mus below
        obj_grad_numpy = x_tensor.grad.detach().numpy().flatten().copy()
        x_tensor.grad.zero_()

        # --- Jacobian for pareto constraint: still 40 passes, but now separated ---
        jac_rows = []
        for mu in mus_list:
            g = torch.autograd.grad(mu, x_tensor, retain_graph=True)[0]
            jac_rows.append(g.detach().numpy().flatten())
        jac_numpy = np.stack(jac_rows)

        mus_numpy = mus_cat.detach().numpy()

        result = {
            'mus': mus_numpy,
            'obj_val': obj.item(),
            'obj_grad': obj_grad_numpy,
            'jac_mus': jac_numpy,
        }

        self.last_x = np.copy(x)
        self.cached_result = result
        return result

    def compute_means_and_obj(self, x):
        """Fast path: only obj_val + obj_grad, no Jacobian."""
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.cached_result

        self._ensure_eval_mode()
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        x_batch = x_tensor.unsqueeze(0)

        mus_list = []
        with gpytorch.settings.max_cholesky_size(2000), \
             gpytorch.settings.skip_posterior_variances(True), \
             gpytorch.settings.debug(False):
            for model in self.models:
                mus_list.append(model(x_batch).mean)

        mus_cat = torch.cat(mus_list)
        obj = -torch.dot(self.welfare_weights, mus_cat)
        obj.backward(retain_graph=True)

        result = {
            'mus': mus_cat.detach().numpy(),
            'obj_val': obj.item(),
            'obj_grad': x_tensor.grad.detach().numpy().flatten().copy(),
            'mus_list': mus_list,       # kept for lazy jac computation
            'x_tensor': x_tensor,      # kept for lazy jac computation
            'jac_mus': None,            # computed on demand
        }
        self.last_x = np.copy(x)
        self.cached_result = result
        return result

    def _ensure_jac(self, res):
        """Compute jac_mus only if not yet cached."""
        if res['jac_mus'] is not None:
            return
        jac_rows = []
        for mu in res['mus_list']:
            g = torch.autograd.grad(mu, res['x_tensor'], retain_graph=True)[0]
            jac_rows.append(g.detach().numpy().flatten())
        res['jac_mus'] = np.stack(jac_rows)

    def objective_func(self, x):
        res = self.compute_means_and_obj(x)
        return res['obj_val'], res['obj_grad']   # never triggers jac computation

    def pareto_constraint_jac(self, x):
        res = self.compute_means_and_obj(x)
        self._ensure_jac(res)                    # computed only when SLSQP needs it
        std_y_np = self.std_y.numpy()
        return res['jac_mus'] * std_y_np[:, np.newaxis]

    def pareto_constraint_func(self, x):
        res = self.compute_means_and_obj(x)
        mus_unnorm = res['mus'] * self.std_y.numpy() + self.mean_y.numpy()
        return mus_unnorm - self.BAU_welfares.numpy() - 5e-5



# transfers sum to 1 constraint
def sum_transfers(x):
        params = x * std_X_init_np + mean_X_init_np
        transfers = params[-12:] # last 12 parameters are transfers
        return np.sum(transfers) - 1.0

def sum_transfers_jac(x):
    # Gradient of sum_transfers
    grad = np.zeros_like(x)
    grad[-12:] = std_X_init_np[-12:]
    return grad





# ------------------------------------------------------------- #

# transfer bounds taken from sampling the dirichlet distribution
transfer_bounds = np.array([[7.5158729e-05, 4.6898937e-01],
    [1e-06, 5.5360019e-01],
    [9.6779549e-06, 5.0776035e-01],
    [3.0654608e-08, 3.6772752e-01],
    [5.3107953e-03, 7.2037429e-01],
    [1e-06, 5.5230325e-01],
    [4.8257571e-09, 4.8253495e-01],
    [9.8546769e-04, 5.8679134e-01],
    [6.2523017e-05, 4.3266138e-01],
    [4.0011150e-07, 4.6021074e-01],
    [6.3756154e-08, 4.1951174e-01],
    [1.1455669e-04, 4.6401733e-01]])

# tax bounds
tax_bounds = param_bounds * 0.9
bounds_optim = np.concatenate((tax_bounds, transfer_bounds), axis=0)
bounds_optim = (bounds_optim - mean_X_init_np[:,None])/std_X_init_np[:,None]
optim_bounds = tuple(map(tuple, bounds_optim))



# Normalize the initial guess
x0 = (x0 - mean_X_init_np) / std_X_init_np  # normalize initial guess





if START_BAL:
    def print_iteration(x, f, accept):
        print(f"Current x: {x}")
        print(f"Function value: {f}")
        print(f"Accepted: {accept}")
        print("-" * 40)

    

    # Load all models into a list
    models, likelihoods = load_all_models()

    
    # test
    test_eval = evaluate_all_gps(models, x0)
    print("Test evaluation of initial guess:", test_eval)

    for i in range(N_BAL_STEPS):
        print("---------------------------")
        print(f"BO Iteration {i+1}")
        print("---------------------------")

        clear_gp_caches(models)
        check_memory(threshold_gb=55)

        # time the optimization
        start_time = time.time()

        # Initialize wrapper
        gp_optimizer = GPOptimizationWrapper(models, welfare_weights, mean_Y_init_np, std_Y_init_np, welfares_BAU)

       
        # constraints list logic
        pareto_const = {'type': 'ineq', 'fun': gp_optimizer.pareto_constraint_func, 'jac': gp_optimizer.pareto_constraint_jac}
        transfer_const = {'type': 'eq', 'fun': sum_transfers, 'jac': sum_transfers_jac}
        

        
        basic_constraints = []

        # transfers sum to 1 and bounds for tax rate
        basic_constraints = [
            transfer_const,
            {'type': 'ineq', 'fun': lambda params: constraint_tax_0(params) - Parameters.tax_range[0]},  
            {'type': 'ineq', 'fun': lambda params: Parameters.tax_range[1] - constraint_tax_0(params)},
            {'type': 'ineq', 'fun': lambda params: constraint_tax_T(params) - (Parameters.tax_range[0])},  
            {'type': 'ineq', 'fun': lambda params: Parameters.tax_range[1]*0.95 - constraint_tax_T(params)},  
        ]
            
        constraints_opt = [pareto_const] + basic_constraints

        
        # basinhopping
        result = basinhopping(
            gp_optimizer.objective_func,
            x0,
            minimizer_kwargs={
                'method': 'SLSQP',
                'bounds': optim_bounds,
                'constraints': constraints_opt,
                'jac': True,
                'options': {'ftol': 1e-5}
            },
            niter=5,
            callback=print_iteration,
            seed = seed
        )
        end_time = time.time()
        print(f"Optimization took {end_time - start_time:.2f} seconds")


        # extract the parameters and train again
        candidates = result.x * std_X_init_np + mean_X_init_np
        print("Candidates (unnormalized):", candidates)
        # ensure candidates is a 2D array
        candidates = candidates.reshape(1, -1)  # shape (1, Nparams)
        # update the training set

        training_set = GPtf.gen_train_data_general_NEW(Pseudostate_samples=candidates, param_names=param_names, N_generations=40, N_batches=10000, verbose=False)
        # append new points to X_train and Y_train
        X_train_new = torch.tensor(training_set[:,:-40], dtype=torch.float64)  # parameters

        Y_train_new_np = training_set[:,-40:]  # welfares
        Y_train_new = torch.tensor(Y_train_new_np, dtype=torch.float64)  # welfares
        print("Y_train_new: ", Y_train_new)

        # break if we have all 40 welfare values > 0 with safety margin for simulation stability
        if np.all(Y_train_new_np - welfares_BAU >=-2e-5): 
            print("All welfare values are positive, stopping training.")
            print("Found candidates: ", candidates)
            print("Saving models")

            # Save optimization result and parameters
            os.makedirs(Parameters.LOG_DIR + "/pareto_models", exist_ok=True)
            save_path_success = Parameters.LOG_DIR + "/pareto_models/optimization_success.txt"
            with open(save_path_success, 'w') as f:
                f.write("Optimization Successful!\n")
                f.write(f"Parameters (unnormalized): {candidates.tolist()}\n")
                f.write(f"Welfares: {Y_train_new_np.tolist()}\n")
                f.write(f"Welfares BAU: {welfares_BAU.tolist()}\n")
                
            
            # Also save parameters as numpy array for easier loading
            np.save(Parameters.LOG_DIR + "/pareto_models/optimal_params.npy", candidates)
            
            print(f"Optimization results saved to {save_path_success}")

            # save the models
            for welfare_idx, (model, likelihood) in enumerate(zip(models, likelihoods)):
                output_file = CHECKPOINT_DIR_BASE + f"reinforced/welfare_surrogate_{welfare_idx}.pth"
                torch.save({
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'train_x': model.train_inputs[0],  # Extract from ExactGP
            'train_y': model.train_targets,
            'N_inputs': N_inputs  # Store the number of inputs
        }, output_file)
            break
        
        # continue training if result was not pareto improving
        # normalize new features
        X_train_new = (X_train_new - mean_x) / std_x
        # normalize new labels
        Y_train_new = (Y_train_new - mean_y) / std_y
        # concatenate the new training data with the existing training data
        X_train_base = torch.cat((X_train_base, X_train_new), dim=0)
        Y_train_base = torch.cat((Y_train_base, Y_train_new), dim=0)
        # print the shapes of the new training data
        print("New X_train shape: ", X_train_base.shape)
        X_train = X_train_base

  
        results = run_parallel_training(models, likelihoods, X_train_base, Y_train_base,
                          learning_rate, N_WORKERS)
        
        # Load the updated states back to the main worker's model representation
        for w_idx, state_data in results:
            models[w_idx].load_state_dict(state_data['model_state_dict'])
            likelihoods[w_idx].load_state_dict(state_data['likelihood_state_dict'])
            models[w_idx].set_train_data(inputs=state_data['train_x'], targets=state_data['train_y'], strict=False)
            
            # save model every single iteration (as requested)
            output_file_pth = CHECKPOINT_DIR_BASE + f"reinforced/welfare_surrogate_{w_idx}.pth"
            output_file_pcl = CHECKPOINT_DIR_BASE + f"reinforced/welfare_surrogate_{w_idx}.pcl"
            
            print(f"Saving model to: {output_file_pth} and {output_file_pcl}")
            
            # Construct the state checkpoint to save
            checkpoint = {
                'model_state_dict': models[w_idx].state_dict(),
                'likelihood_state_dict': likelihoods[w_idx].state_dict(),
                'train_x': models[w_idx].train_inputs[0],  # Extract from ExactGP
                'train_y': models[w_idx].train_targets,
                'N_inputs': N_inputs  # Store the number of inputs
            }

            torch.save(checkpoint, output_file_pth)
            torch.save(checkpoint, output_file_pcl)

    print("We have finished the optimization loop.")

# Final optimization for replication purposes
if FINAL_OPTIMIZATION:
    print("Starting final optimization...")
    
    models, likelihoods = load_all_models(load_reinforced=True)
    
    # Run the final optimization
    # Initialize wrapper
    gp_optimizer = GPOptimizationWrapper(models, welfare_weights, mean_Y_init_np, std_Y_init_np, welfares_BAU)

    # constraints list logic
    pareto_const = {'type': 'ineq', 'fun': gp_optimizer.pareto_constraint_func, 'jac': gp_optimizer.pareto_constraint_jac}
    transfer_const = {'type': 'eq', 'fun': sum_transfers, 'jac': sum_transfers_jac}
    
   
    
    basic_constraints = []
    
    # includes sum_transfers
    basic_constraints = [
        transfer_const,
        {'type': 'ineq', 'fun': lambda params: constraint_tax_0(params) - Parameters.tax_range[0]}, 
        {'type': 'ineq', 'fun': lambda params: Parameters.tax_range[1] - constraint_tax_0(params)},
        {'type': 'ineq', 'fun': lambda params: constraint_tax_T(params) - (Parameters.tax_range[0])},  
        {'type': 'ineq', 'fun': lambda params: Parameters.tax_range[1]*0.95 - constraint_tax_T(params)},  
    ]
         
    constraints_opt = [pareto_const] + basic_constraints

    
    # basinhopping
    result = basinhopping(
        gp_optimizer.objective_func,
        x0,
        minimizer_kwargs={
            'method': 'SLSQP',
            'bounds': optim_bounds,
            'constraints': constraints_opt,
            'jac': True,
            'options': {'ftol': 1e-5}
        },
        niter=5,
        # callback=print_iteration,
        seed = seed
    )

    # extract the parameters 
    candidates = result.x * std_X_init_np + mean_X_init_np
    print("Candidates (unnormalized):", candidates)
    # ensure candidates is a 2D array
    candidates = candidates.reshape(1, -1)  # shape (1, Nparams)


    # Save optimization result and parameters
    os.makedirs(Parameters.LOG_DIR + "/pareto_models", exist_ok=True)
    save_path_success = Parameters.LOG_DIR + "/pareto_models/optimization_success.txt"
    with open(save_path_success, 'w') as f:
        f.write("Optimization Successful!\n")
        f.write(f"Parameters (unnormalized): {candidates.tolist()}\n")



    # Also save parameters as numpy array for easier loading
    np.save(Parameters.LOG_DIR + "/pareto_models/optimal_params.npy", candidates)

    # save them also as csv with parameter names
    df = pd.DataFrame(candidates, columns=param_names)
    df.to_csv(Parameters.LOG_DIR + "/pareto_models/optimal_params.csv", index=False)