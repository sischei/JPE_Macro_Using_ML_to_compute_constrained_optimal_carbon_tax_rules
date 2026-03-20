
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from scipy.stats import uniform


import sys, shutil, os
import time
from scipy.stats import qmc
from torch.linalg import vector_norm, norm
import pickle

import torch
import torch.nn as nn
import gpytorch
from scipy.stats.qmc import LatinHypercube


# we use double precision for the GP
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

'''
This script is used to speed up the BAL GP training for Pareto improvements by parallelizing the GP training
'''

# ------------------------------------------------
# Define model classes
# ------------------------------------------------

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPMatern12(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPMatern12, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPMatern32(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPMatern32, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPMatern52(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPMatern52, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPPP1(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPPP1, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=1,ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPPP2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPPP2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=2,ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPPP3(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs):
        super(ExactGPPP3, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=3,ard_num_dims=N_inputs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the Deep Kernel
class DeepKernel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(DeepKernel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 32, dtype=torch.float64)
        self.fc3 = nn.Linear(32, 16, dtype=torch.float64)
        self.fc4 = nn.Linear(16, output_dim, dtype=torch.float64)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.to(torch.float64)  # Ensure input is float64
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# GP with learned deep features
class DKLGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, N_inputs):
        super(DKLGP, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=N_inputs))

    def forward(self, x):
        deep_features = self.feature_extractor(x)
        mean_x = self.mean_module(deep_features)
        covar_x = self.covar_module(deep_features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ------------------------------------------------
# define training functions
# ------------------------------------------------


# Train the GP model, includes stopping criteria
def train_worker(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=True):
    # Stopping criteria configurations
    STOP_CRITERIA = {
        "change_in_loss": {"enabled": selected_criterion == 1, "threshold": 1e-4},
        "gradient_norm": {"enabled": selected_criterion == 2, "threshold": 1e-3},
        "change_in_parameters": {"enabled": selected_criterion == 3, "threshold": 1e-4}
    }

    # Print enabled stopping criterion
    criterion_names = {
        1: "Change in Loss",
        2: "Gradient Norm",
        3: "Change in Parameters"
    }
    #print(f"Enabled Stopping Criterion: {criterion_names[selected_criterion]}")
    model.train()
    likelihood.train()
    prev_loss = None
    prev_params = [param.clone() for param in model.parameters()]

    # Track the number of optimization steps
    steps = 0
    with gpytorch.settings.max_cholesky_size(2000):
        for i in range(4000):
            optimizer.zero_grad()
            output = model(X)
            # print('F: ', F)
            loss = -mll(output, F)
            loss.backward()
            
            if STOP_CRITERIA["gradient_norm"]["enabled"]:
                grad_norm = vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
                if grad_norm < STOP_CRITERIA["gradient_norm"]["threshold"]:
                    if verbose:
                        print(f"Stopping: Gradient norm < {STOP_CRITERIA['gradient_norm']['threshold']} at step {steps}")
                    return loss.item()
                    #break
            
            optimizer.step()
            
            steps += 1
            if i % 100 == 0 and verbose:
                print("-------------------")
                print("epoch: ", i)
                print(f"Loss: {loss.item()}")
                print("-------------------")
            
            if STOP_CRITERIA["change_in_loss"]["enabled"] and prev_loss is not None:
                if abs(prev_loss - loss.item()) < STOP_CRITERIA["change_in_loss"]["threshold"]:
                    if verbose:
                        print(f"Stopping: Change in loss < {STOP_CRITERIA['change_in_loss']['threshold']} at step {steps}")
                        print("loss: ", loss.item())
                    return loss.item()
                    # break
            
            if STOP_CRITERIA["change_in_parameters"]["enabled"]:
                max_param_change = max(torch.max(torch.abs(prev_param - param)).item() for prev_param, param in zip(prev_params, model.parameters()))
                if max_param_change < STOP_CRITERIA["change_in_parameters"]["threshold"]:
                    if verbose:
                        print(f"Stopping: Change in parameters < {STOP_CRITERIA['change_in_parameters']['threshold']} at step {steps}")
                    return loss.item()
                    # break
            
            prev_loss = loss.item()
            prev_params = [param.clone() for param in model.parameters()]
    if verbose:
        print(f"Total optimization steps: {steps}")
    if steps == 4000:
        print('Warning: max steps reached')

    return loss.item()  # Return the final loss value for logging or further analysis
    
def train_worker_lbfgs(model, likelihood, mll, train_x, train_y, warm_start=True):
    model.train()
    likelihood.train()
    
    with gpytorch.settings.max_cholesky_size(2000):  # force Cholesky for N <= 2000
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            max_iter=50,
            lr=0.1,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-4,
            tolerance_change=1e-4,
        )
        
        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
    print(f"\n Final loss restarts: {loss}")
    return loss.item()

def train_with_restarts(model_cls, likelihood_cls, mll_cls, X, F, selected_criterion, num_restarts=10, verbose=True):
    # Note in this function N_inputs is passed through model_cls directly, this ensures compatibility with parallelization scheme
    best_model_state_dict = None
    best_likelihood_state_dict = None
    best_loss = float("inf")

    for restart in range(num_restarts):
        if verbose:
            print(f"\n=== Restart {restart + 1}/{num_restarts} ===")

        # Fresh initialization
        likelihood = likelihood_cls()
        model = model_cls(X, F, likelihood)

        
        # Randomize the initial hyperparameters
        with torch.no_grad():
            # If restart == 0, you can let it use the defaults. 
            # For all other restarts, randomize the parameters.
            if restart > 0:
                for param in model.parameters():
                    # Initialize raw parameters randomly between -2.0 and 2.0
                    param.uniform_(-2.0, 2.0) 
        # ---------------------------------------------------------
        mll = mll_cls(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Train model
        train_worker(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=False)

        # Evaluate final loss
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(X)
            loss = -mll(output, F).item()

        if verbose:
            print(f"Final loss for restart {restart + 1}: {loss}")

        if loss < best_loss:
            best_loss = loss
            best_model_state_dict = model.state_dict()
            best_likelihood_state_dict = likelihood.state_dict()

    # Final model re-instantiation and loading best parameters
    final_likelihood = likelihood_cls()
    final_model = model_cls(X, F, final_likelihood)
    final_model.load_state_dict(best_model_state_dict)
    final_likelihood.load_state_dict(best_likelihood_state_dict)

    if verbose:
        print(f"\nBest loss after {num_restarts} restarts: {best_loss}")

    return final_model, final_likelihood

def train_with_restarts_single_model(model_cls, N_inputs, likelihood_cls, mll_cls, X, F, selected_criterion, num_restarts=10, verbose=True):
    best_model_state_dict = None
    best_likelihood_state_dict = None
    best_loss = float("inf")

    for restart in range(num_restarts):
        if verbose:
            print(f"\n=== Restart {restart + 1}/{num_restarts} ===")

        # Fresh initialization
        likelihood = likelihood_cls()
        model = model_cls(X, F, likelihood, N_inputs=N_inputs)

        mll = mll_cls(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Train model
        train_worker(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=False)

        # Evaluate final loss
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(X)
            loss = -mll(output, F).item()

        if verbose:
            print(f"Final loss for restart {restart + 1}: {loss}")

        if loss < best_loss:
            best_loss = loss
            best_model_state_dict = model.state_dict()
            best_likelihood_state_dict = likelihood.state_dict()

    # Final model re-instantiation and loading best parameters
    final_likelihood = likelihood_cls()
    final_model = model_cls(X, F, final_likelihood)
    final_model.load_state_dict(best_model_state_dict)
    final_likelihood.load_state_dict(best_likelihood_state_dict)

    if verbose:
        print(f"\nBest loss after {num_restarts} restarts: {best_loss}")

    return final_model, final_likelihood



def train_gp_worker_v2(welfare_idx, model_state, likelihood_state,
                       shared_path, learning_rate, KERNEL_CHOICE, verbose=False):
    """Worker that loads shared data from disk instead of receiving via pickle."""


    worker_seed = 9534 + welfare_idx
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

    # Load shared data (read-only, loaded once per worker)
    shared = torch.load(shared_path)
    X_train_base = shared['X_train_base']
    Y_train_base = shared['Y_train_base']

    Y_train = Y_train_base[:, welfare_idx]
    N_inputs = X_train_base.shape[1]

    # Rebuild model (same as before)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if KERNEL_CHOICE == 'MA52':
        model = ExactGPMatern52(X_train_base, Y_train, likelihood, N_inputs=N_inputs)
    elif KERNEL_CHOICE == 'SE':
        model = ExactGPModel(X_train_base, Y_train, likelihood, N_inputs=N_inputs)
    elif KERNEL_CHOICE == 'DK':
        model = DKLGP(X_train_base, Y_train, likelihood, N_inputs=2,
                               feature_extractor=DeepKernel(input_dim=N_inputs, output_dim=2))

    clean_state = {k: v for k, v in model_state.items() if not k.startswith('train_')}
    model.load_state_dict(clean_state, strict=False)
    likelihood.load_state_dict(likelihood_state, strict=False)
    model.set_train_data(inputs=X_train_base, targets=Y_train, strict=False)

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_worker(model, likelihood, optimizer, mll, X_train_base, Y_train,
                   selected_criterion=1, verbose=verbose)  # verbose=False to reduce I/O

    return welfare_idx, {
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'likelihood_state_dict': {k: v.cpu() for k, v in likelihood.state_dict().items()},
        'train_x': model.train_inputs[0].cpu(),
        'train_y': model.train_targets.cpu()
    }