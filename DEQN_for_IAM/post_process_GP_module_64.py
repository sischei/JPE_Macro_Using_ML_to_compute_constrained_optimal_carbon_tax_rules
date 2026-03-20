import importlib
import pandas as pd
import numpy as np
from scipy.stats import uniform
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode, do_random_step
import sys, shutil, os
import time
from scipy.stats import qmc
from torch.linalg import vector_norm, norm
import pickle

import torch
import torch.nn as nn
import gpytorch
from scipy.stats.qmc import LatinHypercube

import sampling_module as sm


# we use double precision for the GP
torch.set_default_dtype(torch.float64)


@tf.function
def get_policies(batch_tensor):
    '''
    calculates the policies for a batch of states per period
    '''
    results = tf.map_fn(lambda x: Parameters.policy(x), batch_tensor)
    return results


def gen_train_data_general_NEW(Pseudostate_samples, param_names, N_generations = 40, N_batches = 1000, verbose = False):
    '''
    MODEL: S_x, kappa, tp and all interactions (10 parameters)
    This function simulates the DEQN solution for N_generations-11 periods and N_parameter_combinations*N_batches batches.

    Input: 
    Pseudostate samples: tensor of shape (N_param_combinations, N_params)
    param_names: list of parameter names (e.g. ['tau_const', 'tau_S', 'tau_kappa', 'tau_tp']), must be in the same order as the samples
    N_generations: number of generations to simulate
    N_batches: number of batches to simulate, the mean per period will be the welfare of generation t.

    Output:
    A numpy array of shape (N_param_combinations, N_welfares+N_pseudostates) containing the welfare values for each generation and the associated pseudostate values.

    Note: pseudostate updates are hardcoded in the function
    '''
    # housekeeping
    index_u12 = Parameters.definitions.index('u12')
    value_function_policies = [f"v{i}_y" for i in range(Parameters.N-1, 0, -1)] # get value functions in right order
    indexes_v = tf.constant([Parameters.policy_states.index(i) for i in value_function_policies])
    index_v1 = indexes_v[-1] # get index of first value function
    N_param_combinations = Pseudostate_samples.shape[0]  # number of samples

    N_params = Pseudostate_samples.shape[1] # number of parameters
    N_simulated_episode_length = N_generations - 11 # adjust for unborn

    # create output arrary
    output = np.empty((0, N_params+40))

    # create tensor pseudstate_samples_tf
    Pseudostate_samples_tf = tf.convert_to_tensor(Pseudostate_samples, dtype=tf.float32)
    
    # loop through each parameter combination. 
    # Note: could do in one batch if memory allows but needs adjustment to ensure welfare is calculated correctly (separately for each tax schedule)
    for i in range(N_param_combinations):
        if verbose:
            print("Simulation ", i)

        
        # create starting state with correct size
        simulation_starting_state = tf.repeat(Parameters.starting_state[0][tf.newaxis, :], repeats=N_batches, axis=0)
        # Apply parameters to the state
        for j, key in enumerate(param_names):
            repeated_val = tf.repeat(Pseudostate_samples_tf[i, j][tf.newaxis], repeats=N_batches, axis=0)
            simulation_starting_state = State.update(simulation_starting_state, key, repeated_val)
        
        # in some cases the tax itself is part of the statespace
        if "tau_x" in Parameters.states:
            current_state = State.update(current_state, "tau_x", Definitions.tau_x(current_state,None))

        
        if verbose:
            tf.print("simulation_starting_state: ", simulation_starting_state)
        # create state episode    
        state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis=0), [N_simulated_episode_length, 1, 1])

        # run simulation
        state_episode = run_episode(state_episode)
        

        policies = get_policies(state_episode)

        # policies_scaled = tf.convert_to_tensor(policy_state_episode_batch_scaled, dtype=tf.float32)
        policies_reshaped = tf.reshape(policies, [N_simulated_episode_length * N_batches, len(Parameters.policy_states)])   # Shape: (29 * 10000, 22)
        # Apply the policy functions element-wise across policies
        scaled_policies = tf.stack([getattr(PolicyState, ps)(policies_reshaped) for ps in Parameters.policy_states], axis=1)  # Shape: (29 * 10000, 22)

        # Reshape back to the original shape
        policies_scaled = tf.reshape(scaled_policies, [N_simulated_episode_length, N_batches, len(Parameters.policy_states)])
        
      
        # extract welfare values
        welfare_oldest = Definitions.u12(state_episode[0], policies[0])[:, tf.newaxis]
        welfare_younger = tf.gather(policies_scaled[0],  indexes_v, axis=1)
        welfares_first_period = tf.concat([welfare_oldest, welfare_younger], axis=1)
        welfares_next_periods = tf.transpose(policies_scaled[1:,:,index_v1]) # only v1 is needed
        all_welfares = tf.concat([welfares_first_period, welfares_next_periods], axis=1)
        welfares = tf.reduce_mean(all_welfares, axis=0) # calculate mean welfare per period
        
        # create output row
        params = Pseudostate_samples[i,:]
        welf_row = np.hstack([params, welfares.numpy()])
        print('params: ', params)
        output = np.vstack([output, welf_row])                  


    return output
# ------------------------------------------------
# Define model classes
# ------------------------------------------------

class MultitaskSE(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs, output_size):
        super(MultitaskSE, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(dtype=torch.float64), num_tasks=output_size
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=N_inputs, dtype=torch.float64), num_tasks=output_size, rank=1
        )

    def forward(self, x):
        x = x.to(torch.float64)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MultitaskMa32(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_inputs, output_size):
        super(MultitaskMa32, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=output_size
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=N_inputs), num_tasks=output_size, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

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



# Utility function for Bayesian active learning
def bal_utility(e_gp, var_gp, mean_y,std_y, rho=0.5, beta=0.5):
    """Calculate utility for Bayesian active learning."""
    utility = rho * e_gp + (beta / 2.0) * torch.log(var_gp)

    return utility

# Train the GP model, includes stopping criteria
def train(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=True):
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

        
        # randomize initialization
        with torch.no_grad():
            # If restart == 0, use the defaults. 
            # For all other restarts, randomize the parameters.
            if restart > 0:
                for param in model.parameters():
                    # Initialize raw parameters randomly between -2.0 and 2.0
                    param.uniform_(-2.0, 2.0) 
        # ---------------------------------------------------------
        mll = mll_cls(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Train model
        train(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=False)

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
        train(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=False)

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

def train_lbfgs(model, likelihood, optimizer, mll, X, F, selected_criterion=1, verbose=True):
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
    # define closure
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, F)
        loss.backward()
        return loss

    # loss = closure()
    # loss.backward()

    training_iter = 1000
    for i in range(training_iter):

        # perform step and update curvature
        # options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss = optimizer.step(closure)

       
        if STOP_CRITERIA["gradient_norm"]["enabled"]:
            grad_norm = vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            if grad_norm < STOP_CRITERIA["gradient_norm"]["threshold"]:
                if verbose:
                    print(f"Stopping: Gradient norm < {STOP_CRITERIA['gradient_norm']['threshold']} at step {steps}")
                break
        
        # optimizer.step()
        
        steps += 1
        
        if STOP_CRITERIA["change_in_loss"]["enabled"] and prev_loss is not None:
            if abs(prev_loss - loss.item()) < STOP_CRITERIA["change_in_loss"]["threshold"]:
                if verbose:
                    print(f"Stopping: Change in loss < {STOP_CRITERIA['change_in_loss']['threshold']} at step {steps}")
                break
        
        if STOP_CRITERIA["change_in_parameters"]["enabled"]:
            max_param_change = max(torch.max(torch.abs(prev_param - param)).item() for prev_param, param in zip(prev_params, model.parameters()))
            if max_param_change < STOP_CRITERIA["change_in_parameters"]["threshold"]:
                if verbose:
                    print(f"Stopping: Change in parameters < {STOP_CRITERIA['change_in_parameters']['threshold']} at step {steps}")
                break
        
        prev_loss = loss.item()
        prev_params = [param.clone() for param in model.parameters()]
    if verbose:
        print(f"Total optimization steps: {steps}")

# Train the GP model, includes stopping criteria
def train_batch(model, likelihood, optimizer, mll, X, F, selected_criterion, verbose=True):
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

    for i in range(4000):
        optimizer.zero_grad()
        output = model(X)
        # print('F: ', F)
        loss = -mll(output, F).sum()
        loss.backward()
        
        if STOP_CRITERIA["gradient_norm"]["enabled"]:
            grad_norm = vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            if grad_norm < STOP_CRITERIA["gradient_norm"]["threshold"]:
                if verbose:
                    print(f"Stopping: Gradient norm < {STOP_CRITERIA['gradient_norm']['threshold']} at step {steps}")
                break
        
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
                break
        
        if STOP_CRITERIA["change_in_parameters"]["enabled"]:
            max_param_change = max(torch.max(torch.abs(prev_param - param)).item() for prev_param, param in zip(prev_params, model.parameters()))
            if max_param_change < STOP_CRITERIA["change_in_parameters"]["threshold"]:
                if verbose:
                    print(f"Stopping: Change in parameters < {STOP_CRITERIA['change_in_parameters']['threshold']} at step {steps}")
                break
        
        prev_loss = loss.item()
        prev_params = [param.clone() for param in model.parameters()]
    if verbose:
        print(f"Total optimization steps: {steps}")
    if steps == 4000:
        print('Warning: max steps reached')



# active learning iteration 
def active_learning_step(model, likelihood, X, F, candidates, data_generating_function, welfares_BAU, welfare_weights,  X_mean, X_std, F_mean, F_std, param_names = None, num_new_points=1, n_candidates=1000, N_generations=40):
    # Not used, anymore
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Create a large pool of random points to choose from and take points with highest variance
        candidates_np = candidates
        pool_X = torch.tensor(candidates_np, dtype=torch.float64)
        # normalize X here
        pool_X = (pool_X - X_mean) / X_std
        preds = model(pool_X)
        variances = preds.variance
        U_bal = bal_utility(preds.mean, variances, F_mean,F_std,rho=1, beta=100., threshold=-0.02)
        # U_bal = torch.max(U_bal)
        new_mean_Y, indices = torch.topk(U_bal, num_new_points, dim=0)
        indices = torch.unique(indices)
        new_X = pool_X[indices]
        # unnormalize X here
        new_X_tf = new_X * X_std + X_mean
        print('selected point: ', new_X_tf)
        # simulate the new points
        new_X_tf = tf.constant(new_X_tf.numpy(), dtype=tf.float32)
        # new_data = data_generating_function(new_X_tf, welfares_BAU, N_generations=N_generations, N_batches=10000) # N_batches is hardcoded because we always want to simulate 10000 batches
        if param_names is None:
            new_data = data_generating_function(new_X_tf, welfares_BAU, N_generations=N_generations, N_batches=10000)
        else:
            new_data = data_generating_function(new_X_tf, param_names = param_names, welfares_BAU = welfares_BAU, N_generations=N_generations, N_batches=10000)
        # calculate the new weighted welfare
        new_Y = torch.tensor(np.average(new_data[:,candidates_np.shape[1]:], axis=1, weights=welfare_weights), dtype=torch.float64)
        print("New welfare: ", new_Y)
        # normalize Y here
        new_Y_train = (new_Y - F_mean)/ F_std
        X = torch.cat([X, new_X], dim=0)
        F = torch.cat([F, new_Y_train], dim=0)

        return X, F
    

# active learning iteration
def active_learning_step_NEW(model, likelihood, X, F, candidates, welfare_weights,  X_mean, X_std, F_mean, F_std, param_names = None, data_generating_function=None, num_new_points=1, N_generations=40):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Create a large pool of random points to choose from and take points with highest variance
        candidates_np = candidates
        pool_X = torch.tensor(candidates_np, dtype=torch.float64)
        # normalize X here
        pool_X = (pool_X - X_mean) / X_std
        preds = model(pool_X)
        variances = preds.variance
        U_bal = bal_utility(preds.mean, variances, F_mean,F_std,rho=1, beta=100.)# threshold=-0.02
        # U_bal = torch.max(U_bal)
        new_mean_Y, indices = torch.topk(U_bal, num_new_points, dim=0)
        indices = torch.unique(indices)
        new_X = pool_X[indices]
        # unnormalize X here
        new_X_tf = new_X * X_std + X_mean
        print('selected point: ', new_X_tf)
        # simulate the new points
        new_X_tf = tf.constant(new_X_tf.numpy(), dtype=tf.float32)
        # new_data = data_generating_function(new_X_tf, welfares_BAU, N_generations=N_generations, N_batches=10000) # N_batches is hardcoded because we always want to simulate 10000 batches
        if param_names is None:
            new_data = data_generating_function(new_X_tf, N_generations=N_generations, N_batches=10000)
        else:
            new_data = gen_train_data_general_NEW(new_X_tf, param_names = param_names, N_generations=N_generations, N_batches=10000)
        # calculate the new weighted welfare
        new_Y = torch.tensor(np.average(new_data[:,candidates_np.shape[1]:], axis=1, weights=welfare_weights), dtype=torch.float64)
        print("New welfare: ", new_Y)
        # normalize Y here
        new_Y_train = (new_Y - F_mean)/ F_std
        X = torch.cat([X, new_X], dim=0)
        F = torch.cat([F, new_Y_train], dim=0)

        return X, F
    


def compute_loo_error(model, learning_rate, criterion = 1, optim=None):
    # when calculating the LOO we want the same model type (kerneletc)
    modeltype = type(model)
    print('modeltype: ', modeltype)

    # extract data
    X = model.train_inputs[0]
    y = model.train_targets

    loo_error = torch.empty(0)
    for i in range(X.shape[0]):
        
        # data wrangling
        X_new = torch.cat((X[:i], X[i+1:]), dim=0)
        X_test = torch.unsqueeze(X[i], 0)
        y_new = torch.cat((y[:i], y[i+1:]), dim=0)
        y_test = y[i]


        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if "DKLGP" in str(modeltype):
            feature_extractor = DeepKernel(input_dim=X_new.shape[1])
            mod = modeltype(X_new, y_new, likelihood, feature_extractor, N_inputs=X_new.shape[1])
        else:
            # feature_extractor = DeepKernel(input_dim=X_new.shape[1])
            # mod = modeltype(X_new, y_new, likelihood, feature_extractor, N_inputs=X_new.shape[1])
            mod = modeltype(X_new, y_new, likelihood, N_inputs=X_new.shape[1])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, mod)
        if optim=='lbfgs':
            mod.train()
            likelihood.train()
            optimizer = torch.optim.LBFGS(mod.parameters(), lr=learning_rate, max_iter=100)
            train_lbfgs(mod, likelihood, optimizer, mll, X_new, y_new, selected_criterion=criterion, verbose=False)
        else:
            optimizer = torch.optim.Adam(mod.parameters(), lr=learning_rate)

            mod.train()
            likelihood.train()
            train(mod, likelihood, optimizer, mll, X_new, y_new, selected_criterion=criterion, verbose=True)
        
        mod.eval()
        likelihood.eval()
        with torch.no_grad():
            pred = mod(X_test)
            loss = (pred.mean - y_test) ** 2
            loo_error = torch.cat((loo_error, loss), dim=0)
        print('LOO training progress (%): ', i/X.shape[0]*100)
    return torch.mean(loo_error), loo_error


# ------------------------------------------------
# Optimization functions
# ------------------------------------------------
def evaluate_model(x,model):
    # Evaluate the  model
    model.eval()
    # Convert to tensor
    x = torch.tensor(x, dtype=torch.float64).reshape(1, -1)
    # print("x: ", x)
    # Predict
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(x)
        mean = predictions.mean.numpy()
    return mean

# objective function
def objective(x,model):
    # Evaluate the model
    y = evaluate_model(x,model) 
    # return negative as we minimize!
    return -y
    

