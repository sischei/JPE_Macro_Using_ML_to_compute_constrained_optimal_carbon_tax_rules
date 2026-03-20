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
#from torch.linalg import vector_norm, norm


@tf.function
def get_policies(batch_tensor):
    '''
    calculates the policies for a batch of states per period
    '''
    results = tf.map_fn(lambda x: Parameters.policy(x), batch_tensor)
    return results



# ------------------------------------------------
# model with 1 state and constant
def gen_train_data_const_S(Pseudostate_samples, welfares_BAU, N_generations = 40, N_batches = 1000, verbose = False):
    '''
    MODEL: S_x, kappa, tp and all interactions (10 parameters)
    This function simulates the DEQN solution for N_generations-11 periods and N_parameter_combinations*N_batches batches.

    Input: 
    Pseudostate samples: tensor of shape (N_param_combinations, 3)
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

        
        # extract pseudostates, CAREFUL TO GET THE RIGHT ORDER
        tau_const = tf.repeat(Pseudostate_samples_tf[i,0], N_batches)
        tau_S = tf.repeat(Pseudostate_samples_tf[i,1], N_batches)

        if verbose:
            tf.print("tau_const: ", tau_const)
            tf.print("tau_S: ", tau_S)

        # update starting state
        simulation_starting_state = State.update(simulation_starting_state,'tau_const_x', tau_const)
        simulation_starting_state = State.update(simulation_starting_state,'tau_S_x', tau_S)

        if verbose:
            tf.print("simulation_starting_state: ", simulation_starting_state)
        # create state episode    
        state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis=0), [N_simulated_episode_length, 1, 1])

        # run simulation
        state_episode = run_episode(state_episode)
        

        policies = get_policies(state_episode)

  

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
# model with 1 state and constant including gradients
# ------------------------------------------------

# helper function
# get expected lifetime utility
def compute_mean_v1_y(policies_t):
    # policies_t: shape (B, P)
    v1_values = PolicyState.v1_y(policies_t)  # shape (B,)
    return tf.reduce_mean(v1_values)  # scalar


# Note: Not USED. This was used in experiments for targeted sampling (taking into account BAU welfares to guide sampling w.r.t. welfare gains)
def gen_train_data_general(Pseudostate_samples, param_names,welfares_BAU, N_generations = 40, N_batches = 1000, verbose = False):
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
            simulation_starting_state = State.update(simulation_starting_state, "tau_x", Definitions.tau_x(simulation_starting_state,None))

        
        if verbose:
            tf.print("simulation_starting_state: ", simulation_starting_state)
        # create state episode    
        state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis=0), [N_simulated_episode_length, 1, 1])

        # run simulation
        state_episode = run_episode(state_episode)
        

        policies = get_policies(state_episode)

        policies_reshaped = tf.reshape(policies, [N_simulated_episode_length * N_batches, len(Parameters.policy_states)])  
        # Apply the policy functions element-wise across policies
        scaled_policies = tf.stack([getattr(PolicyState, ps)(policies_reshaped) for ps in Parameters.policy_states], axis=1) 

        # Reshape back to the original shape
        policies_scaled = tf.reshape(scaled_policies, [N_simulated_episode_length, N_batches, len(Parameters.policy_states)])
        
      
        # extract welfare values
        welfare_oldest = Definitions.u12(state_episode[0], policies[0])[:, tf.newaxis]
        welfare_younger = tf.gather(policies_scaled[0],  indexes_v, axis=1)
        welfares_first_period = tf.concat([welfare_oldest, welfare_younger], axis=1)
        welfares_next_periods = tf.transpose(policies_scaled[1:,:,index_v1]) # only v1 is needed
        all_welfares = tf.concat([welfares_first_period, welfares_next_periods], axis=1)
        welfares = tf.reduce_mean(all_welfares, axis=0) # calculate mean welfare per period

        welfares_CE = (welfares / welfares_BAU) ** (1 / (1-Parameters.sigma)) - 1
        
        # create output row
        params = Pseudostate_samples[i,:]
        welf_row = np.hstack([params, welfares_CE.numpy()])
        print('params: ', params)
        output = np.vstack([output, welf_row])                  


    return output



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
            simulation_starting_state = State.update(simulation_starting_state, "tau_x", Definitions.tau_x(simulation_starting_state,None))

        
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
        print("Welfares shape: ", welfares.shape)
        
        # create output row
        params = Pseudostate_samples[i,:]
        welf_row = np.hstack([params, welfares.numpy()])
        print('params: ', params)
        output = np.vstack([output, welf_row])                  


    return output