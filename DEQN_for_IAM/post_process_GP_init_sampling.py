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
import post_process_GP_module_tensorflow as GPtf # tf functions to generate training data


tf.get_logger().setLevel('CRITICAL')


Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")


# --------------------------------------------------------------------------- #
# MANUAL INPUT
# --------------------------------------------------------------------------- #

# BAU welfares are not used, just here for backwards compatibility
path_welfare_BAU = 'runs/jpe_bau_final/final/sim_mean_welfare.csv'

initial_samples = 2000

# rng for sampling
seed = 761


# set numpy seed
np.random.seed(seed)

# set tensorflow seed
tf.random.set_seed(seed)

rng = np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# house keeping and setup
# --------------------------------------------------------------------------- #

if Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_scratch']:
    S0 = Parameters.S0
    ST_range = Parameters.ST_range
    param_bounds = Parameters.param_bounds
    tax_range = Parameters.tax_range




# linear 
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

elif Parameters.MODEL_NAME in['jpe_pseudostate_S_transfers_risk_test', 'jpe_pseudostate_S_transfers_risk_test_kappa']:
    # Constants for the problem
    S0 = Parameters.S0
    ST_range = Parameters.ST_range
    param_bounds = Parameters.param_bounds
    tax_range = Parameters.tax_range
    
else:
    raise ValueError("MODEL_NAME not implemented")

# load welfare values from BAU scenario
df_BAU = pd.read_csv(path_welfare_BAU)
welfares_BAU = np.squeeze(df_BAU.values[:40])

# --------------------------------------------------------------------------- #
# Sampling raw training data 
# --------------------------------------------------------------------------- #


# RUN sampling
if Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_scratch']:
    params = sm.sample_params_lin1(num_samples=initial_samples, x0=S0, xT_range=ST_range, param_bounds=param_bounds, tax_range=tax_range, rng=rng, batch_size=1000)
    training_set = GPtf.gen_train_data_const_S(Pseudostate_samples=params, welfares_BAU=welfares_BAU, N_generations=40, N_batches=10000, verbose=False)


elif Parameters.MODEL_NAME in ['jpe_pseudostate_S_transfers_risk_test', 'jpe_pseudostate_S_transfers_risk_test_kappa']:
    # sample tax parameters
    params = sm.sample_params_lin1(num_samples=initial_samples, x0=S0, xT_range=ST_range, param_bounds=param_bounds, tax_range=tax_range, rng=rng, batch_size=1000)
  
    # reinforcement_2 (basis for pareto improvement)
    shares = np.array([0.087, 0.082, 0.08, 0.04,
                   0.18, 0.12, 0.04, 0.117, 0.076,
                   0.043, 0.044, 0.071])
    shares /= shares.sum()
    concentration_scale = 15  # adjust this as needed
    concentration = shares * concentration_scale


    transfers = rng.dirichlet(concentration, initial_samples).astype(np.float32)
    # combine parameters and transfers
    params = np.hstack([params, transfers])
    # generate name list
    param_names = ['tau_const_x', 'tau_S_x', 'tshare1_x', 'tshare2_x', 'tshare3_x', 'tshare4_x', 'tshare5_x', 'tshare6_x', 'tshare7_x', 'tshare8_x', 'tshare9_x', 'tshare10_x', 'tshare11_x', 'tshare12_x']
    training_set = GPtf.gen_train_data_general_NEW(Pseudostate_samples=params, param_names=param_names, N_generations=40, N_batches=10000, verbose=False)

elif Parameters.MODEL_NAME in ['jpe_pseudostate_linear_transfers_risk_test_implied']:

    # Sample parameters
    params = sm.sample_params_lin3(num_samples=initial_samples, x0=S0_norm, y0=kappa0_norm, z0=tp0_norm,xT_range=ST_range_norm,yT_range=kappaT_norm, zT_range=tpT_range_norm, param_bounds=param_bounds, tax_range=tax_range, rng=rng, batch_size=1000)
    


    shares = np.array([0.087, 0.082, 0.08, 0.04,
                   0.18, 0.12, 0.04, 0.117, 0.076,
                   0.043, 0.044, 0.071])
    shares /= shares.sum()
    concentration_scale = 15  # adjust this as needed
    concentration = shares * concentration_scale
    transfers = rng.dirichlet(concentration, initial_samples).astype(np.float32)

    # combine parameters and transfers
    params = np.hstack([params, transfers])
    print("params shape:", params.shape)
    # generate name list
    param_names = ['tau_const_x', 'tau_S_x', 'tau_kappa_x', 'tau_TP_x', 'tshare1_x', 'tshare2_x', 'tshare3_x', 'tshare4_x', 'tshare5_x', 'tshare6_x', 'tshare7_x', 'tshare8_x', 'tshare9_x', 'tshare10_x', 'tshare11_x', 'tshare12_x']
    training_set = GPtf.gen_train_data_general_NEW(Pseudostate_samples=params, param_names=param_names, N_generations=40, N_batches=10000, verbose=False)



# save training data to csv
df = pd.DataFrame(training_set)
df.to_csv(Parameters.LOG_DIR + "/GP_train_data_raw.csv", index=False)

