import importlib
import Parameters
from Parameters import policy, states, policy_states, definitions
import PolicyState
import State
import matplotlib.pyplot as plt
import tensorflow as tf
import sampling_module as sm
import numpy as np
Definitions = importlib.import_module(Parameters.MODEL_NAME + ".Definitions")

def cycle_hook(state,i):
    policy_state = policy(state)
    for s in states:
        tf.summary.histogram("hist_" + s, getattr(State,s)(state), step=i)
        
    for p in policy_states:
        tf.summary.histogram("hist_" + p, getattr(PolicyState,p)(policy_state), step=i)
    

    return True   

# Constants for the problem
S0 = Parameters.S0
ST_range = Parameters.ST_range
param_bounds = Parameters.param_bounds
tax_range = Parameters.tax_range

# Set up RNG
seed = 8917
rng = np.random.default_rng(seed)

       
def post_init():
    num_samples = Parameters.starting_state.shape[0]
    params = sm.sample_params_lin1(num_samples=num_samples, x0=S0, xT_range=ST_range, param_bounds=param_bounds, tax_range=tax_range, rng=rng, batch_size=1000)

    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tcomp_x", tf.constant(0.0,shape=(Parameters.starting_state.shape[0],))) ) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_x",tf.constant(3.,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_reached",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "kappa_x",tf.constant(.35032,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a1_x",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a2_x",tf.constant(0.001880953 ,shape=(Parameters.starting_state.shape[0],))))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a3_x",tf.constant(0.009031237,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a4_x",tf.constant(0.021140737,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a5_x",tf.constant(0.03734695,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a6_x",tf.constant(0.056323968,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a7_x",tf.constant(0.07632247,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a8_x",tf.constant(0.095372155,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a9_x",tf.constant(0.1113821,shape=(Parameters.starting_state.shape[0],))))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a10_x",tf.constant(0.09375164,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a11_x",tf.constant(0.07025542,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a12_x",tf.constant(0.039549492,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "K_total_x",Definitions.K_total_x(Parameters.starting_state))) # initial total capital
    
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "S_x",tf.constant(0.851,shape=(Parameters.starting_state.shape[0],)))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Temp_x", Definitions.Temp_x(Parameters.starting_state))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Omega_x", Definitions.Omega_x(Parameters.starting_state))) 
    # assign pseudo states
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_const_x", tf.constant(params[:,0], dtype=tf.float32)) ) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_S_x", tf.constant(params[:,1], dtype=tf.float32)) )
