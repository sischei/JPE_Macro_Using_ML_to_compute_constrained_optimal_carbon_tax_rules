import importlib
import Parameters
from Parameters import policy, states, policy_states, definitions, global_default_dtype
import PolicyState
import State
import matplotlib.pyplot as plt
import tensorflow as tf
Definitions = importlib.import_module(Parameters.MODEL_NAME + ".Definitions")

def cycle_hook(state,i):
    policy_state = policy(state)
    for s in states:
        tf.summary.histogram("hist_" + s, getattr(State,s)(state), step=i)
        
    for p in policy_states:
        tf.summary.histogram("hist_" + p, getattr(PolicyState,p)(policy_state), step=i)
    
    for d in definitions:
        tf.summary.histogram("hist_" + d, getattr(Definitions,d)(state, policy_state), step=i)

    return True   
        
    
def post_init():

    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tcomp_x", tf.constant(0.0,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) # time 0
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_x",tf.constant(3.,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) # initial TP
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_reached",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) # TP not reached "boolean"
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "kappa_x",tf.constant(0.35032,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) # initial productivity
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a1_x",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a2_x",tf.constant(0.001880953 ,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a3_x",tf.constant(0.009031237,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a4_x",tf.constant(0.021140737,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a5_x",tf.constant(0.03734695,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a6_x",tf.constant(0.056323968,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a7_x",tf.constant(0.07632247,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a8_x",tf.constant(0.095372155,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a9_x",tf.constant(0.1113821,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a10_x",tf.constant(0.09375164,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a11_x",tf.constant(0.07025542,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a12_x",tf.constant(0.039549492,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) 
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "K_total_x",Definitions.K_total_x(Parameters.starting_state))) # initial total capital
    
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "S_x",tf.constant(0.851,shape=(Parameters.starting_state.shape[0],), dtype=global_default_dtype))) # initial carbon stock
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Temp_x", Definitions.Temp_x(Parameters.starting_state))) # initial temperature
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "e_x",Definitions.e_x(Parameters.starting_state))) # initial output
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Omega_x", Definitions.Omega_x(Parameters.starting_state))) # initial damage multiplier
    