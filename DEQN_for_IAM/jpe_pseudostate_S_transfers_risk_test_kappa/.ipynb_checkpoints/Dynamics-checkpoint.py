import itertools
import tensorflow as tf
import State
import PolicyState
import Definitions
import Globals
import Parameters
import os

# shocks
# kappa ar1: kappa_t = 0.05 + 0.95 * kappa_{t-1} + epsilon_t (sigma_epsilon = 0.03)
Nshocks = 9
shocks_kappa = [-0.034, 0., 0.034] # baseline: shocks_kappa = [-0.03, 0., 0.03]

shocks_TP = [-0.1, 0., 0.1]

probs_TP = [0.333, 0.334, 0.333]
probs_kappa = [0.333, 0.334, 0.333]

shock_values = tf.constant(list(itertools.product(shocks_kappa, shocks_TP)))
shock_probs = tf.constant([ p_k * p_tp  for p_k, p_tp in list(itertools.product(probs_kappa, probs_TP))])


# -------------
# post processing sampling for gradients
# -------------
rng_pp = tf.random.Generator.from_seed(1234)

def sample_categorical_from_generator(rng, probs, batch_size):
    """Sample categorical indices using a tf.random.Generator."""
    # Compute cumulative distribution function (CDF)
    cdf = tf.cumsum(tf.convert_to_tensor(probs, dtype=tf.float32))
    
    # Sample uniform random numbers in [0, 1)
    u = rng.uniform([batch_size], dtype=tf.float32)
    
    # Determine which bin each sample falls into
    sample_indices = tf.reduce_sum(tf.cast(u[:, tf.newaxis] > cdf[tf.newaxis, :], tf.int32), axis=1)
    return sample_indices
# -------------------------------------------------- #

def total_step_random(prev_state, policy_state):
    ar = AR_step(prev_state)
    if not Globals.DISABLE_SCHOCKS:
        shock = shock_step_random(prev_state)
    else:
        shock = tf.zeros_like(prev_state)
        
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy     
    total = augment_state(total)
    return (total)

# same as above, but non randomized shock, but rather the same shock for each realization
def total_step_spec_shock(prev_state, policy_state, shock_index):
    # update shock probabilities
    ar = AR_step(prev_state)
    shock = shock_step_spec_shock(prev_state, shock_index)
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy        
    return augment_state(total)

# updates states that depend on previous policy and current shocks
def augment_state(state):
    
    # ensure kappa = 0 if kappa < 0.005
    kappa_updates = tf.where(State.kappa_x(state) < 1e-4, 1e-9, State.kappa_x(state))
    state = State.update(state, "kappa_x", kappa_updates)

    # ensure that TP does not go below 2.5 or above 3.5
    # TP_updates = tf.where(State.TP_x(state) < 2.5, 2.5, State.TP_x(state))
    # TP_updates = tf.where(TP_updates > 3.5, 3.5, TP_updates)
    TP_updates = tf.clip_by_value(State.TP_x(state), clip_value_min=2.5, clip_value_max=3.5)
    state = State.update(state, "TP_x", TP_updates)

    # update freeze variable (needs to be here and in shock_step) this is inefficient
    tp_reached_values = tf.where(State.Temp_x(state)>=State.TP_x(state),1.0,State.TP_reached(state))
    state = State.update(state, "TP_reached",tp_reached_values)
    
    state = State.update(state, "Omega_x", Definitions.Omega_x(state, None))

    # state = State.update(state, "theta1", Definitions.theta1(state, None))
    # state = State.update(state, "mu_x", Definitions.mu_x(state, None))
    # state = State.update(state, "tau_x", Definitions.tau_x(state, None))
    # update firm variables
    # state = State.update(state, "mu_cost", Definitions.mu_cost(state, None))
    # state = State.update(state, "e_x", Definitions.e_x(state, None))
    # state = State.update(state, "ynet_x", Definitions.ynet_x(state, None))
    # state = State.update(state, "r_x", Definitions.r_x(state, None))
    # state = State.update(state, "w_x", Definitions.w_x(state, None))


    return state


def AR_step(prev_state):
    ar_step = tf.zeros_like(prev_state)
    ar_step = State.update(ar_step, "kappa_x", Definitions.rho_t(prev_state,None) * State.kappa_x(prev_state))
    ar_step = State.update(ar_step, "TP_x", 1.0 * State.TP_x(prev_state))
    return ar_step


def shock_step_random(prev_state):

    shock_step = tf.zeros_like(prev_state)

    sample_index = tf.random.categorical(tf.math.log([shock_probs]), prev_state.shape[0])[0,:]
    
    # for post_processing generating gradients
    # sample_index = sample_categorical_from_generator(rng_pp, shock_probs, prev_state.shape[0])
    
    # update z 
    z_values = tf.cast(sample_index,tf.float32)
    # shock values
    shock_vals = tf.reshape(tf.gather(shock_values,sample_index),[prev_state.shape[0],shock_values.shape[1]])

    # kappa values
    kappa_values = shock_vals[:,0]
    # kappa shock should be 0 if kappa is below 0.001
    kappa_values = tf.where(State.kappa_x(prev_state) < 1e-4, 0.0, kappa_values)
 
    # set TP shock to 0 whenever Temp_x is above TP_x (tipping point reached)

    TP_values = shock_vals[:,1] * (State.TP_reached(prev_state) -1.) * (-1.)

    # shock values
    # shock_step = State.update(shock_step, "z_x", z_values)
    shock_step = State.update(shock_step, "kappa_x", kappa_values)
    shock_step = State.update(shock_step, "TP_x", TP_values)
    
    return shock_step



def shock_step_spec_shock(prev_state, shock_index):
    # Use a specific shock - for calculating expectations 
    shock_step = tf.zeros_like(prev_state)

    # update z 
    # shock_step = State.update(shock_step, "z_x", tf.constant(shock_index,shape=(prev_state.shape[0],),dtype=tf.float32))

    # update kappa
    kappa_values = tf.repeat(shock_values[shock_index,0], prev_state.shape[0])
    kappa_values = tf.where(State.kappa_x(prev_state) < 1e-4, 0.0, kappa_values)
    shock_step = State.update(shock_step,"kappa_x", kappa_values)

    # check whether TP is frozen (tipping point reached)
    TP_values = tf.repeat(shock_values[shock_index,1], prev_state.shape[0]) * (State.TP_reached(prev_state) -1.) * (-1.)

    shock_step = State.update(shock_step,"TP_x", TP_values)

    return shock_step


def policy_step(prev_state, policy_state):
    policy_step = tf.zeros_like(prev_state)
    for i in range(1, Parameters.N):
        policy_step = State.update(policy_step, "a" + str(i+1) + "_x", getattr(PolicyState, "anext" + str(i) + "_y")(policy_state))

    # update capital
    policy_step = State.update(policy_step, "K_total_x", Definitions.K_total_next(prev_state, policy_state))


    
    policy_step = State.update(policy_step, "S_x", Definitions.S_next(prev_state,policy_state)) # update stock of carbon
    policy_step = State.update(policy_step, "Temp_x", Definitions.Temp_x(policy_step, None)) # update temperature after stock of carbon
    policy_step = State.update(policy_step, "tcomp_x", Definitions.tcomp2tcompplus(prev_state, policy_state)) # save for augment_state
    policy_step = State.update(policy_step, "TP_reached", State.TP_reached(prev_state)) # move TP_reached to next state, update in augment_state

    # update pseudo state variables
    policy_step = State.update(policy_step, "tau_const_x", State.tau_const_x(prev_state))
    policy_step = State.update(policy_step, "tau_S_x", State.tau_S_x(prev_state))

    # update transfers
    for i in range(1, Parameters.N+1):
        policy_step = State.update(policy_step, "tshare" + str(i) + "_x", getattr(State, "tshare" + str(i) + "_x")(prev_state))
    

    
    
    return policy_step
