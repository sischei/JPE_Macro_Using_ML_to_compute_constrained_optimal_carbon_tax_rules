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
    
    for d in definitions:
        tf.summary.histogram("hist_" + d, getattr(Definitions,d)(state, policy_state), step=i)

    return True   

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


# Set up RNG with a seed for reproducibility
seed = 8917
rng = np.random.default_rng(seed)


# transfer shares
shares = np.array([0.087, 0.082, 0.08, 0.04,
                   0.18, 0.12, 0.04, 0.117, 0.076,
                   0.043, 0.044, 0.071])
shares /= shares.sum()
concentration_scale = 15  # adjust this as needed
concentration = shares * concentration_scale

def sample_uniform_perturbed(num_samples=1000, radius=0.03, upper_bound=0.3):
    """
    Generate samples of approximately uniform distributions with a strict upper bound constraint
    using rejection sampling.
    
    Args:
        num_samples: Number of samples to generate
        radius: Maximum perturbation from uniform
        upper_bound: Maximum allowed value for any component (strictly enforced)
        
    Returns:
        Array of samples where each sample is a probability distribution
    """
    base = np.full(12, 1/12)  # Uniform distribution
    samples = []
    
    while len(samples) < num_samples:
        # Generate more samples at once for efficiency
        batch_size = min(num_samples * 2, 1000)  # Generate extra samples for rejection
        batch_samples = []
        
        for _ in range(batch_size):
            # Generate noise
            noise = np.random.uniform(-radius, radius, size=12)
            perturbed = base + noise
            
            # Ensure non-negativity
            perturbed = np.clip(perturbed, 0.005, None)
            
            # Normalize to sum to 1
            perturbed /= perturbed.sum()
            
            batch_samples.append(perturbed)
        
        # Convert to numpy array for vectorized operations
        batch_samples = np.array(batch_samples)
        
        # Find samples that meet our criteria (all values <= upper_bound)
        valid_mask = np.all(batch_samples <= upper_bound, axis=1)
        valid_samples = batch_samples[valid_mask]
        
        # Add as many valid samples as we need
        samples_needed = num_samples - len(samples)
        valid_to_add = valid_samples[:samples_needed]
        samples.extend(valid_to_add)
    
    return np.array(samples)

def post_init():

    current_episode = tf.cast(Parameters.ckpt.current_episode, tf.int32) 

    # Tax range expansion parameters
    initial_tax_max = 0.5
    final_tax_max = Parameters.tax_range[1]  # Assuming this is the maximum tax range
    
    # for transfers:
    init_concentration = 50
    final_concentration = Parameters.alpha_dirichlet
    start_episode = 100  # Start expanding at episode 50
    expansion_duration = 200  # Take 100 episodes to reach full range
    initial_radius = 0.04
    final_radius = 0.08
    
    # Calculate progress (0 to 1) through the expansion period
    if current_episode < start_episode:
        # Before expansion starts, use initial range
        current_tax_max = initial_tax_max
        # Before expansion starts, use initial range
        current_radius = initial_radius
    else:
        # Linear progression through expansion period
        episodes_into_expansion = current_episode - start_episode
        progress = min(float(episodes_into_expansion) / expansion_duration, 1.0)
        current_tax_max = initial_tax_max + (final_tax_max - initial_tax_max) * progress
        current_radius = initial_radius + (final_radius - initial_radius) * progress
       
    # Create adjusted tax range
    adjusted_tax_range = (0, current_tax_max)

    num_samples = Parameters.starting_state.shape[0]
    params = sm.sample_params_lin3(num_samples=num_samples, x0=S0_norm, y0=kappa0_norm, z0=tp0_norm,xT_range=ST_range_norm,yT_range=kappaT_norm, zT_range=tpT_range_norm, param_bounds=param_bounds, tax_range=adjusted_tax_range, rng=rng, batch_size=1000)
    
    

    # sample transfer shares
    transfer_shares = rng.dirichlet(concentration, num_samples).astype(np.float32)

    # Alternative sampling, helps with stability in early training
    # transfer_shares = sample_uniform_perturbed(num_samples= num_samples, radius=final_radius, upper_bound=0.2)

    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tcomp_x", tf.constant(0.0,shape=(Parameters.starting_state.shape[0],))) ) # initial computational time
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_x",tf.constant(3.,shape=(Parameters.starting_state.shape[0],)))) # initial TP
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_reached",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) # TP not reached "boolean"
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "kappa_x",tf.constant(.35032,shape=(Parameters.starting_state.shape[0],)))) # initial emissions
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a1_x",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) # initial capital
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
    
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "S_x",tf.constant(0.851,shape=(Parameters.starting_state.shape[0],)))) # initial carbon stock
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Temp_x", Definitions.Temp_x(Parameters.starting_state))) # initial temperature
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Omega_x", Definitions.Omega_x(Parameters.starting_state))) # initial damage multiplier
    # assign pseudo states
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_const_x", tf.constant(params[:,0], dtype=tf.float32)) ) # initial computational time
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_S_x", tf.constant(params[:,1], dtype=tf.float32)) )
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_kappa_x", tf.constant(params[:,2], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_TP_x", tf.constant(params[:,3], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tau_x", Definitions.tau_x(Parameters.starting_state, None)))

    # transfers
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare1_x", tf.constant(transfer_shares[:,0], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare2_x", tf.constant(transfer_shares[:,1], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare3_x", tf.constant(transfer_shares[:,2], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare4_x", tf.constant(transfer_shares[:,3], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare5_x", tf.constant(transfer_shares[:,4], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare6_x", tf.constant(transfer_shares[:,5], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare7_x", tf.constant(transfer_shares[:,6], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare8_x", tf.constant(transfer_shares[:,7], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare9_x", tf.constant(transfer_shares[:,8], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare10_x", tf.constant(transfer_shares[:,9], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare11_x", tf.constant(transfer_shares[:,10], dtype=tf.float32)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tshare12_x", tf.constant(transfer_shares[:,11], dtype=tf.float32)))
        


