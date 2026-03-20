import numpy as np
from scipy.stats import uniform

'''
This file is a collection of functions that are used for sampling of pseudostates
'''




# -------------------------------------------------------------------------------------------- #
# Sample parameters for the tax linear in 1 state and constant
# -------------------------------------------------------------------------------------------- #

# Tax function for linear tax (1 state) -> tax
def tax_function_lin_1(x, params):
    """
    Compute the tax linear in S
    """
    const, tau_x = params.T
    return const + tau_x * x

# sample parameters for the tax linear in 1 state and constant
def sample_params_lin1(
    num_samples, x0, xT_range, param_bounds, tax_range, rng, batch_size=1000
):
    """
    Sample parameters and compute tax values for t=0 and t=T in a vectorized manner,
    stopping when the desired number of feasible samples is reached.
    - num_samples: the number of samples to collect
    - x0: the initial state value
    - xT_range: the range of the final state value (upper bound from BAU, lower bound arbitrary)
    - param_bounds: the bounds for the parameters, arbitrary but hould be within some range
    - tax_range: the range for the tax values
    - rng: the random number generator
    - batch_size: the number of samples to generate at a time
    """
    num_params = len(param_bounds)
    samples_collected = []
    taxes_collected = []
    extremes_T = xT_range

    while len(samples_collected) < num_samples:
        # Generate a batch of samples
        params = uniform.rvs(
            size=(batch_size, num_params),
            loc=param_bounds[:, 0],
            scale=param_bounds[:, 1] - param_bounds[:, 0],
            random_state=rng,
        )
        
        # Compute tax values for t=0 and t=T
        tax_0 = tax_function_lin_1(x0, params)

      

        # ----------------------------------
        tax_T = np.array([tax_function_lin_1(x, params) for x in extremes_T])  # Shape: [num_combinations, batch_size]
        # Identify feasible samples
        tax_T_min = np.min(tax_T, axis=0) # get lowest possible tax_T value
        tax_T_max = np.max(tax_T, axis=0) # get max tax_T value

        feasible_mask = (
                    (tax_range[0] <= tax_0) & (tax_0 <= tax_range[1]) & # tax in period 0 should be larger than lower bound and lower than upper bound
                    (tax_range[0] <= tax_T_min) & (tax_T_max <= tax_range[1]) # tax in period T
                )
        # Collect feasible samples
        params_feasible = params[feasible_mask]

        # Append to the collected samples
        samples_collected.extend(params_feasible)
        

        # Stop if we've collected enough samples
        if len(samples_collected) >= num_samples:
            break

    # Return the desired number of samples
    return np.array(samples_collected[:num_samples])

# -------------------------------------------------------------------------------------------- #
# Sample parameters for the tax linear in 3 states and constant
# -------------------------------------------------------------------------------------------- #

# Tax function for linear tax (2 states) -> tax
def tax_function_lin_3(x, y, z, params):
    """
    Compute the tax linear in S
    """
    const, tau_x, tau_y, tau_z = params.T
    return const + tau_x * x + tau_y * y + tau_z * z


# sample parameters for the tax linear in 2 states and constant
def sample_params_lin3(
    num_samples, x0, y0, z0, xT_range, yT_range, zT_range, param_bounds, tax_range, rng, batch_size=1000
):
    """
    Sample parameters and compute tax values for t=0 and t=T in a vectorized manner,
    stopping when the desired number of feasible samples is reached.
    - num_samples: the number of samples to collect
    - x0, y0: the initial state values
    - xT_range, yT_range, zT_range: the ranges of the final state values (upper bounds from BAU, lower bounds arbitrary)
    - param_bounds: the bounds for the parameters, arbitrary but hould be within some range
    - tax_range: the range for the tax values
    - rng: the random number generator
    - batch_size: the number of samples to generate at a time
    """
    num_params = len(param_bounds)
    samples_collected = []
    taxes_collected = []
    extremes_T = [(a, b, c) for a in xT_range for b in yT_range for c in zT_range]

    while len(samples_collected) < num_samples:
        # Generate a batch of samples
        params = uniform.rvs(
            size=(batch_size, num_params),
            loc=param_bounds[:, 0],
            scale=param_bounds[:, 1] - param_bounds[:, 0],
            random_state=rng,
        )
        
        # Compute tax values for t=0 and t=T
        tax_0 = tax_function_lin_3(x0, y0, z0, params)
        tax_T = []
        # Compute tax values for all t=T combinations
        tax_T = np.array([tax_function_lin_3(xT, yT, zT, params) for xT, yT , zT in extremes_T])  # Shape: [num_combinations, batch_size]

        # Identify feasible samples
        tax_T_min = np.min(tax_T, axis=0) # get lowest possible tax_T value
        tax_T_max = np.max(tax_T, axis=0) # get max tax_T value
        feasible_mask = (
            (tax_range[0] <= tax_0) & (tax_0 <= tax_range[1]) & # tax in period 0 should be larger than lower bound and lower than upper bound
            (tax_range[0] <= tax_T_min) & (tax_T_max <= tax_range[1]) # tax in period T
        )
        
        # Collect feasible samples
        params_feasible = params[feasible_mask]

        # Append to the collected samples
        samples_collected.extend(params_feasible)
        

        # Stop if we've collected enough samples
        if len(samples_collected) >= num_samples:
            break

    # Return the desired number of samples
    return np.array(samples_collected[:num_samples])




# -------------------------------------------------------------------------------- #
def sample_uniform_perturbed(num_samples, rng, radius=0.03, upper_bound=0.3):
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
                noise = rng.uniform(-radius, radius, size=12)
                perturbed = base + noise
                
                # Ensure non-negativity
                perturbed = np.clip(perturbed, 0.00, None)
                
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

def sample_constrained_shares(n_samples, n_agents=12, max_share=0.5, rng=None):
    """
    This function samples dirichlet with an upper bound
    """
    alpha = np.ones(n_agents)
    if rng is not None:
        shares = rng.dirichlet(alpha, size=n_samples)
    else:
        shares = np.random.dirichlet(alpha, size=n_samples)
    mask_invalid = (shares > max_share).any(axis=1)
    
    while np.any(mask_invalid):
        n_invalid = np.sum(mask_invalid)
        if rng is not None:
            new_samples = rng.dirichlet(alpha, size=n_invalid)
        else:
            new_samples = np.random.dirichlet(alpha, size=n_invalid)
        shares[mask_invalid] = new_samples
        mask_invalid = (shares > max_share).any(axis=1)
    return np.clip(shares,a_min=1e-7, a_max=1)