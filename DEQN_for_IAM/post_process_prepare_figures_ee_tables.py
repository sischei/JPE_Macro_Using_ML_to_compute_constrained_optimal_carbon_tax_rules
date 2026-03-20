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
import utils_dataprep as udp


# --------------------------------------------------------------------------- #
# Manual Settings for post processing
# --------------------------------------------------------------------------- #

# Set the reference model 
BAU_MODEL_NAME = 'jpe_bau_final'

BAU_MODEL_SOLUTION = 'final' #  change to 'high_kappa' for replication of online appendix D.2.2 


# set rng for random pseudostate sampling
seed = 4914
rng = np.random.default_rng(seed)
# --------------------------------------------------------------------------- # 
# End of user inputs
# --------------------------------------------------------------------------- # 

# aditional stuff
os.makedirs(Parameters.LOG_DIR + '/latex', exist_ok=True)
path_latex = Parameters.LOG_DIR + '/latex'


MAIN_DIRECTORY = os.getcwd()

# path for BAU scenario (only relevant to load simulation)
path_BAU = os.getcwd() + '/runs/'+ BAU_MODEL_NAME+ '/' + BAU_MODEL_SOLUTION
# Get the size of the current terminal
terminal_size_col = shutil.get_terminal_size().columns

# Font size
# plt.rcParams["font.size"] = 12
# plt.rcParams["axes.labelsize"] = 12
# plt.rcParams["axes.titlesize"] = 12
# plt.rcParams["legend.title_fontsize"] = 12

plt.ticklabel_format(useOffset=False)

# Figure size
fsize = (9, 6.75)
line_args = {'markerfacecolor': 'None', 'color': 'tab:blue', 'marker': None,
             'linestyle': '-'}
distribution_args = {'markerfacecolor': 'None', 'color': 'tab:blue',
                     'marker': '.', 'linestyle': 'None'}



Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Dynamics = importlib.import_module(Parameters.MODEL_NAME + ".Dynamics")
import Globals
Globals.POST_PROCESSING=True

tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)
starting_policy = Parameters.policy(Parameters.starting_state)

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

lb_quantiles = [10, 25, 50, 75, 90]


# --------------------------------------------------------------------------- #
# Simulation periods and batch size
# --------------------------------------------------------------------------- #
begyear = 2015

# Number of years between each simulated episode, 12 geneations corresponds to 60 years
year_step = 5
# Simulate the economy for N_simulated episode length
# N_episode_length = Parameters.N_episode_length + 1
N_simulated_episode_length = 50 # we only need 50 for the transfers

N_simulated_epsiode_length_analysis = 29 # 29
N_simulated_batch_size = 10000 # Large to get good welfare estimates

ts = range(begyear,N_simulated_epsiode_length_analysis + begyear)

# Adjusted time axis to reflect the actual years
ts_adjusted = [2015 + year_step * t for t in range(len(ts))] 

ts_beg = begyear
ts_end = ts_adjusted[-1]
# add to environment variables (workaround for now)
if N_simulated_batch_size!= Parameters.N_simulated_batch_size:
    os.environ['N_simulated_batch_size'] = str(N_simulated_batch_size)

print("-" * terminal_size_col)
print("Simulate the economy for {} years".format(N_simulated_episode_length*5))
print("Number of simulated economies: {}".format(N_simulated_batch_size))

# Number of state, policy and defined variables
N_state = len(Parameters.states)  # Number of state variables
N_policy_state = len(Parameters.policy_states)  # Number of policy variables
N_definitions = len(Parameters.definitions)  # Number of defined variables

# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


simulation_starting_state = tf.tile(tf.expand_dims(Parameters.starting_state[0], axis=0), [N_simulated_batch_size, 1])

# --------------------------------------------------------------------------- #
# to set specific taxes for the simulation:

if Parameters.MODEL_NAME in ['jpe_pseudostate_const_S_trans_pension_risk_loose_test','jpe_pseudostate_const_S_trans_pension_risk_loose_scratch']:
 
    optimal_params = pd.read_csv(Parameters.LOG_DIR +"/optimal_parameters.csv")
    simulation_starting_state = State.update(simulation_starting_state, "tau_const_x", tf.constant(optimal_params['tau_const_x'].iloc[0],shape=(simulation_starting_state.shape[0],),dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tau_S_x", tf.constant(optimal_params['tau_S_x'].iloc[0],shape=(simulation_starting_state.shape[0],),dtype=tf.float32)) # initial S tax

    udp.write_optimal_tax_latex(optimal_params['tau_const_x'].iloc[0], optimal_params['tau_S_x'].iloc[0], Parameters.LOG_DIR + "/optimal_tax_parameters.tex")
    
    # also save it as csv
    pd.DataFrame({'tau_const_x': [optimal_params['tau_const_x'].iloc[0]], 'tau_S_x': [optimal_params['tau_S_x'].iloc[0]]}).to_csv(MAIN_DIRECTORY + "/Figures_Paper/Sec_5_2_optimal_tax.csv", index=False)






if Parameters.MODEL_NAME in ['jpe_pseudostate_S_transfers_risk','jpe_pseudostate_S_transfers_risk_test','jpe_pseudostate_S_transfers_risk_test_kappa']:



    # Load optimal parameters
    params_optimal = np.load(Parameters.LOG_DIR +"/pareto_models/optimal_params.npy")
    params_optimal = np.squeeze(params_optimal) 
    print("Optimal Parameters", params_optimal)
    print("Optimal Parameters shape", params_optimal.shape)

    # set taxes
    simulation_starting_state = State.update(simulation_starting_state, "tau_const_x", tf.constant(params_optimal[0],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tau_S_x", tf.constant(params_optimal[1],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial S tax
    
    udp.write_optimal_tax_latex(params_optimal[0], params_optimal[1], Parameters.LOG_DIR + "/optimal_tax_parameters.tex")

    # calculate theta1 definition
    theta1_check = Definitions.theta1(simulation_starting_state[0], None)
    print("theta1_check", theta1_check)
    if theta1_check == 0.7 and Parameters.MODEL_NAME == 'jpe_pseudostate_S_transfers_risk_test':
        # also save it as csv
        pd.DataFrame({'tau_const_x': [params_optimal[0]], 'tau_S_x': [params_optimal[1]]}).to_csv(Parameters.LOG_DIR + "/Table_5_tax.csv", index=False)
        # Generate the column names dynamically
        col_names = [f'tshare{i}_x' for i in range(1, 13)]

        # Create the DataFrame and export (note the extra brackets around params_optimal)
        pd.DataFrame([params_optimal[2:14]], columns=col_names).to_csv(
            Parameters.LOG_DIR + "/Table_6_transfers.csv",
            index=False
        )
    elif theta1_check == 0.6 and Parameters.MODEL_NAME == 'jpe_pseudostate_S_transfers_risk_test':
        # also save it as csv
        pd.DataFrame({'tau_const_x': [params_optimal[0]], 'tau_S_x': [params_optimal[1]]}).to_csv(Parameters.LOG_DIR + "/Appendix_Table_6_tax.csv", index=False)
        # Generate the column names dynamically
        col_names = [f'tshare{i}_x' for i in range(1, 13)]

        # Create the DataFrame and export (note the extra brackets around params_optimal)
        pd.DataFrame([params_optimal[2:14]], columns=col_names).to_csv(
            Parameters.LOG_DIR + "/Appendix_Table_7_transfers.csv",
            index=False
        )
    elif theta1_check == 0.7 and Parameters.MODEL_NAME == 'jpe_pseudostate_S_transfers_risk_test_kappa':
        # also save it as csv
        pd.DataFrame({'tau_const_x': [params_optimal[0]], 'tau_S_x': [params_optimal[1]]}).to_csv(Parameters.LOG_DIR + "/Appendix_Table_9_tax.csv", index=False)
        # Generate the column names dynamically
        col_names = [f'tshare{i}_x' for i in range(1, 13)]

        # Create the DataFrame and export (note the extra brackets around params_optimal)
        pd.DataFrame([params_optimal[2:14]], columns=col_names).to_csv(
            Parameters.LOG_DIR + "/Appendix_Table_10_transfers.csv",
            index=False
        )

    # set transfers 
    simulation_starting_state = State.update(simulation_starting_state, "tshare1_x", tf.constant(params_optimal[2] ,shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare2_x", tf.constant(params_optimal[3],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare3_x", tf.constant(params_optimal[4],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare4_x", tf.constant(params_optimal[5],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare5_x", tf.constant(params_optimal[6],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare6_x", tf.constant(params_optimal[7],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare7_x", tf.constant(params_optimal[8],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare8_x", tf.constant(params_optimal[9],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare9_x", tf.constant(params_optimal[10],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare10_x", tf.constant(params_optimal[11],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare11_x", tf.constant(params_optimal[12],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare12_x", tf.constant(params_optimal[13],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax

    udp.write_optimal_transfers_latex(params_optimal[2:14], Parameters.LOG_DIR + "/optimal_transfer_parameters.tex")
    

if Parameters.MODEL_NAME in ['jpe_pseudostate_linear_transfers_risk_test_implied']:

    # Load optimal parameters
    params_optimal = np.load(Parameters.LOG_DIR +"/pareto_models/optimal_params.npy")
    params_optimal = np.squeeze(params_optimal) 
    print("Optimal Parameters", params_optimal)
    print("Optimal Parameters shape", params_optimal.shape)

    # set taxex
    simulation_starting_state = State.update(simulation_starting_state, "tau_const_x", tf.constant(params_optimal[0],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial computational time
    simulation_starting_state = State.update(simulation_starting_state, "tau_S_x", tf.constant(params_optimal[1] ,shape=(simulation_starting_state.shape[0],), dtype=tf.float32))
    simulation_starting_state = State.update(simulation_starting_state, "tau_kappa_x", tf.constant(params_optimal[2] ,shape=(simulation_starting_state.shape[0],), dtype=tf.float32))
    simulation_starting_state = State.update(simulation_starting_state, "tau_TP_x", tf.constant(params_optimal[3],shape=(simulation_starting_state.shape[0],), dtype=tf.float32))
    simulation_starting_state = State.update(simulation_starting_state, "tau_x", Definitions.tau_x(simulation_starting_state, None))

    # set transfers 
    simulation_starting_state = State.update(simulation_starting_state, "tshare1_x", tf.constant(params_optimal[4],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare2_x", tf.constant(params_optimal[5],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare3_x", tf.constant(params_optimal[6],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare4_x", tf.constant(params_optimal[7],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare5_x", tf.constant(params_optimal[8],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare6_x", tf.constant(params_optimal[9],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare7_x", tf.constant(params_optimal[10],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare8_x", tf.constant(params_optimal[11],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare9_x", tf.constant(params_optimal[12],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare10_x", tf.constant(params_optimal[13],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare11_x", tf.constant(params_optimal[14],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax
    simulation_starting_state = State.update(simulation_starting_state, "tshare12_x", tf.constant(params_optimal[15],shape=(simulation_starting_state.shape[0],), dtype=tf.float32)) # initial constant tax

    # save optimal parameters
    udp.write_optimal_tax_4_instruments_latex(params_optimal[0], params_optimal[1], params_optimal[2], params_optimal[3], Parameters.LOG_DIR + "/optimal_tax_parameters.tex")
    # also save as csv (Table 8)
    pd.DataFrame({
        'tau_const_x': [params_optimal[0]],
        'tau_S_x':     [params_optimal[1]],
        'tau_kappa_x': [params_optimal[2]],
        'tau_TP_x':    [params_optimal[3]],
    }).to_csv(Parameters.LOG_DIR + "/Table_8_optimal_tax_parameters.csv", index=False)

    # save optimal transfers
    udp.write_optimal_transfers_latex(params_optimal[4:16], Parameters.LOG_DIR + "/optimal_transfer_parameters.tex")
    # also save as csv (Table 9)
    col_names = [f'tshare{i}_x' for i in range(1, 13)]
    pd.DataFrame([params_optimal[4:16]], columns=col_names).to_csv(
        Parameters.LOG_DIR + "/Table_9_optimal_transfer_parameters.csv", index=False
    )

    
    





print('N_simulated_episode_length', N_simulated_episode_length)
#print('simulation_starting_state', simulation_starting_state)
state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, 1, 1])


print("Running episode to get range of variables...")

state_episode = run_episode(state_episode)
#print("state_episode", state_episode)




# calculate average and min/max values of states over the simulated episode
mean_states = tf.math.reduce_mean(state_episode, axis = 1)
min_states = tf.math.reduce_min(state_episode, axis = 1)
max_states = tf.math.reduce_max(state_episode, axis = 1)

# create numpy arrays for plotting
state_episode_batch = state_episode.numpy()

# calculate states scaled
state_episode_batch_scaled = np.empty_like(state_episode_batch, dtype=np.float32)
for sidx, state in enumerate(Parameters.states):
    for tidx in range(N_simulated_episode_length):
        state_batch = state_episode_batch[tidx,:,:]
        state_val = getattr(State,state)(state_batch)
        state_episode_batch_scaled[tidx, :, sidx] = state_val
    


# Policy variables for N_sim_batch times
policy_state_episode_batch = np.empty(
    shape=[N_simulated_episode_length, N_simulated_batch_size, N_policy_state], dtype=np.float32)
for tidx in range(N_simulated_episode_length):
    policy_state_batch = Parameters.policy(state_episode_batch[tidx, :, :])
    policy_state_episode_batch[tidx, :, :] = policy_state_batch


# Policy variables
policy_state_episode_batch_scaled = np.empty_like(
    policy_state_episode_batch, dtype=np.float32)

for pidx, policy in enumerate(Parameters.policy_states):
    # Adjust policy variables
    for tidx in range(N_simulated_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        policy_val = getattr(PolicyState, policy)(policy_state_batch)
        policy_state_episode_batch_scaled[tidx, :, pidx] = policy_val



print("Calculating defined variables...")
# Definitions
definition_episode_batch = np.empty(
    shape=[N_simulated_episode_length, N_simulated_batch_size, N_definitions], dtype=np.float32)

for didx, de in enumerate(Parameters.definitions):
    for tidx in range(N_simulated_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        defined_val = getattr(Definitions, de)(state_batch, policy_state_batch)
        definition_episode_batch[tidx, :, didx] = defined_val

print("Defined variables calculation Done...")






# ----------------------------------------------------------------------------- #
# Shorten simulation as we only need the first 150 periods
# ----------------------------------------------------------------------------- #
# Note: For the calculation of transfers we use 50 periods, after that we only consider 29 periods
# for the calculation of welfare
N_simulated_episode_length = N_simulated_epsiode_length_analysis
state_episode = state_episode[:N_simulated_episode_length, :, :] # raw states
state_episode_batch = state_episode_batch[:N_simulated_episode_length, :, :] # raw states
state_episode_batch_scaled = state_episode_batch_scaled[:N_simulated_episode_length, :, :] # states
policy_state_episode_batch = policy_state_episode_batch[:N_simulated_episode_length, :, :] # raw policies
policy_state_episode_batch_scaled = policy_state_episode_batch_scaled[:N_simulated_episode_length, :, :] # policies
definition_episode_batch = definition_episode_batch[:N_simulated_episode_length, :, :] # definitions

mean_states = mean_states[:N_simulated_episode_length, :] # mean states
min_states = min_states[:N_simulated_episode_length, :] # min states
max_states = max_states[:N_simulated_episode_length, :] # max states

# --------------------------------------------------------------------------- #
# calculate welfare
# --------------------------------------------------------------------------- #
print("Calculating welfare...")
# generate welfare array 
index_v1 = Parameters.policy_states.index('v1_y')
index_u12 = Parameters.definitions.index('u12')

# define the value function policies for the welfare calculation in reverse order to get oldest first
value_function_policies = [f"v{i}_y" for i in range(Parameters.N-1, 0, -1)]

indexes_v = [Parameters.policy_states.index(i) for i in value_function_policies]
#print("indexes_v", indexes_v)
# remaining lifetime utility of the living generations
welfares_first_period = tf.transpose(policy_state_episode_batch_scaled[0, :, indexes_v])

# extract utility of agent 12 in the first period
welfare_oldest = definition_episode_batch[0, :, index_u12].reshape(-1, 1)

welfares_first_period = np.concatenate((welfare_oldest, welfares_first_period), axis=1)

# lifetime utilities of generations born in each period
welfares_next_periods = tf.transpose(policy_state_episode_batch_scaled[1:, :, index_v1])

# all welfares 
welfares = tf.concat([welfares_first_period, welfares_next_periods], axis=1)


# calculate mean welfare
mean_welfare = tf.reduce_mean(welfares, axis=0)
# save the welfares
print("Saving welfares of BAU scenario...")


mean_welfare_df = pd.DataFrame(mean_welfare.numpy())
mean_welfare_df.to_csv(Parameters.LOG_DIR +'/welfares_simulated.csv', index=False)


print("-" * terminal_size_col)

# --------------------------------------------------------------------------- #
# Create plots
# --------------------------------------------------------------------------- #
print("Creating plots...")

# --------------------------------------------------------------------------- #
# plot distributions of all states separately

state_labels = {
    "Temp_x": "Temperature (°C)",
    "S_x": "Carbon Stock (GtC)",
    # "TP_x": "Tipping Probability",
    "Damages_GDP": "Damages (share of final output)",
    "tau_x": "tax rate (τ)"
    # Add other states as needed
}
# Compute the quantiles of each variable along with the number of simulations
quantile_state = np.percentile(
    state_episode_batch_scaled, q=lb_quantiles, axis=1)
quantile_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=lb_quantiles, axis=1)
quantile_defined = np.percentile(
    definition_episode_batch, q=lb_quantiles, axis=1)

# Compute the range of each variable
range_state = np.percentile(state_episode_batch_scaled, q=[1, 99], axis=1)
range_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=[1, 99], axis=1)
range_defined = np.percentile(
    definition_episode_batch, q=[1, 99], axis=1)

# Compute the average of each variable
avg_state = np.average(state_episode_batch_scaled, axis=1)
avg_policy_state = np.average(policy_state_episode_batch_scaled, axis=1)
avg_defined = np.average(definition_episode_batch, axis=1)

# Plot the distribution of state variables
for sidx, state in enumerate(Parameters.states):
    if state not in state_labels:
        continue

    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts_adjusted, range_state[0, :, sidx], range_state[1, :, sidx],
        facecolor='tab:gray', alpha=0.3,
        label=r'Range of sample paths (1% to 99%)')
    for qidx in range(len(lb_quantiles)):
        q_val = lb_quantiles[qidx]
        if q_val == 50:
            ls = '-'
            color = 'black'
        elif q_val in [25, 75]:
            ls = '--'
            color = 'blue'
        elif q_val in [10, 90]:
            ls = ':'
            color = 'red'
        else:
            ls = '-'
            color = 'black'            
        ax.plot(ts_adjusted, quantile_state[qidx, :, sidx],
                label=r'{}% quantile'.format(lb_quantiles[qidx]), linestyle=ls, color=color, linewidth=2)
    ax.set_xlabel('Year', fontsize=14, fontweight="bold")
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(state_labels[state], fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', labelsize=12)  
    ax.tick_params(axis='y', labelsize=12)   
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(
        Parameters.LOG_DIR + '/states_distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + state + '.pdf', bbox_inches='tight')
    plt.close()

    # --- CSV export for state distribution figure ---
    csv_stem = 'states_distribution_' + str(ts_beg) + '-' + str(ts_end) + '_' + state
    pd.DataFrame({
        'year': ts_adjusted,
        'range_1pct':  range_state[0, :, sidx],
        'range_99pct': range_state[1, :, sidx],
        'q10': quantile_state[0, :, sidx],
        'q25': quantile_state[1, :, sidx],
        'q50': quantile_state[2, :, sidx],
        'q75': quantile_state[3, :, sidx],
        'q90': quantile_state[4, :, sidx],
    }).to_csv(Parameters.LOG_DIR + '/' + csv_stem + '.csv', index=False)



# define list of variables to save for latex
variables_to_save = ['Temp_x', 'S_x', 'Omega_x', 'TP_x', 'mu_x', 'tau_x']
# plot the distribution of defined variables
for didx, definition in enumerate(Parameters.definitions):
    if definition not in state_labels:
        continue

    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts_adjusted, range_defined[0, :, didx], range_defined[1, :, didx],
        facecolor='tab:gray', alpha=0.3,
        label=r'Range of sample paths (1% to 99%)')
    for qidx in range(len(lb_quantiles)):
        q_val = lb_quantiles[qidx]
        
        # 1. Determine styles AND specific labels based on the quantile
        if q_val == 50:
            ls = '-'
            color = 'black'
            label_text = '50% quantile'
        elif q_val == 25:
            ls = '--'
            color = 'blue'
            label_text = '25–75% quantile' # Set label on the lower bound
        elif q_val == 75:
            ls = '--'
            color = 'blue'
            label_text = None              # Hide label for the upper bound
        elif q_val == 10:
            ls = ':'
            color = 'red'
            label_text = '10–90% quantile' # Set label on the lower bound
        elif q_val == 90:
            ls = ':'
            color = 'red'
            label_text = None              # Hide label for the upper bound
        else:
            ls = '-'
            color = 'black'
            label_text = '{}% quantile'.format(q_val)
            
        # 2. Pass the conditional label_text to the plot function
        ax.plot(ts_adjusted, quantile_defined[qidx, :, didx],
                label=label_text, linestyle=ls, color=color, linewidth=2)
        
    ax.set_xlabel('Year', fontsize=14, fontweight="bold")
    ax.set_xlim([ts_beg, ts_end])
    #plt.ylabel(definition_labels[defined])
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.ticklabel_format(useOffset=False)
    # ax.set_ylabel(definition, fontsize=14, fontweight="bold")
    ax.set_ylabel(state_labels[definition], fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', labelsize=12)  
    ax.tick_params(axis='y', labelsize=12) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(
        Parameters.LOG_DIR + '/def_distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + definition + '.pdf')
    if definition in variables_to_save:
        plt.savefig(path_latex + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
                    + '_' + definition + '.pdf')
    plt.close()

    # --- CSV export for definition distribution figure ---
    csv_stem = 'def_distribution_' + str(ts_beg) + '-' + str(ts_end) + '_' + definition
    pd.DataFrame({
        'year': ts_adjusted,
        'range_1pct':  range_defined[0, :, didx],
        'range_99pct': range_defined[1, :, didx],
        'q10': quantile_defined[0, :, didx],
        'q25': quantile_defined[1, :, didx],
        'q50': quantile_defined[2, :, didx],
        'q75': quantile_defined[3, :, didx],
        'q90': quantile_defined[4, :, didx],
    }).to_csv(Parameters.LOG_DIR + '/' + csv_stem + '.csv', index=False)

# ---------------------------------------------------- #
# create a plot of climate variables together 
series = ['Temp_x', 'S_x', 'Omega_x', 'TP_x', 'kappa_x']
df_all = pd.DataFrame()
for s in series:
    y_mean = getattr(State,s)(mean_states).numpy()
    
   
    df_all[s] = y_mean

# add years to df_all, starting year is 2015 and we have a datapoint every 5 years
df_all['year'] = [begyear + i * 5 for i in range(N_simulated_episode_length)]
# add Emissions (note lb_quantiles = [10, 25, 50, 75, 90])
df_all['Emissions'] = definition_episode_batch[:, :, Parameters.definitions.index('Emissions')].mean(axis=1) / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_1'] = range_defined[0, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_99'] = range_defined[1, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_10'] = quantile_defined[0, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_25'] = quantile_defined[1, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_50'] = quantile_defined[2, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_75'] = quantile_defined[3, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.
df_all['Emissions_90'] = quantile_defined[4, :, Parameters.definitions.index('Emissions')] / year_step # rescale emissions to get GtCO2 per year.


# --------------------------------------------------------------------------- #
# plot emissions and scenarios
# load data from calibration_data.xlsx
calibration_data = pd.read_excel("calibration_data/dice_rcp_data.xlsx", skiprows=1)

# filter out every fifth year
calibration_data = calibration_data[calibration_data['year'] % 5 == 0]


if Parameters.MODEL_NAME == BAU_MODEL_NAME:
    print("Saving emissions data...")
    emissions_df = df_all[['year', 'Emissions']]

    # rename e_x_iota to emissions_BAU
    emissions_df.rename(columns={'Emissions': 'emissions_BAU'}, inplace=True)
    emissions_df.to_csv(Parameters.LOG_DIR + '/emissions_BAU.csv', index=False)
else:
    # load emissions_BAU
    emissions_BAU = pd.read_csv(path_BAU + '/emissions_BAU.csv')
    print("emissions_BAU", emissions_BAU.head())
    # merge calibration_data with emissions_BAU
    calibration_data = calibration_data.merge(emissions_BAU, how='left', on='year')


# merge calibration_data with df_all
df_all = df_all.merge(calibration_data, how='left', on='year')

print("df_all", df_all.head())
calibration_cols = list(calibration_data.drop(columns=['year'], inplace=False).columns)
print("calibration_cols", calibration_cols)


# delete years after 2300
df_all = df_all[df_all['year'] < 2301]

if Parameters.MODEL_NAME == BAU_MODEL_NAME:

    # === NEW PLOTTING CODE =====
    colors_dict = {
        calibration_cols[0]: '#d62728',     # Dark Red
        calibration_cols[1]: '#2ca02c',     # Dark Green
        calibration_cols[2]: 'blue',     # Orange
        calibration_cols[3]: '#ff7f0e',     # Purple
        'emissions_BAU': 'black'
    }

    # Add markers to save the scenarios in B&W
    markers_dict = {
        calibration_cols[0]: 'o',           # Circle
        calibration_cols[1]: 's',           # Square
        calibration_cols[2]: '^',           # Triangle
        calibration_cols[3]: 'D',           # Diamond
        'emissions_BAU': None
    }

    styles_dict = {
        calibration_cols[0]: '-',
        calibration_cols[1]: ':',
        calibration_cols[2]: '-.',
        calibration_cols[3]: '--',
        'emissions_BAU': '-'
    }

    plt.figure(figsize=fsize)
    ax = plt.gca()
    year = df_all['year']

    # --- UNCERTAINTY BANDS (Shaded Areas & Lines) ---
    fill_1_99 = ax.fill_between(year, df_all['Emissions_1'], df_all['Emissions_99'], color='tab:gray', alpha=0.3, zorder=1)

    # --- CALIBRATION SCENARIOS ---
    rcp_handles = []
    rcp_labels = []
    for column in calibration_cols:
        if column in df_all.columns:
            h, = ax.plot(
                year, 
                df_all[column],
                color=colors_dict[column],
                linestyle=styles_dict[column], 
                marker=markers_dict[column], 
                markersize=7, 
                markerfacecolor='white', 
                markevery=5,             
                linewidth=2, 
                zorder=5
            )
            rcp_handles.append(h)
            rcp_labels.append(column)

    # --- MEAN LINE ---
    l_mean, = ax.plot(year, df_all['Emissions'], color='black', linewidth=2, zorder=6)

    # Labels and formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Emissions (GtCO2)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # --- LEGEND CONSTRUCTION ---
    legend_handles = rcp_handles + [l_mean, fill_1_99] 
    legend_labels = rcp_labels + ['Emissions (mean)', '1–99% Range']

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=12,
        frameon=True,
        edgecolor='black'
    )

    # Layout adjustments
    # plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)

    # Save plot
    plt.savefig(Parameters.LOG_DIR + '/climate_mean_simulation_calibration_uncertainty.pdf', bbox_inches='tight')
    plt.savefig(path_latex + '/climate_mean_simulation_calibration_uncertainty.pdf', bbox_inches='tight')
    plt.close()

    # --- CSV export for BAU emissions figure ---
    _csv_cols = ['year', 'Emissions', 'Emissions_1', 'Emissions_99',
                 'Emissions_10', 'Emissions_25', 'Emissions_50', 'Emissions_75', 'Emissions_90']
    _csv_cols += [c for c in calibration_cols if c in df_all.columns]
    df_all[[c for c in _csv_cols if c in df_all.columns]].to_csv(
        Parameters.LOG_DIR + '/climate_mean_simulation_calibration_uncertainty.csv', index=False)

else: 

    
    # ---------------------------------------------------------
    # 1. UPDATED DICTIONARIES (Darker colors + Markers)
    # ---------------------------------------------------------
    colors_dict = {
        calibration_cols[0]: '#d62728',     # Dark Red
        calibration_cols[1]: '#2ca02c',     # Dark Green
        calibration_cols[2]: 'blue',     # Orange
        calibration_cols[3]: '#ff7f0e',     # Purple
        calibration_cols[4]: 'black',
        'emissions_BAU': 'black',
        'Emissions': 'black'                # Keep mean pure black for dominance
    }

    # Add markers to save the scenarios in B&W
    markers_dict = {
        calibration_cols[0]: 'o',           # Circle
        calibration_cols[1]: 's',           # Square
        calibration_cols[2]: '^',           # Triangle
        calibration_cols[3]: 'D',           # Diamond
        calibration_cols[4]: 'v',           # Down Triangle
        'emissions_BAU': None,
        'Emissions': None
    }

    styles_dict = {
        calibration_cols[0]: '-',
        calibration_cols[1]: ':',
        calibration_cols[2]: '-.',
        calibration_cols[3]: '--',
        calibration_cols[4]: '--',
        'emissions_BAU': '--',
        'Emissions': '-'
    }

    # ... (Assume Plot 1 code remains mostly the same but uses markers) ...

    # ---------------------------------------------------------
    # 2. PLOT 2: UNCERTAINTY GRAPH (Print Compliant)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. PLOT 2: UNCERTAINTY GRAPH 
    # ---------------------------------------------------------
    plt.figure(figsize=fsize)
    ax = plt.gca()
    year = df_all['year']

    # --- UNCERTAINTY BANDS (Shaded Areas & Lines) ---
    # Uncomment the 'fill_between' or 'ax.plot' lines you want to use.

    # 1. Shaded Areas 
    # fill_25_75 = ax.fill_between(year, df_all['Emissions_25'], df_all['Emissions_75'], color='tab:gray', alpha=0.4, zorder=1)
    # fill_10_90 = ax.fill_between(year, df_all['Emissions_10'], df_all['Emissions_90'], color='tab:gray', alpha=0.3, zorder=1)
    fill_1_99 = ax.fill_between(year, df_all['Emissions_1'], df_all['Emissions_99'], color='tab:gray', alpha=0.3, zorder=1)

    # 2. Boundary Lines
    # l1, = ax.plot(year, df_all['Emissions_25'], color='darkgray', linestyle='--', linewidth=2, zorder=4)
    # ax.plot(year, df_all['Emissions_75'], color='darkgray', linestyle='--', linewidth=2, zorder=4)
    # 
    # l2, = ax.plot(year, df_all['Emissions_10'], color='gray', linestyle='-.', linewidth=2, zorder=4)
    # ax.plot(year, df_all['Emissions_90'], color='gray', linestyle='-.', linewidth=2, zorder=4)
    # 
    # l3, = ax.plot(year, df_all['Emissions_1'], color='dimgray', linestyle=(0, (1, 2)), linewidth=2.5, zorder=4)
    # ax.plot(year, df_all['Emissions_99'], color='dimgray', linestyle=(0, (1, 2)), linewidth=2.5, zorder=4)

    # --- CALIBRATION SCENARIOS ---
    rcp_handles = []
    rcp_labels = []
    for column in calibration_cols:
        if column in df_all.columns:
            h, = ax.plot(
                year, 
                df_all[column],
                color=colors_dict[column],
                linestyle=styles_dict[column], 
                marker=markers_dict[column], 
                markersize=7, 
                markerfacecolor='white', 
                markevery=5,             
                linewidth=2, 
                zorder=5
            )
            rcp_handles.append(h)
            rcp_labels.append(column)

    # --- MEAN LINE ---
    l_mean, = ax.plot(year, df_all['Emissions'], color='black', linewidth=3.5, zorder=6)

    # Labels and formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Emissions (GtCO2)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # --- LEGEND CONSTRUCTION ---
    # Add whatever handles you uncommented above to these lists
    legend_handles = rcp_handles + [l_mean, fill_1_99] 
    legend_labels = rcp_labels + ['Emissions (mean)', '1–99% Range']
    
    # If you wanted to show all lines instead, it would look like this:
    # legend_handles = rcp_handles + [l_mean, l1, l2, l3]
    # legend_labels = rcp_labels + ['Emissions (mean)', '25–75% Range', '10–90% Range', '1–99% Range']

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=12,
        frameon=True,
        edgecolor='black'
    )

    # Layout adjustments
    # plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)

    # Save plot
    plt.savefig(Parameters.LOG_DIR + '/climate_mean_simulation_calibration_uncertainty.pdf', bbox_inches='tight')
    # plt.savefig(path_latex + '/climate_mean_simulation_calibration_uncertainty.pdf', bbox_inches='tight')
    plt.close()

    # --- CSV export for non-BAU emissions figure ---
    _csv_cols = ['year', 'Emissions', 'Emissions_1', 'Emissions_99',
                 'Emissions_10', 'Emissions_25', 'Emissions_50', 'Emissions_75', 'Emissions_90']
    if 'emissions_BAU' in df_all.columns:
        _csv_cols.append('emissions_BAU')
    _csv_cols += [c for c in calibration_cols if c in df_all.columns]
    df_all[[c for c in _csv_cols if c in df_all.columns]].to_csv(
        Parameters.LOG_DIR + '/climate_mean_simulation_calibration_uncertainty.csv', index=False)


# --------------------------------------------------------------------------- #
# Euler discrepancies
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Finished plots. Calculating Euler discrepancies...")

print("-" * terminal_size_col)
print(r"Compute the Euler discrepancies for {} years in {} simulation "
      "batch".format(N_simulated_episode_length, N_simulated_batch_size))


state_episode_export = state_episode
## calculate euler deviations
state_episode = tf.reshape(state_episode, [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])
policy_episode = Parameters.policy(state_episode)
euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))
print("Euler discrepancy (absolute value) metrics")

euler_discrepancies_des = euler_discrepancies.abs().describe([.25, .5, .75, .99, 0.999],include='all')
print(euler_discrepancies_des)



print("Saving Euler discrepancies...")
# # save all relevant quantities along the trajectory 
euler_discrepancies.to_csv(Parameters.LOG_DIR + "/simulated_euler_discrepancies.csv", index=False)
euler_discrepancies_des.to_csv(Parameters.LOG_DIR + "/simulated_euler_discrepancies_des.csv", index=True)



# save the tex
udp.save_latex_table_to_file(euler_discrepancies_des, Parameters.LOG_DIR +"/simulated_euler_disc_latex.tex")


state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
state_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_states.csv", index=False)

policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
policy_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_policies.csv", index=False)
policy_episode_export = tf.reshape(policy_episode, [N_simulated_episode_length, N_simulated_batch_size, len(Parameters.policy_states)])




definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
definition_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_definitions.csv", index=False)


print("State metrics")
print(state_episode_df.describe(include='all'))

print("Policy metrics")
print(policy_episode_df.describe(include='all'))

print("Definition metrics")
print(definition_episode_df.describe(include='all'))

# --------------------------------------------------------------------------- #
# export to latex folder
# --------------------------------------------------------------------------- #
# Replace underscores in column names and index with \_
euler_discrepancies_des.columns = [col.replace('_', r'\_') for col in euler_discrepancies_des.columns]
euler_discrepancies_des.index = [idx.replace('%', r'\%') for idx in euler_discrepancies_des.index]
cols_to_write = ['ee\_1', 'ee\_2', 'ee\_3', 'ee\_4', 'ee\_5', 'ee\_6', 'ee\_7', 'ee\_8', 'ee\_9', 'ee\_10', 'ee\_11']
euler_discrepancies_des.to_latex(path_latex + "/sim_euler_discrepancies_d.tex",index=True, float_format="%.3f", columns=cols_to_write)
euler_discrepancies_des.to_csv(path_latex + "/sim_euler_discrepancies_d.csv", index=False)
# state_episode_df.to_csv(path_latex + "/sim_states.csv", index=False)
# policy_episode_df.to_csv(path_latex + "/sim_policies.csv", index=False)
# definition_episode_df.to_csv(path_latex + "/sim_definitions.csv", index=False)


# --------------------------------------------------------------------------- #
# welfare gains of tax policy
# --------------------------------------------------------------------------- #
N_periods_welfare = 40


# exit if we are in the BAU model
if Parameters.MODEL_NAME == BAU_MODEL_NAME:
    print("Exiting the program since we are in the BAU model")
    sys.exit()

    

print("-" * terminal_size_col)
print("Welfare analysis")

# load results from BAU scenario (TIME is in axis 1)
# welfares_BAU = pd.read_csv(path_BAU + '/sim_welfares.csv')

welfares_BAU = pd.read_csv(path_BAU + '/sim_mean_welfare.csv')

# ----------------------------
# welfare weights specification 
# ----------------------------



def generate_welfare_weights(growth_rate: float, n_generations: int = 40, first_n_equal: int = 12):
    """
    Generate welfare weights:
    - First `first_n_equal` generations have equal weights summing to 1.
    - Remaining generations grow by `growth_rate` per generation.

    Args:
        growth_rate (float): Growth rate per generation (e.g., 0.02 for 2%).
        n_generations (int): Total number of generations (default 40).
        first_n_equal (int): Generations with equal weight (default 12).

    Returns:
        np.array: Welfare weights summing to 1.
    """
    # Equal weights for first generations
    equal_weight = 1 / first_n_equal
    weights = [equal_weight] * first_n_equal

    # Grow the rest geometrically
    for i in range(first_n_equal, n_generations):
        weight = weights[-1] * (1 + growth_rate)
        weights.append(weight)

    # Normalize to sum to 1
    weights = np.array(weights)
    weights /= weights.sum()

    return weights

welfare_weights = generate_welfare_weights(growth_rate=0.0, n_generations=40, first_n_equal=12)


print("welfare_weights", tf.transpose(welfare_weights))


# calculate weighted sum of welfare weights welfares are calculated using Monte Carlo
# welfares_BAU = tf.cast(welfares_BAU.iloc[:, :N_periods_welfare], dtype=tf.float32)
# welfares_BAU = tf.reduce_mean(welfares_BAU, axis=0) # Monte Carlo average

welfares_BAU = tf.cast(welfares_BAU["0"], dtype=tf.float32)
welfares_tax = tf.reduce_mean(welfares, axis=0)[:N_periods_welfare]

print("welfares_BAU", welfares_BAU)
print("welfares_tax", welfares_tax)


# calculate welfare gains of tax policy
average_welfare_gains = (welfares_tax/welfares_BAU) -1

# consumption equivalent welfare gains
consumption_equivalent_gains = (welfares_tax/welfares_BAU)**(1/(1-Parameters.sigma)) -1



# Calculate birth years for each generation
start_year = 2015 - 5 * 12  # First generation birth year
birth_years = [start_year + i * 5 for i in range(40)]

# Create bar plot
plt.figure(figsize=fsize)
plt.bar(birth_years, consumption_equivalent_gains, color="black", width=1.5)

# Label the plot
plt.xlabel("Year entering work force", fontsize=14, fontweight='bold')
plt.ylabel("Consumption equiv. gains/losses", fontsize=14, fontweight='bold')
# plt.title("Consumption equivalent gains of tax policy", fontsize=16)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)  
plt.tick_params(axis='y', labelsize=12) 
# plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig(Parameters.LOG_DIR + '/welfare_gains.pdf')
plt.savefig(path_latex + '/welfare_gains.pdf')
plt.close()

# welfare_weights = tf.constant([0.025]*40, dtype=tf.float32)


# planner's welfare
planner_welfare = tf.reduce_sum(consumption_equivalent_gains)

print('welfare gain: ', average_welfare_gains)
print('consumption_equivalent_gains: ', consumption_equivalent_gains)
print('planner_welfare: ', planner_welfare)
print('weighted sum of welfares with tax: ', tf.reduce_sum(welfare_weights * welfares_tax))
print('weighted sum of welfares with BAU: ', tf.reduce_sum(welfare_weights * welfares_BAU))
print("weighted CE gains: ", tf.reduce_sum(welfare_weights * consumption_equivalent_gains))
# calculate per period average welfare gains
# save consumption equivalent gains
welfare_gains_df = pd.DataFrame({
    'birth_year': birth_years,
    'welfare_gains': average_welfare_gains.numpy(),
    'consumption_equivalent_gains': consumption_equivalent_gains.numpy()
})
welfare_gains_df.to_csv(Parameters.LOG_DIR + '/welfare_gains.csv', index=False)


welfare_weights_df = pd.DataFrame({'welfare_weights': welfare_weights})
welfare_weights_df.to_csv(Parameters.LOG_DIR + '/welfare_weights.csv', index=False)
social_welfare_df = pd.DataFrame(
    {'social_welfare_CE': [tf.reduce_sum(welfare_weights * consumption_equivalent_gains).numpy()]})
social_welfare_df.to_csv(Parameters.LOG_DIR + '/social_welfare.csv', index=False)


