# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import numpy as np

################################################
# constants from config file



constants = {'alpha': 0.3, 
            'beta': 0.9, # discount factor approx 0.95 per year
            'sigma': 3.0, # risk aversion
            'delta': 0.2, # 0.41 = approx. 0.1 per year
            'lamda': 1.7, # 1.5 climate sensitivity (for C concentration)
            'psi1': 13.16, # multiplier for damage function, original = 20.46, high damage = 13.16, super high d = 8.16
            'TP0': 3.0, # initial tipping point
            'vartheta': 0.015, # time scale parameter
            'theta2': 2.6, # Exponent of mitigation cost function
            'c2co2': 3.666, #transformation from c to co2
            'N': 12, # number of agents
            'Tstep': 5, # number of years per time step
            'rho0': 1.08, # initial ar value for kappa
            'rhoinfty': 0.91, # absorbing state for kappa
            'deltarho': 0.04, # rate of convergence to rhoinfty
    }

# labour endowments
# Initializing the array
A = constants['N']
LABOR_ENDOW = np.zeros(A)

# HARD CODED, NEED TO CHANGE if N is changed
retirement_age = (41.0/5.0)

# Loop to populate the array
for agent in range(1,A+1):
    a = agent
    if agent < retirement_age:
        a = a*constants['Tstep'] # convert to years to get hump shape
        LABOR_ENDOW[agent-1]= np.exp(4.47+0.033*a-0.00067*a**2)
    elif agent >= retirement_age:
        LABOR_ENDOW[agent-1]=0.4 *LABOR_ENDOW[round(retirement_age)-1] # -1 because of 0 indexing

# normalize labor to 1
LABOR_ENDOW = LABOR_ENDOW / sum(LABOR_ENDOW)
# Creating dictionary l_dict
l_dict = {'l{}'.format(i + 1): LABOR_ENDOW[i] for i in range(len(LABOR_ENDOW))}

l_total = {'L_total': sum(LABOR_ENDOW) * 600.0} # rescale

# Updating the dictionary
constants.update(l_dict)
constants.update(l_total)

################################################
# states

# create individual states
individual_states = []
for i in range(1, constants['N']+1):
    if i <retirement_age:
        capital_dict = {'name': 'a{}_x'.format(i)}
    elif i >=retirement_age:
        capital_dict = {'name': 'a{}_x'.format(i)}
    individual_states.append(capital_dict)

# global states
global_states = []
global_states.append({'name': 'K_total_x'})
global_states.append({'name': 'e_x'})
global_states.append({'name': 'S_x'})
global_states.append({'name': 'Temp_x'})
global_states.append({'name': 'Omega_x'})
# exogenous states
exogenous_states = []
# exogenous_states.append({'name': 'z_x'})
exogenous_states.append({'name': 'kappa_x'})
exogenous_states.append({'name': 'TP_x'})
exogenous_states.append({'name': 'TP_reached'})
exogenous_states.append({'name': 'tcomp_x'})

# total states
states = exogenous_states + individual_states + global_states

print("number of states", len(states))

#################################################
### Policies

# create individual policies
individual_policies = []
for i in range(1, constants['N']):
    if i <constants['N']-1:
        asset_dict = {'name': 'anext{}_y'.format(i)}
    elif i >=constants['N']-1:
        asset_dict = {'name': 'anext{}_y'.format(i), 'activation': 'tf.nn.softplus'}
    # asset_dict = {'name': 'anext{}_y'.format(i)}
    individual_policies.append(asset_dict)

# activation function
my_activation = "lambda x: -tf.nn.softplus(x)" 
# VFs
VF_y = []
for i in range(1,constants['N']):
    #VF_dict = {'name': 'v{}_y'.format(i)}
    VF_dict = {'name': 'v{}_y'.format(i), 'activation': my_activation}
    VF_y.append(VF_dict)

# global policies
global_policies = []

# total policies
policies = individual_policies + global_policies + VF_y

print("number of policies", len(policies))

##################################################
### definitions

individual_definitions = []
for i in range(1, constants['N']+1):
    cons = {'name': 'c{}'.format(i), 'bounds': {'lower': 3e-4}} 
    util = {'name': 'u{}'.format(i)}
    individual_definitions.append(cons)
    individual_definitions.append(util)

climate_definitions = []

Temp = {'name': 'Temp_x', 'activation': 'tf.nn.softplus'}
climate_definitions.append(Temp)

S_next = {'name': 'S_next', 'activation': 'tf.nn.softplus'}
climate_definitions.append(S_next)

rho_t = {'name': 'rho_t'}
climate_definitions.append(rho_t)


emissions = {'name': 'e_x', 'activation': 'tf.nn.softplus'}
climate_definitions.append(emissions)
climate_definitions.append({'name': 'Emissions'})

climate_definitions.append({'name': 'Omega_x', 'activation': 'tf.nn.sigmoid'})


global_definitions = []
K_total = {'name': 'K_total_x', 'bounds': {'lower': 1e-4}}
global_definitions.append(K_total)

K_total_next = {'name': 'K_total_next', 'bounds': {'lower': 1e-4}}
global_definitions.append(K_total_next)

r = {'name': 'r_x'}
global_definitions.append(r)

w = {'name': 'w_x'}
global_definitions.append(w)

global_definitions.append({'name': 'c_all', 'activation': 'tf.nn.softplus'})
global_definitions.append({'name': 'yprod'})
global_definitions.append({'name': 'ynet_x', 'activation': 'tf.nn.softplus'})
global_definitions.append({'name': 'u_all'})
global_definitions.append({'name': 'Damages_GDP'})


exo_definitions = []
exo_definitions.append({'name': 'tcomp2t'})
exo_definitions.append({'name': 'tcomp2tcompplus'})


# definitions
definitions =   individual_definitions + climate_definitions + global_definitions + exo_definitions 