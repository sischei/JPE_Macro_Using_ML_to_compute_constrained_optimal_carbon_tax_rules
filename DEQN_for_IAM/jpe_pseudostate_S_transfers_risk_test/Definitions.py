import Parameters
import PolicyState
import State
import sys
import tensorflow as tf
from Parameters import alpha, beta, delta , sigma, lamda, psi1, vartheta, theta2, c2co2, Tstep, L_total, N, rho0, rhoinfty, deltarho


def theta1(state, policy_state=None):
    """ Cost coefficient of carbon mitigation """
    # _theta1 = 0.6
    _theta1 = 0.7
    return _theta1

# # mu derived through firm foc
def mu_y(state, policy_state=None):
    _mu = tf.where(State.kappa_x(state) == 0.0, .0, ((tau_x(state, policy_state) * State.kappa_x(state)) / (State.Omega_x(state) * theta1(state, policy_state) * theta2) ) ** (1/(theta2-1))) 
    return _mu


def mu_cost(state, policy_state=None):
    return 1 - theta1(state, policy_state) * mu_y(state,policy_state) ** theta2


# tax rate on carbon emissions derived through firm foc
def tau_x(state, policy_state):
    _tau = State.tau_const_x(state) + State.tau_S_x(state) * State.S_x(state)
    _tau = tf.clip_by_value(_tau, 1e-7, 2.0) # ensure to clip in case of numerical errors
    return  _tau


def tax_income(state, policy_state=None):
    return tau_x(state, policy_state) * e_x(state, policy_state) 

def K_total_x(state, policy_state=None):
    return tf.reduce_sum(tf.stack([(getattr(State,"a" + str(i) + "_x")(state)) for i in range(1, Parameters.N + 1)],axis=1),axis=1)

# next period capital used to update K_total_x
def K_total_next(state, policy_state):
    return tf.reduce_sum(tf.stack([(getattr(PolicyState,"anext" + str(i) + "_y")(policy_state)) for i in range(1, Parameters.N)],axis=1),axis=1)

def r_x(state, policy_state=None):
    return (mu_cost(state,policy_state) * State.Omega_x(state) - tau_x(state, policy_state) * State.kappa_x(state) * (1 - mu_y(state,policy_state))) * alpha * (State.K_total_x(state)**(alpha - 1))  - delta

def w_x(state, policy_state=None):
    return (mu_cost(state,policy_state) * State.Omega_x(state) - tau_x(state, policy_state) * State.kappa_x(state) * (1 - mu_y(state,policy_state))) * (1 - alpha) * (State.K_total_x(state))**alpha

def yprod(state, policy_state=None):
    '''Production function'''
    return  State.K_total_x(state)**Parameters.alpha 

# Real and computational time periods
def tcomp2t(state, policy_state):
    """ Scale back from the computational time tau to the real time t """
    _t = - tf.math.log(1 - State.tcomp_x(state)) / vartheta
    return _t


def tcomp2tcompplus(state, policy_state):
    """ Update the computational time tau by tau + 1 based on the current real
    time t """
    _t = tcomp2t(state, policy_state)  # Current real time
    _tplus = _t + tf.ones_like(_t)  # Real time t + 1
    _tauplus = 1 - tf.math.exp(- vartheta * _tplus)  # Computational time tau+1
    return _tauplus

# AR factor for carbon intensity
def rho_t(state, policy_state):
    """ World population [million] """
    _t = tcomp2t(state, policy_state)
    _rho = rho0 + (rhoinfty - rho0) * (1.0 - tf.math.exp(-Tstep * deltarho * _t))
    return _rho

# transfer definitions
for i in range(1,Parameters.N + 1):
    setattr(sys.modules[__name__], "transfer" + str(i), (lambda ind: lambda s, ps: getattr(State, "tshare" + str(ind) + "_x")(s) * tax_income(s,ps))(i)) # growth in transfers 


# consumption definitions
for i in range(1,Parameters.N + 1):
    if i == 1:
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: w_x(s,ps) * getattr(Parameters, "l" + str(ind)) + getattr(sys.modules[__name__], "transfer" + str(ind))(s,ps) - getattr(PolicyState, "anext" + str(ind) + "_y")(ps))(i))
    if i >1 and i < (Parameters.N ):
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: (1 + r_x(s,ps)) * getattr(State, "a" + str(ind) + "_x")(s) + w_x(s,ps) * getattr(Parameters, "l" + str(ind)) + getattr(sys.modules[__name__], "transfer" + str(ind))(s,ps) - getattr(PolicyState, "anext" + str(ind) + "_y")(ps))(i))
    if i == (Parameters.N): # using implicitly anext12=0
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: (1 + r_x(s,ps)) * getattr(State, "a" + str(ind) + "_x")(s) + w_x(s,ps) * getattr(Parameters, "l" + str(ind)) + getattr(sys.modules[__name__], "transfer" + str(ind))(s,ps))(i))

def c_all(state, policy_state):
    return tf.reduce_sum(tf.stack([getattr(sys.modules[__name__], "c" + str(i))(state,policy_state) for i in range(1, Parameters.N + 1)], axis=1),axis=1)

# climate definitions
def e_x(state, policy_state):
    """scaled emissions (to match 2015 levels in BAU)"""
    return (1 - mu_y(state,policy_state)) * State.kappa_x(state) * yprod(state)

# actual CO2 emissions in Gt CO2
def Emissions(state, policy_state=None):
    '''Emissions in Gt CO2'''
    return e_x(state, policy_state) * L_total

# next period stock of carbon
def S_next(state, policy_state=None):
    '''Stock of carbon in the next period thousend GtC'''
    return   State.S_x(state) + Emissions(state, policy_state)/3666.0 # emissions are first scaled by population and then converted to thousend GtC


# temperature function
def Temp_x(state, policy_state=None):
    return  lamda * State.S_x(state)

# damage function 
def Omega_x(state, policy_state=None):
    return 1 / (1 + (1/psi1 * Temp_x(state,policy_state))**2 + (Temp_x(state,policy_state)/(2 * State.TP_x(state)))**6.754 )

def Omega_prime_x(state, policy_state):
    a = 1./psi1
    b = 1. / (2. * State.TP_x(state))
    c = 6.754
    tempx = Temp_x(state, policy_state)
    return (-1) * (2*(a**2) * tempx + c * b ** c * tempx ** (c-1)) / (1 + (a * tempx) ** 2 + (b * tempx) ** c) ** 2

# netoutput
def ynet_x(state, policy_state=None):
    capital = State.K_total_x(state)
    return mu_cost(state,policy_state) * State.Omega_x(state) * capital**Parameters.alpha + (1- delta) * capital - tau_x(state, policy_state) * e_x(state, policy_state)  

# Damages as a fraction of total output 
def Damages_GDP(state, policy_state=None):
    return (1 - Omega_x(state, policy_state)) * yprod(state, policy_state) / (yprod(state, policy_state) + (1- delta) *State.K_total_x(state))

# individual utilities
for i in range(1,Parameters.N + 1):
    setattr(sys.modules[__name__], "u" + str(i), (lambda ind: lambda s, ps: ((20*getattr(sys.modules[__name__],"c" + str(ind))(s,ps))**(1-sigma)) / (1 - sigma))(i))

# aggregate utility
def u_all(state, policy_state):
    return tf.reduce_sum(tf.stack([getattr(sys.modules[__name__], "u" + str(i))(state,policy_state) for i in range(1, Parameters.N + 1)], axis=1),axis=1)

