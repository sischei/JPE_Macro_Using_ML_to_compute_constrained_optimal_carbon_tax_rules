import Parameters
import PolicyState
import State
import sys
import tensorflow as tf
from Parameters import  alpha, beta, delta , sigma, lamda, psi1, vartheta, L_total, N, Tstep, rho0, rhoinfty, deltarho


def K_total_x(state, policy_state=None):
    return tf.reduce_sum(tf.stack([(getattr(State,"a" + str(i) + "_x")(state)) for i in range(1, Parameters.N + 1)],axis=1),axis=1)

# next period capital used to update K_total_x
def K_total_next(state, policy_state):
    return tf.reduce_sum(tf.stack([(getattr(PolicyState,"anext" + str(i) + "_y")(policy_state)) for i in range(1, Parameters.N)],axis=1),axis=1)

def r_x(state, policy_state=None):
    return State.Omega_x(state) * alpha * (State.K_total_x(state))**(alpha - 1)  - delta

def w_x(state, policy_state=None):
    return State.Omega_x(state) * (1 - alpha) * (State.K_total_x(state))**alpha

def yprod(state, policy_state=None):
    '''Production function'''
    return  K_total_x(state)**Parameters.alpha 
# --------------------------------------------------------------------------- #
# Real and computational time periods
# --------------------------------------------------------------------------- #
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

def rho_t(state, policy_state):
    """ World population [million] """
    _t = tcomp2t(state, policy_state)
    _rho = rho0 + (rhoinfty - rho0) * (1 - tf.math.exp(-Tstep * deltarho * _t))
    return _rho

# consumption definitions
for i in range(1,Parameters.N + 1):
    if i == 1:
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: w_x(s,ps) * getattr(Parameters, "l" + str(ind)) - getattr(PolicyState, "anext" + str(ind) + "_y")(ps))(i))
    if i >1 and i < (Parameters.N ):
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: (1 + r_x(s,ps)) * getattr(State, "a" + str(ind) + "_x")(s) + w_x(s,ps) * getattr(Parameters, "l" + str(ind)) - getattr(PolicyState, "anext" + str(ind) + "_y")(ps))(i))
    if i == (Parameters.N): # using implicitly aN=0
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: (1 + r_x(s,ps)) * getattr(State, "a" + str(ind) + "_x")(s) + w_x(s,ps) * getattr(Parameters, "l" + str(ind)))(i) ) 

def c_all(state, policy_state):
    return tf.reduce_sum(tf.stack([getattr(sys.modules[__name__], "c" + str(i))(state,policy_state) for i in range(1, Parameters.N + 1)], axis=1),axis=1)

# climate definitions
def e_x(state, policy_state=None):
    return State.kappa_x(state) * yprod(state) 

def Emissions(state, policy_state=None):
    '''Emissions in Gt CO2'''
    return e_x(state) * L_total

def S_next(state, policy_state=None):
    '''Stock of carbon in the next period thousend GtC'''
    return   State.S_x(state) + Emissions(state, policy_state)/3666.0 # emissions are first scaled by population and then converted to thousend GtC

# temperature function
def Temp_x(state, policy_state=None):
    return  lamda * State.S_x(state)

# Damage function
def Omega_x(state, policy_state=None):
    return 1 / (1 + (1/psi1 * Temp_x(state,policy_state))**2 + (Temp_x(state,policy_state)/(2 * State.TP_x(state)))**6.754 ) #* 0.98


# netoutput
def ynet_x(state, policy_state=None):
    capital = State.K_total_x(state)
    return Omega_x(state, policy_state) * capital**alpha + (1- delta) * capital

# Damages as a fraction of total output
def Damages_GDP(state, policy_state=None):
    return (1 - Omega_x(state, policy_state)) * yprod(state, policy_state) / (yprod(state, policy_state) + (1- delta) *State.K_total_x(state))

# individual utilities
for i in range(1,Parameters.N + 1):
    setattr(sys.modules[__name__], "u" + str(i), (lambda ind: lambda s, ps: ((20.0*getattr(sys.modules[__name__],"c" + str(ind))(s,ps))**(1-sigma)) / (1 - sigma))(i))

# aggregate utility
def u_all(state, policy_state):
    return tf.reduce_sum(tf.stack([getattr(sys.modules[__name__], "u" + str(i))(state,policy_state) for i in range(1, Parameters.N + 1)], axis=1),axis=1)

