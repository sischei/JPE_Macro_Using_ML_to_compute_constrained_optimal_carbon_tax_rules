import Definitions
import State
from Parameters import beta, N, sigma, theta2
import Parameters
import PolicyState
import tensorflow as tf



def equations(state, policy_state, training=None):
     current_episode = tf.cast(Parameters.ckpt.current_episode, tf.int32)
     E_t = State.E_t_gen(state, policy_state, training=training)
    
     loss_dict = {}
    
     # Euler equations
     for i in range(1,N):
          loss_dict['ee_' + str(i)] = -1+  (beta * E_t(lambda s, ps: (1+Definitions.r_x(s,ps)) * ((getattr(Definitions,"c" + str(i+1))(s,ps)) ** (-sigma))))** (-1 / sigma) / (getattr(Definitions,"c" + str(i))(state,policy_state)) 
     
     
     # budget constraints 
     for i in range(1,N+1):
          if i < N:
               loss_dict['bc_' + str(i)] = (1 + Definitions.r_x(state,policy_state)) * getattr(State,"a" + str(i) + "_x")(state) + getattr(Parameters,"l" + str(i)) * Definitions.w_x(state,policy_state) + getattr(Definitions, "transfer"+ str(i))(state,policy_state) - getattr(Definitions,"c" + str(i))(state,policy_state) - getattr(PolicyState,"anext" + str(i) + "_y")(policy_state)
          elif i == N:
               loss_dict['bc_' + str(i)] = (1 + Definitions.r_x(state,policy_state)) * getattr(State,"a" + str(i) + "_x")(state) + getattr(Parameters,"l" + str(i)) * Definitions.w_x(state,policy_state) + getattr(Definitions, "transfer"+ str(i))(state,policy_state) - getattr(Definitions,"c" + str(i))(state,policy_state) 
     
    # Value functions (introduced once euler error is stable)
     for i in range(1, N):
          if current_episode > 10:
               if i < (N-1):
                    loss_val = (getattr(Definitions,"u" + str(i))(state,policy_state) + beta * E_t(lambda s, ps: getattr(PolicyState,"v" + str(i+1) + "_y")(ps)))/  getattr(PolicyState,"v" + str(i) + "_y")(policy_state) - 1
               else: # i == (N-1)
                    loss_val = (getattr(Definitions,"u" + str(i))(state,policy_state) + beta * E_t(lambda s, ps: getattr(Definitions,"u" + str(i+1))(s,ps)))/ getattr(PolicyState,"v" + str(i) + "_y")(policy_state) - 1
          else:
               # initially zero
               loss_val = tf.zeros_like(getattr(Definitions,"u" + str(i))(state,policy_state))
          loss_dict['v_' + str(i)] = loss_val

     
     return loss_dict
