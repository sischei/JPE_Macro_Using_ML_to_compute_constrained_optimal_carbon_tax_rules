
# EXTREME
def post_init():
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "tcomp_x", tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) # time 0
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_x",tf.constant(3.,shape=(Parameters.starting_state.shape[0],)))) # initial TP
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "TP_reached",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) # TP not reached "boolean"
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "kappa_x",tf.constant(0.35032,shape=(Parameters.starting_state.shape[0],)))) # initial productivity
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a1_x",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a2_x",tf.constant(0.0206,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.0208
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a3_x",tf.constant(0.0442,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.0446
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a4_x",tf.constant(0.0701,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.0707
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a5_x",tf.constant(0.0978,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.0986
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a6_x",tf.constant(0.1267,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.1278
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a7_x",tf.constant(0.1564,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.1577
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a8_x",tf.constant(0.1867,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.188
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a9_x",tf.constant(0.217,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.219
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a10_x",tf.constant(0.146,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.147
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a11_x",tf.constant(0.0871,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.0878
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a12_x",tf.constant(0.039,shape=(Parameters.starting_state.shape[0],)))) # initial capital lamda=1.5: 0.0394
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "K_total_x",Definitions.K_total_x(Parameters.starting_state))) # initial total capital
    
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "S_x",tf.constant(0.851,shape=(Parameters.starting_state.shape[0],)))) # initial carbon stock
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Temp_x", Definitions.Temp_x(Parameters.starting_state))) # initial temperature
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "e_x",Definitions.e_x(Parameters.starting_state))) # initial output
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "Omega_x", Definitions.Omega_x(Parameters.starting_state))) # initial damage multiplier
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "ynet_x", Definitions.ynet_x(Parameters.starting_state))) # initial net output
    # Parameters.starting_state.assign(State.update(Parameters.starting_state, "r_x",Definitions.r_x(Parameters.starting_state))) # initial interest rate
    # Parameters.starting_state.assign(State.update(Parameters.starting_state, "w_x",Definitions.w_x(Parameters.starting_state))) # initial wage


# MILD
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a1_x",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a2_x",tf.constant(0.0208,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a3_x",tf.constant(0.0446,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a4_x",tf.constant(0.0707,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a5_x",tf.constant(0.0986,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a6_x",tf.constant(0.1278,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a7_x",tf.constant(0.1577,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a8_x",tf.constant(0.188,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a9_x",tf.constant(0.219,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a10_x",tf.constant(0.147,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a11_x",tf.constant(0.0878,shape=(Parameters.starting_state.shape[0],)))) # initial capital
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "a12_x",tf.constant(0.0394,shape=(Parameters.starting_state.shape[0],)))) # initial capital