import importlib
import tensorflow as tf
from tensorflow import keras  
import Equilibrium
import Parameters
import gc
import numpy as np
import math  # <<< ADDED: needed for ceil in multi-phase curriculum

if Parameters.horovod:
    import horovod.tensorflow as hvd

Dynamics = importlib.import_module(Parameters.MODEL_NAME + ".Dynamics")
Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

"""
Overview:
    - Each "episode" simulates forward in time, building a dataset of size
      [episode_length, N_sim_batch, #states].
    - We then do N_epochs_per_episode training epochs on that dataset.
"""

# >>> ADDED: helper for improved pre-batch shuffling ------------------------
def build_dataset_prebatch_shuffle(state_episode):
    """
    Flattens episode to 2-D, shuffles individual samples *before* batching,
    then batches and prefetches.  Used when Parameters.use_new_shuffling == True.
    """
    n_states       = len(Parameters.states)
    effective_size = state_episode.shape[0] * state_episode.shape[1]

    if Parameters.sorted_within_batch:
        # keep agent-major order before shuffling
        flat = tf.reshape(tf.transpose(state_episode, [1, 0, 2]),
                          [effective_size, n_states])
    else:
        flat = tf.reshape(state_episode, [effective_size, n_states])

    buffer_size = min(int(effective_size), 10_000)  # cap buffer to save RAM

    return (tf.data.Dataset.from_tensor_slices(flat)
              .shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
              .batch(Parameters.N_minibatch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
# ---------------------------------------------------------------------------

@tf.function
def do_random_step(current_state):
    """One random forward step: calls user-defined dynamics + policy."""
    return Dynamics.total_step_random(current_state, Parameters.policy(current_state))

def run_episode(state_episode):
    """
    Runs forward in time for state_episode.shape[0] steps,
    saving the resulting states in state_episode.
    """
    current_state = state_episode[0, :, :]
    for i in range(1, state_episode.shape[0]):
        current_state = do_random_step(current_state)
        state_episode = tf.tensor_scatter_nd_update(
            state_episode,
            tf.constant([[i]]),
            tf.expand_dims(current_state, axis=0)
        )

    
    return state_episode

@tf.function
def run_grads(state_sample, first_batch):
    """
    Runs a single gradient step on the given mini‐batch state_sample,
    using whichever optimizer is currently in Parameters.optimizer.
    """
    if Parameters.optim_name != 'mao':
        with Parameters.writer.as_default():
            with tf.GradientTape() as tape:
                loss, net_loss = Equilibrium.loss(state_sample, Parameters.policy(state_sample, training=True), training=True)
        if Parameters.horovod:
            tape = hvd.DistributedGradientTape(tape)
        if Parameters.optim_name != 'adahessian':
            grads = tape.gradient(loss, Parameters.policy_net.trainable_variables)
            Parameters.optimizer.apply_gradients(zip(grads, Parameters.policy_net.trainable_variables))
        else:
            grads, Hessian = Parameters.optimizer.get_gradients_hessian(loss, Parameters.policy_net.trainable_weights)
            Parameters.optimizer.apply_gradients_hessian(zip(grads, Hessian, Parameters.policy_net.trainable_weights))

        if first_batch and Parameters.horovod:
            tf.print("Broadcasting variables....")
            # --- Keras 3 / TF 2.20 compatible retrieval of optimizer variables
            opt_vars_attr = getattr(Parameters.optimizer, "variables", None)
            if callable(opt_vars_attr):
                opt_vars = Parameters.optimizer.variables()
            elif opt_vars_attr is not None:
                opt_vars = opt_vars_attr
            else:
                # some builds expose `.weights`
                opt_vars = getattr(Parameters.optimizer, "weights", [])
            hvd.broadcast_variables(Parameters.policy_net.variables, root_rank=0)
            hvd.broadcast_variables(opt_vars, root_rank=0)

        return loss, net_loss

    else:
        # If using MAO (multi‐task) optimizer
        with Parameters.writer.as_default():
            with tf.GradientTape(persistent=True) as tape:
                total_loss, net_loss, task_losses = Equilibrium.loss_mao(state_sample, Parameters.policy(state_sample, training=True), training=True)
        variables = Parameters.policy_net.trainable_variables
        task_grads = []
        for loss_component in task_losses:
            grads = tape.gradient(loss_component, variables)
            task_grads.append(grads)
        del tape
        Parameters.optimizer.apply_mao_gradients(task_grads)
        return total_loss, net_loss

def run_epoch(state_episode):
    """
    One epoch of training: we break the data in state_episode (which is shape
    [episode_length, N_sim_batch, #states]) into mini‐batches and run run_grads.
    """
    effective_size = state_episode.shape[0] * state_episode.shape[1]
    # Flatten from [ep_length, batch_size, #states] → [ep_length * batch_size, #states]
    reshaped = tf.reshape(state_episode, [effective_size, len(Parameters.states)])

    # >>> ADDED: choose new or legacy shuffling based on cfg flag -----------
    if getattr(Parameters, "use_new_shuffling", False):
        dataset = build_dataset_prebatch_shuffle(state_episode)
    else:
        # -------------------------------------------------------------------
        # legacy dataset construction (unchanged)
        # -------------------------------------------------------------------
        if not Parameters.sorted_within_batch:
            dataset = tf.data.Dataset.from_tensor_slices(reshaped)\
                .shuffle(buffer_size=effective_size)\
                .batch(Parameters.N_minibatch_size, drop_remainder=True)
        else:
            # If user wants sorted batches, we often do a different permutation
            transposed = tf.transpose(state_episode, [1, 0, 2])
            reshaped_sorted = tf.reshape(transposed, [effective_size, len(Parameters.states)])
            dataset = tf.data.Dataset.from_tensor_slices(reshaped_sorted)\
                .batch(Parameters.N_minibatch_size, drop_remainder=True)\
                .shuffle(buffer_size=int(effective_size / Parameters.N_minibatch_size))
    # -----------------------------------------------------------------------

    epoch_loss = 0.0
    net_epoch_loss = 0.0

    # Possibly adjust weights if first iteration
    if Parameters.use_weight_adjustment and Parameters.optimizer.iterations == 0:
        from unit_activation_reinitializer import adjust_weight_init
        print("Adjusting weight initialization...")
        adjust_weight_init(
            Parameters.policy_net,
            dataset,
            batch_size=Parameters.N_minibatch_size,
            tol=0.2,
            max_iters=10,
            exclude_layers=None,
        )

    for batch in dataset:
        # Check if this is the very first batch for a Horovod worker
        first_batch = (
            Parameters.horovod
            and (Parameters.optimizer.iterations == Parameters.optimizer_starting_iteration)
        )
        epoch_loss_1, net_epoch_loss_1 = run_grads(batch, first_batch)
        epoch_loss += epoch_loss_1
        net_epoch_loss += net_epoch_loss_1

    return epoch_loss, net_epoch_loss

def run_cycle(state_episode):
    """
    Runs a single episode:
      1) simulates forward to fill state_episode
      2) does N_epochs_per_episode training passes
      3) logs the metrics
    """
    state_episode = run_episode(state_episode)

    file1 = open(Parameters.LOG_DIR + "/" + Parameters.error_filename, "a")

    for e in range(Parameters.N_epochs_per_episode):
        epoch_loss, net_epoch_loss = run_epoch(state_episode)

        # normalize by the actual ep length times the batch
        ep_len = state_episode.shape[0]
        denom  = ep_len * Parameters.N_sim_batch

        MSE_epoch_loss = epoch_loss / denom
        Norm_epoch_loss = tf.math.sqrt(epoch_loss / denom)
        MSE_epoch_no_penalty = net_epoch_loss / denom
        Norm_epoch_loss_no_penalty = tf.math.sqrt(net_epoch_loss / denom)

        tf.print("----------------------------------")
        tf.print("Normalized MSE epoch loss: " + str(MSE_epoch_loss))
        tf.print("Normalized epoch loss: " + str(Norm_epoch_loss))
        tf.print("----------------------------------")
        tf.print("Normalized MSE epoch loss without penalties: " + str(MSE_epoch_no_penalty))
        tf.print("Normalized epoch loss without penalties: " + str(Norm_epoch_loss_no_penalty))
        tf.print("==================================")

        file1.write("MSE:            " + str(tf.get_static_value(MSE_epoch_loss)) + "  ")
        file1.write("MAE:            " + str(tf.get_static_value(Norm_epoch_loss)) + "  ")
        file1.write("MSE_no_penalty: " + str(tf.get_static_value(MSE_epoch_no_penalty)) + "  ")
        file1.write("MAE_no_penatly: " + str(tf.get_static_value(Norm_epoch_loss_no_penalty)) + "\n")

        if Parameters.use_loss_reweighting:
            Parameters.loss_reweighter.compute_epoch_weights()

    file1.close()

    # Possibly print re-weighting info
    episode_now = Parameters.ckpt.current_episode.numpy()
    freq = Parameters.print_weights_every_n_episodes
    if Parameters.use_loss_reweighting and freq > 0 and (episode_now % freq) == 0:
        file2 = open(Parameters.LOG_DIR + "/" + Parameters.error_filename, "a")
        dummy_losses = Equations.equations(state_episode[0, :, :], Parameters.policy(state_episode[0, :, :]))
        eq_f_list = list(dummy_losses.keys())
        tf.print(f"*** Weights at Episode {episode_now}: ***")
        file2.write(f"*** Weights at Episode {episode_now}: ***\n")
        weights = Parameters.loss_reweighter.current_weights
        for i, eq_f in enumerate(eq_f_list):
            wval = weights[i].numpy() if weights is not None else 0.0
            tf.print(f"   task {i+1} (weight {i+1}): {wval}")
            file2.write(f"   task {i+1} (weight {i+1}): {wval}\n")
        file2.close()

    return state_episode

def run_cycles():
    """
    Main training loop over N_episodes, each time incrementing the
    episode_length from N_episode_length_min by N_increment until hitting
    N_episode_length, or continuing at the max.
    """
    if "post_init" in dir(Hooks) and Parameters.ckpt.current_episode < 2:
        print("Running post-init hook...")
        Hooks.post_init()
        print("Starting state after post-init:")
        print(Parameters.starting_state)

        # >>> add the quick sanity check here <<<
        # NOTE: if you keep this block, ensure required symbols are imported.
        # init_scc = Definitions.scc(starting_state, policy(starting_state))
        # tf.print("Initial SCC (first path):", init_scc[0])
        # quit()
# --------------------------------------------------------

    start_time = tf.timestamp()

    # If using a "warm start" to NGD, track that
    use_warm_start = hasattr(Parameters, 'warm_start_in_use') and Parameters.warm_start_in_use
    if use_warm_start:
        warm_n_episodes = getattr(Parameters, 'warm_start_n_episodes', 5)

    for _ in range(Parameters.N_episodes):
        episode_now = Parameters.ckpt.current_episode.numpy()

        # Possibly switch from warm-start to final NGD
        if use_warm_start and episode_now > warm_n_episodes and Parameters.warm_start_in_use:
            print(f"Switching from warm-start optimizer to final {Parameters.final_optim_name} at episode {episode_now}...")
            Parameters.optimizer = Parameters.final_optimizer_ngd
            Parameters.optim_name = Parameters.final_optim_name
            setattr(Parameters, "warm_start_in_use", False)

        # -------------------------------------------------------------------
        # Compute current episode length (NEW: optional multi-phase curriculum)
        # -------------------------------------------------------------------
        bp = getattr(Parameters, "curriculum_breakpoints", None)
        st = getattr(Parameters, "curriculum_steps", None)

        if bp is not None and st is not None and len(bp) == len(st):
            # Multi-phase schedule: start at min, advance across phases in fixed-size steps
            L = int(Parameters.N_episode_length_min)
            remaining_eps = int(episode_now - 1)
            cap = int(Parameters.N_episode_length)
            for target, step in zip(bp, st):
                target = int(min(target, cap))
                if target <= L:
                    continue
                need = int(math.ceil((target - L) / float(step)))
                take = min(remaining_eps, need)
                L += take * int(step)
                remaining_eps -= take
                if remaining_eps <= 0:
                    break
            episode_length_current = int(min(L, cap))
        else:
            # Legacy uniform schedule (unchanged)
            episode_length_current = Parameters.N_episode_length_min + (episode_now - 1) * Parameters.N_increment_steps
            episode_length_current = min(episode_length_current, Parameters.N_episode_length)
            episode_length_current = int(episode_length_current)

        tf.print("Running episode:", episode_now,
                 "; with episode_length =", episode_length_current)

        # Build an [episode_length_current, N_sim_batch, #states] tensor
        state_episode = tf.tile(
            tf.expand_dims(Parameters.starting_state, axis=0),
            [episode_length_current, 1, 1]
        )

        # Run the simulation + training
        state_episode = run_cycle(state_episode)

        # Decide how to handle next ep's starting state
        if Parameters.initialize_each_episode:
            print("Running with states re-drawn after each episode!")
            # e.g.:
            # Parameters.starting_state.assign(Parameters.initialize_states())
            # Hooks.post_init()
        else:
            # By default, we continue from the last state
            Parameters.starting_state.assign(state_episode[-1, :, :])

        # Advance the episode counter
        Parameters.ckpt.current_episode.assign_add(1)

        # Save and run final hooks
        if not Parameters.horovod_worker:
            Parameters.manager.save()
            with Parameters.writer.as_default():
                Hooks.cycle_hook(
                    state_episode[0, :, :],
                    int(Parameters.ckpt.current_episode.numpy())
                )

        tf.print("Elapsed time since start: ", tf.timestamp() - start_time)

        # Occasional garbage collection
        if episode_now % 50 == 0:
            tf.print("Garbage collecting")
            tf.keras.backend.clear_session()
            gc.collect()
