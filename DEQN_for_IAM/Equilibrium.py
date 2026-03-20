# Equilibrium.py — updated to correctly include policy pathway in derivative penalty
# --------------------------------------------------------------------------------

import importlib
import tensorflow as tf
import PolicyState
import Definitions

from Parameters import (
    definition_bounds_hard, horovod_worker, MODEL_NAME, optimizer,
    policy_bounds_hard, loss_choice, global_default_dtype,
    use_loss_reweighting
)

# Optional loss reweighting
if use_loss_reweighting:
    from Parameters import loss_reweighter

# Access to runtime flags / schedules
import Parameters

Equations = importlib.import_module(MODEL_NAME + ".Equations")

def penalty_bounds_policy(state, policy_state):
    """Quadratic penalties that measure how much RAW variables are clipped by bounds."""
    res = tf.constant(0.0, dtype=global_default_dtype)

    # Policy lower bounds
    for bound_vars in policy_bounds_hard['lower'].keys():
        raw_vs_bounded = (
            getattr(PolicyState, bound_vars)(policy_state)
            - getattr(PolicyState, bound_vars + "_RAW")(policy_state)
        )
        penalty = tf.math.reduce_sum(
            policy_bounds_hard['penalty_lower'][bound_vars] * (raw_vs_bounded ** 2)
        )
        tf.summary.scalar('penalty_lower_policy_' + bound_vars, penalty)
        res += penalty

    # Policy upper bounds
    for bound_vars in policy_bounds_hard['upper'].keys():
        raw_vs_bounded = (
            getattr(PolicyState, bound_vars + "_RAW")(policy_state)
            - getattr(PolicyState, bound_vars)(policy_state)
        )
        penalty = tf.math.reduce_sum(
            policy_bounds_hard['penalty_upper'][bound_vars] * (raw_vs_bounded ** 2)
        )
        if not horovod_worker:
            tf.summary.scalar('penalty_upper_policy_' + bound_vars, penalty)
        res += penalty

    # Definition lower bounds
    for bound_vars in definition_bounds_hard['lower'].keys():
        raw_vs_bounded = (
            getattr(Definitions, bound_vars)(state, policy_state)
            - getattr(Definitions, bound_vars + "_RAW")(state, policy_state)
        )
        penalty = tf.math.reduce_sum(
            definition_bounds_hard['penalty_lower'][bound_vars] * (raw_vs_bounded ** 2)
        )
        if not horovod_worker:
            tf.summary.scalar('penalty_lower_def_' + bound_vars, penalty)
        res += penalty

    # Definition upper bounds
    for bound_vars in definition_bounds_hard['upper'].keys():
        raw_vs_bounded = (
            getattr(Definitions, bound_vars + "_RAW")(state, policy_state)
            - getattr(Definitions, bound_vars)(state, policy_state)
        )
        penalty = tf.math.reduce_sum(
            definition_bounds_hard['penalty_upper'][bound_vars] * (raw_vs_bounded ** 2)
        )
        if not horovod_worker:
            tf.summary.scalar('penalty_upper_def_' + bound_vars, penalty)
        res += penalty

    return res

# Huber loss
Huber_loss_delta = 1.0
def Huber_loss(delta_y, delta):
    abs_ = tf.math.abs(delta_y)
    return tf.where(
        tf.less_equal(abs_, delta),
        0.5 * delta_y ** 2,
        delta * (abs_ - 0.5 * delta)
    )

def _current_lambda():
    _lambda_sched = getattr(Parameters, "foc_derivative_lambda_schedule", None)
    if callable(_lambda_sched):
        return tf.cast(_lambda_sched(optimizer.iterations), global_default_dtype)
    return tf.cast(getattr(Parameters, "foc_derivative_lambda", 0.0), global_default_dtype)

# [IFT] Separate schedule for the IFT tangent penalty
def _current_ift_lambda():
    _lambda_sched = getattr(Parameters, "ift_tangent_lambda_schedule", None)
    if callable(_lambda_sched):
        return tf.cast(_lambda_sched(optimizer.iterations), global_default_dtype)
    return tf.cast(getattr(Parameters, "ift_tangent_lambda", 0.0), global_default_dtype)

def _maybe_log_scalar(name, value):
    if not horovod_worker:
        tf.summary.scalar(name, value)

# [IFT] Utility: random normalized direction in R^n (kept constant across batch)
def _rand_unit_direction(n, dtype):
    v = tf.random.normal([n], dtype=dtype)
    nrm = tf.norm(v) + tf.cast(1e-12, dtype)
    return v / nrm

# [IFT] Utility: generate a dictionary of random probes with shapes matching eq residuals
def _sample_v_probes(losses, num_probes, dtype):
    probes = {}
    for eq_name, eq_res in losses.items():
        flat_shape = tf.concat([tf.shape(eq_res)[:1], [-1]], axis=0)  # [B, M_e]
        m = tf.shape(tf.reshape(eq_res, flat_shape))[1]
        v = tf.random.normal([m], dtype=dtype)
        v /= (tf.norm(v) + tf.cast(1e-12, dtype))
        probes[eq_name] = [tf.identity(v) for _ in range(num_probes)]
    return probes

def _flatten_concat_residuals(losses):
    """[IFT-COMPACT] Flatten all equation residuals to [B, -1] and concat → [B, Mtot]."""
    flats = []
    for eq_res in losses.values():
        flats.append(tf.reshape(eq_res, tf.concat([tf.shape(eq_res)[:1], [-1]], axis=0)))
    return tf.concat(flats, axis=1)  # [B, Mtot]

def loss(state, _policy_state_ignored, training=None):
    """
    Total loss = base residual loss + gated derivative penalty (+ bounds),
    normalized by the number of equations.

    IMPORTANT FIX:
      `policy_state` is recomputed INSIDE the inner GradientTape so that
      ∂(residual)/∂(state) includes the dependency through the policy network.
    """
    tf.summary.experimental.set_step(optimizer.iterations)

    use_deriv   = getattr(Parameters, "use_foc_derivative_loss", False)
    sens_list   = list(getattr(Parameters, "sensitivity_parameters", []))
    tau         = getattr(Parameters, "foc_derivative_gate_tau", 1e10)
    kappa       = getattr(Parameters, "foc_derivative_gate_kappa", 100.0)
    gate_mode   = getattr(Parameters, "foc_derivative_gate_mode", "per_equation")  # "per_equation" | "batch_rmse"
    comp_deriv  = getattr(Parameters, "deriv_compensate_no_eq", False)
    comp_bounds = getattr(Parameters, "bounds_compensate_no_eq", False)

    # [IFT] flags
    use_ift     = getattr(Parameters, "use_ift_tangent_loss", False)
    ift_v       = int(getattr(Parameters, "ift_num_v_probes", 2))
    ift_u       = int(getattr(Parameters, "ift_num_theta_dirs", 1))
    ift_gate    = getattr(Parameters, "ift_gate_mode", gate_mode)
    ift_tau     = tf.cast(getattr(Parameters, "ift_gate_tau", tau), global_default_dtype)
    ift_kappa   = tf.cast(getattr(Parameters, "ift_gate_kappa", kappa), global_default_dtype)
    # [IFT-AGG] aggregate & compact toggles
    ift_aggregate = bool(getattr(Parameters, "ift_aggregate_probes", True))
    ift_compact   = bool(getattr(Parameters, "ift_compact_mode", True))

    # ### >>> ADDED: Variance Penalty Parameters
    lambda_var  = getattr(Parameters, "lambda_variance", 0.0) # Default to 0 (off)

    loss_val     = tf.constant(0.0, dtype=global_default_dtype)
    net_loss_val = tf.constant(0.0, dtype=global_default_dtype)

    # Inner tape: watch state and recompute policy inside
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(state)
        policy_state = Parameters.policy(state)
        losses = Equations.equations(state, policy_state, training=training)

        base_losses = []
        # ### >>> ADDED: Accumulator for per-sample errors
        # We need a vector of shape [Batch_Size] to track total error per particle
        batch_size = tf.shape(state)[0]
        per_sample_accumulator = tf.zeros([batch_size], dtype=global_default_dtype)
        # ### <<<
        for eq_f, eq_res in losses.items():
            if loss_choice == 'huber':
                eq_loss = tf.math.reduce_sum(Huber_loss(eq_res, Huber_loss_delta))
            else:
                eq_loss = tf.math.reduce_sum(eq_res ** 2)
            _maybe_log_scalar('dev_' + eq_f, eq_loss)

            if use_loss_reweighting and loss_reweighter.current_weights is not None:
                i = len(base_losses)
                eq_loss = eq_loss * loss_reweighting_weight(loss_reweighter, i)

            base_losses.append(eq_loss)

            # ### >>> ADDED: 
            # Sum up erros (duplicate)
            per_sample_accumulator += eq_res ** 2
            # ### <<<

        base_sum = tf.add_n(base_losses) if base_losses else tf.constant(0.0, dtype=global_default_dtype)

        # ### >>> ADDED: Calculate Variance and Add to Total Loss
        if lambda_var > 0.0:
            # Calculate variance
            variance_term = tf.math.reduce_variance(base_sum)
            
            # Scale by lambda and add
            loss_val += lambda_var * variance_term
            
            # Optional: Log it to TensorBoard
            _maybe_log_scalar('loss_variance', lambda_var * variance_term)
        # ### <<<
        if gate_mode == "batch_rmse":
            B = tf.cast(tf.shape(state)[0], global_default_dtype)
            no_eq_local = tf.cast(len(losses), global_default_dtype)
            rmse_no_penalty = tf.sqrt(base_sum / tf.maximum(B * no_eq_local, 1.0))
        else:
            rmse_no_penalty = None

        # [IFT-COMPACT] build concatenated residuals for single VJP if needed
        if use_ift and ift_compact:
            R_flat = _flatten_concat_residuals(losses)  # [B, Mtot]

    if use_loss_reweighting:
        loss_reweighter.update_weights(losses)

    loss_val     += base_sum
    net_loss_val += base_sum

    # Derivative penalty (gated) — unchanged
    if use_deriv and len(sens_list) > 0:
        param_to_idx = {p: Parameters.states.index(p) for p in sens_list if p in Parameters.states}
        if len(param_to_idx) > 0:
            lambda_ = _current_lambda()
            _maybe_log_scalar('lambda/current', lambda_)

            deriv_loss = tf.constant(0.0, dtype=global_default_dtype)
            for j, (eq_name, eq_res) in enumerate(losses.items()):
                grads_all_states = tape.gradient(
                    eq_res, state, unconnected_gradients=tf.UnconnectedGradients.ZERO
                )

                if gate_mode == "batch_rmse" and rmse_no_penalty is not None:
                    gate = tf.nn.sigmoid(kappa * (tf.cast(tau, global_default_dtype) - rmse_no_penalty))
                    param_term = tf.constant(0.0, dtype=global_default_dtype)
                    for pidx in param_to_idx.values():
                        g = grads_all_states[:, pidx]
                        param_term += tf.reduce_sum(g * g) * gate
                else:
                    rank = tf.rank(eq_res)
                    gate_row = tf.cond(
                        tf.greater(rank, 1),
                        true_fn=lambda: tf.reduce_mean(tf.math.abs(eq_res), axis=tf.range(1, rank)),
                        false_fn=lambda: tf.math.abs(eq_res)
                    )
                    gate = tf.nn.sigmoid(kappa * (tf.cast(tau, global_default_dtype) - gate_row))
                    param_term = tf.constant(0.0, dtype=global_default_dtype)
                    for pidx in param_to_idx.values():
                        g = grads_all_states[:, pidx]
                        param_term += tf.reduce_sum(gate * (g * g))

                if use_loss_reweighting and loss_reweighter.current_weights is not None:
                    param_term = param_term * loss_reweighting_weight(loss_reweighter, j)

                deriv_loss += param_term

            weighted_deriv = lambda_ * deriv_loss
            if comp_deriv:
                weighted_deriv *= tf.cast(len(losses), global_default_dtype)

            _maybe_log_scalar('loss_derivative_penalty', weighted_deriv)
            ratio = weighted_deriv / (net_loss_val + tf.constant(1e-12, dtype=global_default_dtype))
            _maybe_log_scalar('metric/deriv_to_base_ratio', ratio)

            loss_val += weighted_deriv
        else:
            _maybe_log_scalar('loss_derivative_penalty', 0.0)
    else:
        _maybe_log_scalar('loss_derivative_penalty', 0.0)

    # [IFT] IFT-consistent tangent penalty (safe VJP form)
    if use_ift and len(sens_list) > 0:
        param_to_idx = [Parameters.states.index(p) for p in sens_list if p in Parameters.states]
        if len(param_to_idx) > 0:
            lambda_ift = _current_ift_lambda()
            _maybe_log_scalar('lambda_ift/current', lambda_ift)

            # Short-circuit: if λ == 0, skip heavy path
            if tf.equal(lambda_ift, tf.constant(0.0, dtype=global_default_dtype)):
                _maybe_log_scalar('loss_ift_tangent_vjp', 0.0)
            else:
                B = tf.shape(state)[0]
                S = tf.shape(state)[1]

                # Batch/global gate (used in compact mode; also available in aggregate mode)
                if ift_gate == "batch_rmse" and rmse_no_penalty is not None:
                    gate_global = tf.nn.sigmoid(ift_kappa * (ift_tau - rmse_no_penalty))
                    _maybe_log_scalar('ift/rmse_no_penalty', rmse_no_penalty)
                    _maybe_log_scalar('ift/gate_global', gate_global)
                else:
                    gate_global = None

                ift_loss_accum = tf.constant(0.0, dtype=global_default_dtype)

                if ift_compact:
                    # --------- COMPACT MODE: single residual tensor, one VJP per probe ----------
                    Mtot = tf.shape(R_flat)[1]

                    for _ in range(ift_u):
                        # θ-direction
                        u_param = _rand_unit_direction(len(param_to_idx), global_default_dtype)
                        u_full = tf.zeros([S], dtype=global_default_dtype)
                        u_full = tf.tensor_scatter_nd_update(
                            u_full,
                            indices=tf.reshape(tf.constant(param_to_idx, dtype=tf.int32), [-1, 1]),
                            updates=u_param
                        )
                        u_full_batched = tf.tile(tf.reshape(u_full, [1, -1]), [B, 1])

                        for _k in range(ift_v):
                            v = tf.random.normal([Mtot], dtype=global_default_dtype)
                            v /= (tf.norm(v) + tf.cast(1e-12, global_default_dtype))
                            v_batched = tf.tile(tf.reshape(v, [1, -1]), [B, 1])  # [B, Mtot]

                            # Single VJP
                            grad_vjp_state = tape.gradient(
                                R_flat, state,
                                output_gradients=v_batched,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO
                            )  # [B, S]

                            rv = tf.reduce_sum(tf.reduce_sum(grad_vjp_state * u_full_batched, axis=1))
                            rv2 = rv * rv
                            gated_rv2 = (gate_global * rv2) if gate_global is not None else rv2
                            ift_loss_accum += gated_rv2

                else:
                    # --------- AGGREGATED MODE: VJPs per probe over the eq list ----------
                    v_probes = _sample_v_probes(losses, num_probes=ift_v, dtype=global_default_dtype)
                    ys = [eq_res for _, eq_res in losses.items()]

                    for _ in range(ift_u):
                        u_param = _rand_unit_direction(len(param_to_idx), global_default_dtype)
                        u_full = tf.zeros([S], dtype=global_default_dtype)
                        u_full = tf.tensor_scatter_nd_update(
                            u_full,
                            indices=tf.reshape(tf.constant(param_to_idx, dtype=tf.int32), [-1, 1]),
                            updates=u_param
                        )
                        u_full_batched = tf.tile(tf.reshape(u_full, [1, -1]), [B, 1])

                        for k in range(ift_v):
                            out_grads = []
                            for (eq_name, eq_res) in losses.items():
                                v = v_probes[eq_name][k]
                                v_batched = tf.tile(tf.reshape(v, [1, -1]), [B, 1])
                                v_shaped  = tf.reshape(v_batched, tf.shape(eq_res))
                                out_grads.append(v_shaped)

                            grad_vjp_state = tape.gradient(
                                ys, state,
                                output_gradients=out_grads,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO
                            )
                            rv = tf.reduce_sum(tf.reduce_sum(grad_vjp_state * u_full_batched, axis=1))
                            rv2 = rv * rv
                            gated_rv2 = (gate_global * rv2) if gate_global is not None else rv2
                            ift_loss_accum += gated_rv2

                weighted_ift = lambda_ift * ift_loss_accum
                _maybe_log_scalar('loss_ift_tangent_vjp', weighted_ift)
                ratio_ift = weighted_ift / (net_loss_val + tf.constant(1e-12, dtype=global_default_dtype))
                _maybe_log_scalar('metric/ift_to_base_ratio', ratio_ift)

                loss_val += weighted_ift
        else:
            _maybe_log_scalar('loss_ift_tangent_vjp', 0.0)
    else:
        _maybe_log_scalar('loss_ift_tangent_vjp', 0.0)

    # One-shot diagnostic
    if (optimizer.iterations == 1) and (len(sens_list) > 0):
        s_res = tf.add_n([tf.reduce_sum(v) for v in losses.values()])
        g_res = tape.gradient(s_res, state, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        per_theta_l2 = []
        for pname in sens_list:
            if pname in Parameters.states:
                idx = Parameters.states.index(pname)
                l2 = tf.sqrt(tf.reduce_sum(tf.square(g_res[:, idx])) + 1e-30)
                per_theta_l2.append((pname, l2))
                _maybe_log_scalar(f'diag/grad_theta_l2/{pname}', l2)
        total_l2 = (
            tf.sqrt(tf.add_n([l for _, l in per_theta_l2]) + 1e-30)
            if per_theta_l2 else tf.constant(0.0, dtype=global_default_dtype)
        )
        _maybe_log_scalar('diag/grad_theta_l2_total', total_l2)
        tf.print("[One-shot] ∥∂(Σres)/∂θ∥₂ =", total_l2,
                 " | per-θ:", {k: v for k, v in per_theta_l2})

    del tape

    # Bounds penalty
    bounds_term = penalty_bounds_policy(state, policy_state)
    if comp_bounds:
        bounds_term *= tf.cast(len(losses), global_default_dtype)
    loss_val += bounds_term

    # Normalize
    no_eq = len(losses)
    _maybe_log_scalar('dev_loss',     loss_val     / no_eq)
    _maybe_log_scalar('dev_net_loss', net_loss_val / no_eq)
    return loss_val / no_eq, net_loss_val / no_eq


def loss_mao(state, _policy_state_ignored, training=None):
    """
    Multi-task variant for MAO:
      - Recompute policy inside inner tape (same critical fix).
      - Derivative penalty added as its own task.
      - Bounds penalty appended as its own task.

      [IFT] The IFT-consistent tangent penalty is also added as its own task (when enabled).
    """
    tf.summary.experimental.set_step(optimizer.iterations)

    use_deriv = getattr(Parameters, "use_foc_derivative_loss", False)
    sens_list = list(getattr(Parameters, "sensitivity_parameters", []))
    tau       = getattr(Parameters, "foc_derivative_gate_tau", 1e10)
    kappa     = getattr(Parameters, "foc_derivative_gate_kappa", 100.0)
    gate_mode = getattr(Parameters, "foc_derivative_gate_mode", "per_equation")

    # [IFT] flags
    use_ift   = getattr(Parameters, "use_ift_tangent_loss", False)
    ift_v     = int(getattr(Parameters, "ift_num_v_probes", 2))
    ift_u     = int(getattr(Parameters, "ift_num_theta_dirs", 1))
    ift_gate  = getattr(Parameters, "ift_gate_mode", gate_mode)
    ift_tau   = tf.cast(getattr(Parameters, "ift_gate_tau", tau), global_default_dtype)
    ift_kappa = tf.cast(getattr(Parameters, "ift_gate_kappa", kappa), global_default_dtype)
    ift_aggregate = bool(getattr(Parameters, "ift_aggregate_probes", True))
    ift_compact   = bool(getattr(Parameters, "ift_compact_mode", True))

    task_losses = []

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(state)
        policy_state = Parameters.policy(state)
        losses = Equations.equations(state, policy_state, training=training)

        for i, (eq_f, eq_res) in enumerate(losses.items()):
            if loss_choice == 'huber':
                eq_loss = tf.math.reduce_sum(Huber_loss(eq_res, Huber_loss_delta))
            else:
                eq_loss = tf.math.reduce_sum(eq_res ** 2)
            _maybe_log_scalar('dev_' + eq_f, eq_loss)
            if use_loss_reweighting and loss_reweighter.current_weights is not None:
                eq_loss = eq_loss * loss_reweighting_weight(loss_reweighter, i)
            task_losses.append(eq_loss)

        net_loss_val = tf.add_n(task_losses) if task_losses else tf.constant(0.0, dtype=global_default_dtype)

        if gate_mode == "batch_rmse":
            B = tf.cast(tf.shape(state)[0], global_default_dtype)
            no_eq_local = tf.cast(len(losses), global_default_dtype)
            rmse_no_penalty = tf.sqrt(net_loss_val / tf.maximum(B * no_eq_local, 1.0))
        else:
            rmse_no_penalty = None

        if use_ift and ift_compact:
            R_flat = _flatten_concat_residuals(losses)

    if use_loss_reweighting:
        loss_reweighter.update_weights(losses)

    if use_deriv and len(sens_list) > 0:
        param_to_idx = {p: Parameters.states.index(p) for p in sens_list if p in Parameters.states}
        if len(param_to_idx) > 0:
            lambda_ = _current_lambda()
            _maybe_log_scalar('lambda/current', lambda_)

            deriv_loss = tf.constant(0.0, dtype=global_default_dtype)
            for j, (eq_name, eq_res) in enumerate(losses.items()):
                grads_all_states = tape.gradient(
                    eq_res, state, unconnected_gradients=tf.UnconnectedGradients.ZERO
                )

                if gate_mode == "batch_rmse" and rmse_no_penalty is not None:
                    gate = tf.nn.sigmoid(kappa * (tf.cast(tau, global_default_dtype) - rmse_no_penalty))
                    for pidx in param_to_idx.values():
                        g = grads_all_states[:, pidx]
                        deriv_loss += tf.reduce_sum(g * g) * gate
                else:
                    rank = tf.rank(eq_res)
                    gate_row = tf.cond(
                        tf.greater(rank, 1),
                        true_fn=lambda: tf.reduce_mean(tf.math.abs(eq_res), axis=tf.range(1, rank)),
                        false_fn=lambda: tf.math.abs(eq_res)
                    )
                    gate = tf.nn.sigmoid(kappa * (tf.cast(tau, global_default_dtype) - gate_row))
                    for pidx in param_to_idx.values():
                        g = grads_all_states[:, pidx]
                        deriv_loss += tf.reduce_sum(gate * (g * g))

            weighted_deriv = lambda_ * deriv_loss
            _maybe_log_scalar('loss_derivative_penalty_mao', weighted_deriv)
            task_losses.append(weighted_deriv)
        else:
            _maybe_log_scalar('loss_derivative_penalty_mao', 0.0)
    else:
        _maybe_log_scalar('loss_derivative_penalty_mao', 0.0)

    if use_ift and len(sens_list) > 0:
        param_idxs = [Parameters.states.index(p) for p in sens_list if p in Parameters.states]
        if len(param_idxs) > 0:
            lambda_ift = _current_ift_lambda()
            _maybe_log_scalar('lambda_ift/current', lambda_ift)

            if tf.equal(lambda_ift, tf.constant(0.0, dtype=global_default_dtype)):
                _maybe_log_scalar('loss_ift_tangent_vjp_mao', 0.0)
            else:
                B = tf.shape(state)[0]
                S = tf.shape(state)[1]

                if ift_gate == "batch_rmse" and rmse_no_penalty is not None:
                    gate_global = tf.nn.sigmoid(ift_kappa * (ift_tau - rmse_no_penalty))
                    _maybe_log_scalar('ift/rmse_no_penalty', rmse_no_penalty)
                    _maybe_log_scalar('ift/gate_global', gate_global)
                else:
                    gate_global = None

                ift_loss_accum = tf.constant(0.0, dtype=global_default_dtype)

                if ift_compact:
                    Mtot = tf.shape(R_flat)[1]
                    for _ in range(ift_u):
                        u_param = _rand_unit_direction(len(param_idxs), global_default_dtype)
                        u_full = tf.zeros([S], dtype=global_default_dtype)
                        u_full = tf.tensor_scatter_nd_update(
                            u_full,
                            indices=tf.reshape(tf.constant(param_idxs, dtype=tf.int32), [-1, 1]),
                            updates=u_param
                        )
                        u_full_batched = tf.tile(tf.reshape(u_full, [1, -1]), [B, 1])

                        for _k in range(ift_v):
                            v = tf.random.normal([Mtot], dtype=global_default_dtype)
                            v /= (tf.norm(v) + tf.cast(1e-12, global_default_dtype))
                            v_batched = tf.tile(tf.reshape(v, [1, -1]), [B, 1])

                            grad_vjp_state = tape.gradient(
                                R_flat, state,
                                output_gradients=v_batched,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO
                            )
                            rv = tf.reduce_sum(tf.reduce_sum(grad_vjp_state * u_full_batched, axis=1))
                            rv2 = rv * rv
                            gated_rv2 = (gate_global * rv2) if gate_global is not None else rv2
                            ift_loss_accum += gated_rv2

                else:
                    v_probes = _sample_v_probes(losses, num_probes=ift_v, dtype=global_default_dtype)
                    ys = [eq_res for _, eq_res in losses.items()]
                    for _ in range(ift_u):
                        u_param = _rand_unit_direction(len(param_idxs), global_default_dtype)
                        u_full = tf.zeros([S], dtype=global_default_dtype)
                        u_full = tf.tensor_scatter_nd_update(
                            u_full,
                            indices=tf.reshape(tf.constant(param_idxs, dtype=tf.int32), [-1, 1]),
                            updates=u_param
                        )
                        u_full_batched = tf.tile(tf.reshape(u_full, [1, -1]), [B, 1])

                        for k in range(ift_v):
                            out_grads = []
                            for (eq_name, eq_res) in losses.items():
                                v = v_probes[eq_name][k]
                                v_batched = tf.tile(tf.reshape(v, [1, -1]), [B, 1])
                                v_shaped  = tf.reshape(v_batched, tf.shape(eq_res))
                                out_grads.append(v_shaped)

                            grad_vjp_state = tape.gradient(
                                ys, state,
                                output_gradients=out_grads,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO
                            )
                            rv = tf.reduce_sum(tf.reduce_sum(grad_vjp_state * u_full_batched, axis=1))
                            rv2 = rv * rv
                            gated_rv2 = (gate_global * rv2) if gate_global is not None else rv2
                            ift_loss_accum += gated_rv2

                weighted_ift = lambda_ift * ift_loss_accum
                _maybe_log_scalar('loss_ift_tangent_vjp_mao', weighted_ift)
                task_losses.append(weighted_ift)
        else:
            _maybe_log_scalar('loss_ift_tangent_vjp_mao', 0.0)

    del tape

    penalty_term = penalty_bounds_policy(state, policy_state)
    task_losses.append(penalty_term)

    total_loss = tf.add_n(task_losses)
    num_tasks  = len(task_losses)
    _maybe_log_scalar('dev_loss',     total_loss / num_tasks)
    _maybe_log_scalar('dev_net_loss', net_loss_val / num_tasks)
    return total_loss / num_tasks, net_loss_val / num_tasks, task_losses


def loss_reweighting_weight(reweighter, idx):
    w = reweighter.current_weights
    return w[idx] if w is not None else 1.0
