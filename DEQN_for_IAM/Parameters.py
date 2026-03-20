#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters.py
"""

import tensorflow as tf
import hydra
import os
import sys
from tensorflow import keras  # Keras 3 via TF 2.20
from omegaconf import OmegaConf
from optim.adahessian import AdaHessian
from FlexCheckpoint import FlexCheckpoint, FlexCheckpointManager
import Globals
Globals.POST_PROCESSING = False

# Detect if we have MPI/Horovod
if os.getenv('OMPI_COMM_WORLD_SIZE'):
    import horovod.tensorflow as hvd
    setattr(sys.modules[__name__], "horovod", True)
else:
    setattr(sys.modules[__name__], "horovod", False)

# If we are re‐running from a hydra run directory, load that config first
if "USE_CONFIG_FROM_RUN_DIR" in os.environ.keys():
    conf = OmegaConf.load(os.environ["USE_CONFIG_FROM_RUN_DIR"] + "/.hydra/config.yaml")
    conf_dict = OmegaConf.to_container(conf)
    if "run" not in conf_dict:
       import copy
       conf_dict["run"] = copy.deepcopy(conf_dict)
       conf_dict["constants"] = copy.deepcopy(conf_dict)
       conf_dict["net"] = copy.deepcopy(conf_dict)
       conf_dict["optimizer"] = copy.deepcopy(conf_dict)

    conf_new = OmegaConf.create(conf_dict)
    OmegaConf.save(config=conf_new, f="config_postprocess/config.yaml")


# --------------------------------------------------------------------------
# ADDED: Lambda (derivative weight) scheduler factory
# --------------------------------------------------------------------------
def _build_lambda_schedule(cfg, default_val, dtype, module_ref):
    """
    Returns a callable f(step) -> tf.Tensor(dtype), using either step- or episode-based schedules.
    Backward compatible: if cfg is None, returns constant(default_val).
    """
    def _constant(v):
        v = tf.convert_to_tensor(v, dtype=dtype)
        return lambda step: v

    if cfg is None:
        return _constant(default_val)

    by   = cfg.get("by", "steps").lower()
    typ  = cfg.get("type", "constant").lower()
    args = cfg.get("kwargs", {})

    def _current_episode_tensor():
        ce = getattr(module_ref, "ckpt", None)
        if ce is None:
            return tf.convert_to_tensor(1.0, dtype=tf.float32)
        return tf.cast(module_ref.ckpt.current_episode, tf.float32)

    if by == "steps":
        if typ == "constant":
            return _constant(args.get("value", default_val))

        if typ == "piecewise":
            boundaries = args.get("boundaries", [])
            values     = args.get("values", [default_val])
            if not values:
                values = [default_val]
            sched = keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries,
                values=[tf.convert_to_tensor(v, dtype=dtype) for v in values]
            )
            return lambda step: tf.cast(sched(step), dtype)

        if typ == "exponential":
            sched = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.get("initial", default_val),
                decay_steps=args.get("decay_steps", 50_000),
                decay_rate=args.get("decay_rate", 2.0),
                staircase=bool(args.get("staircase", True))
            )
            return lambda step: tf.cast(sched(step), dtype)

        if typ == "linear_warmup":
            start = float(args.get("start", min(1e-6, float(default_val))))
            end   = float(args.get("end",   default_val))
            steps = float(args.get("steps", 100_000))
            def fn(step):
                step = tf.cast(step, tf.float32)
                t = tf.clip_by_value(step / steps, 0.0, 1.0)
                val = start + (end - start) * t
                return tf.cast(val, dtype)
            return fn

        if typ == "cosine":
            sched = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=args.get("initial", default_val),
                first_decay_steps=args.get("first_decay_steps", 50_000),
                t_mul=args.get("t_mul", 2.0),
                m_mul=args.get("m_mul", 1.0),
                alpha=args.get("alpha", 0.0)
            )
            return lambda step: tf.cast(sched(step), dtype)

        return _constant(default_val)

    if by == "episodes":
        if typ == "constant":
            return _constant(args.get("value", default_val))

        if typ == "piecewise":
            boundaries = [float(b) for b in args.get("boundaries", [])]
            values     = [float(v) for v in args.get("values", [default_val])]
            def fn(_):
                ep = _current_episode_tensor()
                v = tf.convert_to_tensor(values[0], dtype=dtype)
                for b, nv in zip(boundaries, values[1:]):
                    v = tf.where(ep >= b, tf.convert_to_tensor(nv, dtype=dtype), v)
                return v
            return fn

        if typ == "linear_warmup":
            start = float(args.get("start", min(1e-6, float(default_val))))
            end   = float(args.get("end",   default_val))
            e0    = float(args.get("start_episode", 1))
            e1    = float(args.get("end_episode",   e0 + 100.0))
            width = max(1.0, e1 - e0)
            def fn(_):
                ep = _current_episode_tensor()
                t  = tf.clip_by_value((ep - e0) / width, 0.0, 1.0)
                val = start + (end - start) * t
                return tf.cast(val, dtype)
            return fn

        return _constant(default_val)

    return _constant(default_val)
# --------------------------------------------------------------------------


@hydra.main(
    config_path=("config_postprocess" if "USE_CONFIG_FROM_RUN_DIR" in os.environ.keys() else "config"),
    config_name="config.yaml"
)
def set_conf(cfg):
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("enable_check_numerics"):
        print("Enabling numerics debugging...")
        tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)

    setattr(sys.modules[__name__], "MODEL_NAME", cfg.MODEL_NAME)

    seed_offset = 0
    setattr(sys.modules[__name__], "horovod_worker", False)

    if horovod:
        hvd.init()
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
        if gpus:
            tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        seed_offset = hvd.rank()
        if seed_offset > 0:
            setattr(sys.modules[__name__], "horovod_worker", True)

    tf.random.set_seed(cfg.seed + seed_offset)
    rng_state = tf.Variable([0, 0, cfg.seed + seed_offset], dtype=tf.int64)
    setattr(sys.modules[__name__], "rng", tf.random.Generator.from_state(rng_state, alg='philox'))

    # Basic run parameters
    setattr(sys.modules[__name__], "N_sim_batch", cfg.run.N_sim_batch)
    setattr(sys.modules[__name__], "N_epochs_per_episode", cfg.run.N_epochs_per_episode)
    setattr(sys.modules[__name__], "N_minibatch_size", cfg.run.N_minibatch_size)
    # Ensure that episode length parameters are integers
    setattr(sys.modules[__name__], "N_episode_length", int(cfg.run.N_episode_length))
    setattr(sys.modules[__name__], "N_episodes", cfg.run.N_episodes)

    setattr(sys.modules[__name__], "n_quad_pts", cfg.run.get('n_quad_pts', 3))
    setattr(sys.modules[__name__], "expectation_pseudo_draws", cfg.run.get('expectation_pseudo_draws', 5))
    setattr(sys.modules[__name__], "expectation_type", cfg.run.get('expectation_type', 'product'))
    setattr(sys.modules[__name__], "sorted_within_batch", cfg.run.get('sorted_within_batch', False))

    # If user sets sorted_within_batch=true but the length < batch size, warn
    if sorted_within_batch and N_episode_length < N_minibatch_size:
        print("WARNING: minibatch size is larger than the episode length and sorted batches were requested!")

    setattr(sys.modules[__name__], "use_new_shuffling", cfg.get("use_new_shuffling", False))

    setattr(sys.modules[__name__], "error_filename", cfg.error_filename)

    # ---------------------------------------------------------------
    # NEW: incremental schedule. If not specified, default to old behavior.
    default_min_len = int(cfg.run.N_episode_length)  # fallback if user doesn't specify a min
    setattr(sys.modules[__name__], "N_episode_length_min", int(cfg.run.get("N_episode_length_min", default_min_len)))
    setattr(sys.modules[__name__], "N_increment_steps", int(cfg.run.get("N_increment_steps", 0)))
    setattr(sys.modules[__name__], "curriculum_breakpoints", cfg.run.get("curriculum_breakpoints", None))
    setattr(sys.modules[__name__], "curriculum_steps",       cfg.run.get("curriculum_steps", None))
    
    # ---------------------------------------------------------------

    # --- START: ADDED CODE FOR SENSITIVITY TRAINING ---
    setattr(sys.modules[__name__], "use_foc_derivative_loss", cfg.get("use_foc_derivative_loss", False))
    setattr(sys.modules[__name__], "foc_derivative_lambda", cfg.get("foc_derivative_lambda", 1.0))
    setattr(sys.modules[__name__], "sensitivity_parameters", list(cfg.get("sensitivity_parameters", [])))

    setattr(sys.modules[__name__], "foc_derivative_gate_tau", cfg.get("foc_derivative_gate_tau", 0.01))
    setattr(sys.modules[__name__], "foc_derivative_gate_kappa", cfg.get("foc_derivative_gate_kappa", 100.0))
    setattr(sys.modules[__name__], "deriv_compensate_no_eq", cfg.get("deriv_compensate_no_eq", False))
    setattr(sys.modules[__name__], "bounds_compensate_no_eq", cfg.get("bounds_compensate_no_eq", False))
    setattr(sys.modules[__name__], "foc_derivative_gate_mode", cfg.get("foc_derivative_gate_mode", "per_equation"))


    lambda_sched_cfg = cfg.get("lambda_scheduler", None)
    lambda_fn = _build_lambda_schedule(
        lambda_sched_cfg,
        cfg.get("foc_derivative_lambda", 1.0),
        dtype=tf.float32,
        module_ref=sys.modules[__name__],
    )
    setattr(sys.modules[__name__], "foc_derivative_lambda_schedule", lambda_fn)

    tf.print("Sensitivity training enabled:", getattr(sys.modules[__name__], "use_foc_derivative_loss"))
    if getattr(sys.modules[__name__], "use_foc_derivative_loss"):
        tf.print("Sensitivity parameters:", getattr(sys.modules[__name__], "sensitivity_parameters"))
        tf.print("Derivative loss lambda (initial):", getattr(sys.modules[__name__], "foc_derivative_lambda"))
        if lambda_sched_cfg is not None:
            from omegaconf import OmegaConf as _OC
            cfg_dump = _OC.to_container(lambda_sched_cfg, resolve=True)
            print("Lambda scheduler config:", cfg_dump)
        tf.print("Gating parameters (tau, kappa):",
                 getattr(sys.modules[__name__], "foc_derivative_gate_tau"), ",",
                 getattr(sys.modules[__name__], "foc_derivative_gate_kappa"))
        tf.print("Derivative penalty compensate /no_eq:",
                 getattr(sys.modules[__name__], "deriv_compensate_no_eq"))
    # --- END: ADDED CODE FOR SENSITIVITY TRAINING ---

    # --- START: ADDED CODE FOR IFT-CONSISTENT TANGENT PENALTY -----------------
    # Expose IFT loss flags to Equilibrium.py (safe VJP/JVP tangent residual term)
    setattr(sys.modules[__name__], "use_ift_tangent_loss", cfg.get("use_ift_tangent_loss", False))
    setattr(sys.modules[__name__], "ift_tangent_lambda", cfg.get("ift_tangent_lambda", 0.0))
    setattr(sys.modules[__name__], "ift_num_v_probes",   int(cfg.get("ift_num_v_probes",   2)))
    setattr(sys.modules[__name__], "ift_num_theta_dirs", int(cfg.get("ift_num_theta_dirs", 1)))

    # Gate settings (reuse semantics of foc_derivative_*):
    setattr(sys.modules[__name__], "ift_gate_mode",  cfg.get("ift_gate_mode",  cfg.get("foc_derivative_gate_mode", "per_equation")))
    setattr(sys.modules[__name__], "ift_gate_tau",   cfg.get("ift_gate_tau",   cfg.get("foc_derivative_gate_tau", 1e10)))
    setattr(sys.modules[__name__], "ift_gate_kappa", cfg.get("ift_gate_kappa", cfg.get("foc_derivative_gate_kappa", 100.0)))

    # Optional: its own scheduler (independent from foc_derivative_lambda)
    ift_sched_cfg = cfg.get("ift_lambda_scheduler", None)
    ift_lambda_fn = _build_lambda_schedule(
        ift_sched_cfg,
        cfg.get("ift_tangent_lambda", 0.0),
        dtype=tf.float32,
        module_ref=sys.modules[__name__],
    )
    setattr(sys.modules[__name__], "ift_tangent_lambda_schedule", ift_lambda_fn)

    # [NEW OPTIONS] Aggregation/compactness toggles to control memory/trace size
    setattr(sys.modules[__name__], "ift_aggregate_probes", cfg.get("ift_aggregate_probes", True))
    setattr(sys.modules[__name__], "ift_compact_mode",     cfg.get("ift_compact_mode", True))

    tf.print("IFT tangent loss enabled:", getattr(sys.modules[__name__], "use_ift_tangent_loss"))
    if getattr(sys.modules[__name__], "use_ift_tangent_loss"):
        tf.print("IFT lambda (initial):", getattr(sys.modules[__name__], "ift_tangent_lambda"))
        if ift_sched_cfg is not None:
            from omegaconf import OmegaConf as _OC
            cfg_dump_ift = _OC.to_container(ift_sched_cfg, resolve=True)
            print("IFT lambda scheduler config:", cfg_dump_ift)
        tf.print("IFT probes (v, u):", getattr(sys.modules[__name__], "ift_num_v_probes"),
                 ",", getattr(sys.modules[__name__], "ift_num_theta_dirs"))
        tf.print("IFT gating (mode, tau, kappa):",
                 getattr(sys.modules[__name__], "ift_gate_mode"), ",",
                 getattr(sys.modules[__name__], "ift_gate_tau"), ",",
                 getattr(sys.modules[__name__], "ift_gate_kappa"))
        tf.print("IFT aggregation/compact:", getattr(sys.modules[__name__], "ift_aggregate_probes"),
                 ",", getattr(sys.modules[__name__], "ift_compact_mode"))
    # --- END: ADDED CODE FOR IFT-CONSISTENT TANGENT PENALTY -------------------

    # --- START: OPTIONAL THETA-SAMPLING / QUAD / TIME-STRAT FLAGS ------------
    # These are used by Dynamics.py; harmless if you don't enable those options.
    setattr(sys.modules[__name__], "theta_sampling_mode",        cfg.get("theta_sampling_mode", "uniform"))  # "uniform" | "lhs_mix"
    setattr(sys.modules[__name__], "theta_lhs_frac",             cfg.get("theta_lhs_frac", 0.5))
    setattr(sys.modules[__name__], "theta_halton_frac",          cfg.get("theta_halton_frac", 0.0))
    setattr(sys.modules[__name__], "theta_edges_frac",           cfg.get("theta_edges_frac", 0.3))
    setattr(sys.modules[__name__], "theta_use_antithetic",       cfg.get("theta_use_antithetic", True))
    setattr(sys.modules[__name__], "theta_edge_eps",             cfg.get("theta_edge_eps", 0.02))
    setattr(sys.modules[__name__], "theta_always_fullrange_frac",cfg.get("theta_always_fullrange_frac", 0.1))

    setattr(sys.modules[__name__], "quad_order_f",               cfg.get("quad_order_f", 0))  # 0 -> ignore override
    setattr(sys.modules[__name__], "quad_order_T",               cfg.get("quad_order_T", 0))  # 0 -> ignore override

    setattr(sys.modules[__name__], "taux_stratify",              cfg.get("taux_stratify", False))
    setattr(sys.modules[__name__], "taux_strata",                cfg.get("taux_strata", 16))
    setattr(sys.modules[__name__], "taux_min_init",              cfg.get("taux_min_init", 0.0))
    setattr(sys.modules[__name__], "taux_max_init",              cfg.get("taux_max_init", 0.95))

    setattr(sys.modules[__name__], "theta_hist_every_n_episodes",cfg.get("theta_hist_every_n_episodes", 10))
    setattr(sys.modules[__name__], "theta_adaptive_edges",       cfg.get("theta_adaptive_edges", True))
    setattr(sys.modules[__name__], "theta_adaptive_target_centrality", cfg.get("theta_adaptive_target_centrality", 0.35))
    setattr(sys.modules[__name__], "theta_adaptive_eta",         cfg.get("theta_adaptive_eta", 0.15))
    # --- END: OPTIONAL THETA-SAMPLING / QUAD / TIME-STRAT FLAGS --------------

    try:
        import importlib
        variables = importlib.import_module(MODEL_NAME + ".Variables")
        config_states = variables.states
        config_policies = variables.policies
        config_definitions = variables.definitions
        config_constants = variables.constants
        if cfg.constants.constants:
            config_constants.update(cfg.constants.constants)
        print("Variables imported from Variables module:", MODEL_NAME + ".Variables")
    except ImportError:
        config_states = cfg.variables.states
        config_policies = cfg.variables.policies
        config_definitions = cfg.variables.definitions
        config_constants = cfg.constants.constants

    setattr(sys.modules[__name__], "states", [s['name'] for s in config_states])
    setattr(sys.modules[__name__], "policy_states", [s['name'] for s in config_policies])
    setattr(sys.modules[__name__], "definitions", [s['name'] for s in config_definitions])

    # Collect bounds for states
    state_bounds = {"lower": {}, "penalty_lower": {}, "upper": {}, "penalty_upper": {}}
    for s in config_states:
        if "bounds" in s.keys() and "lower" in s["bounds"].keys():
            state_bounds["lower"][s["name"]] = s["bounds"]["lower"]
            if 'penalty_lower' in s['bounds'].keys():
                penalty = s["bounds"]["penalty_lower"]
            else:
                penalty = 1 / s['bounds']['lower'] ** 2
            state_bounds["penalty_lower"][s["name"]] = penalty
        if "bounds" in s.keys() and "upper" in s["bounds"].keys():
            state_bounds["upper"][s["name"]] = s["bounds"]["upper"]
            if 'penalty_upper' in s['bounds'].keys():
                penalty = s["bounds"]["penalty_upper"]
            else:
                penalty = 1 / s['bounds']['upper'] ** 2
            state_bounds["penalty_upper"][s["name"]] = penalty
    setattr(sys.modules[__name__], "state_bounds_hard", state_bounds)

    # Collect bounds for policy
    policy_bounds = {'lower': {}, 'penalty_lower': {}, 'upper': {}, 'penalty_upper': {}}
    for s in config_policies:
        if 'activation' in s.keys():
            if s['activation'] != 'implied':
                if 'bounds' in s.keys() and 'lower' in s['bounds'].keys():
                    policy_bounds['lower'][s['name']] = s['bounds']['lower']
                    if 'penalty_lower' in s['bounds'].keys():
                        penalty = s["bounds"]["penalty_lower"]
                    else:
                        penalty = 1 / s['bounds']['lower'] ** 2
                    policy_bounds['penalty_lower'][s['name']] = penalty
                if 'bounds' in s.keys() and 'upper' in s['bounds'].keys():
                    policy_bounds['upper'][s['name']] = s['bounds']['upper']
                    if 'penalty_upper' in s['bounds'].keys():
                        penalty = s["bounds"]["penalty_upper"]
                    else:
                        penalty = 1 / s['bounds']['upper'] ** 2
                    policy_bounds['penalty_upper'][s['name']] = penalty
        else:
            if 'bounds' in s.keys() and 'lower' in s['bounds'].keys():
                policy_bounds['lower'][s['name']] = s['bounds']['lower']
                if 'penalty_lower' in s['bounds'].keys():
                    penalty = s["bounds"]["penalty_lower"]
                else:
                    penalty = 1 / s['bounds']['lower'] ** 2
                policy_bounds['penalty_lower'][s['name']] = penalty
            if 'bounds' in s.keys() and 'upper' in s['bounds'].keys():
                policy_bounds['upper'][s['name']] = s['bounds']['upper']
                if 'penalty_upper' in s['bounds'].keys():
                    penalty = s["bounds"]["penalty_upper"]
                else:
                    penalty = 1 / s['bounds']['upper'] ** 2
                policy_bounds['penalty_upper'][s['name']] = penalty
    setattr(sys.modules[__name__], "policy_bounds_hard", policy_bounds)

    # Collect bounds for definitions
    definition_bounds = {'lower': {}, 'penalty_lower': {}, 'upper': {}, 'penalty_upper': {}}
    for s in config_definitions:
        if 'bounds' in s.keys():
            if 'lower' in s['bounds'].keys():
                definition_bounds['lower'][s['name']] = s['bounds']['lower']
                if 'penalty_lower' in s['bounds'].keys():
                    penalty = s["bounds"]["penalty_lower"]
                else:
                    penalty = 1 / s['bounds']['lower'] ** 2
                definition_bounds['penalty_lower'][s['name']] = penalty
            if 'upper' in s['bounds'].keys():
                definition_bounds['upper'][s['name']] = s['bounds']['upper']
                if 'penalty_upper' in s['bounds'].keys():
                    penalty = s["bounds"]["penalty_upper"]
                else:
                    penalty = 1 / s['bounds']['upper'] ** 2
                definition_bounds['penalty_upper'][s['name']] = penalty
    setattr(sys.modules[__name__], "definition_bounds_hard", definition_bounds)

    # Precision
    keras.backend.set_floatx(cfg.run.get('keras_precision', 'float32'))
    if cfg.run.get('keras_precision', 'float32') == 'float32':
        default_dtype = tf.float32
    elif cfg.run.get('keras_precision', 'float32') == 'float64':
        default_dtype = tf.float64
    elif cfg.run.get('keras_precision', 'float32') == 'float16':
        default_dtype = tf.float16
    else:
        raise Exception("Unsupported dtype in config.")
    setattr(sys.modules[__name__], "global_default_dtype", default_dtype)

    # Build or import the policy_net
    try:
        import importlib
        net = importlib.import_module(MODEL_NAME + ".Net")
        policy_net = net.define_net(cfg, states, policy_states)
    except ImportError:
        layers = []
        for i, layer in enumerate(cfg.net.layers, start=1):
            if i < len(cfg.net.layers):
                if 'dropout_rate' in layer['hidden']:
                    layers.append(keras.layers.Dropout(rate=layer['hidden']['dropout_rate']))
                if 'batch_normalize' in layer['hidden']:
                    layers.append(keras.layers.BatchNormalization(**layer['hidden']['batch_normalize']))
                layers.append(keras.layers.Dense(
                    units=layer['hidden']['units'],
                    activation=layer['hidden']['activation'],
                    kernel_initializer=keras.initializers.VarianceScaling(
                        scale=layer['hidden'].get('init_scale', 1.0),
                        mode=cfg.net.get('net_initializer_mode', 'fan_in'),
                        distribution=cfg.net.get('net_initializer_distribution', 'truncated_normal'),
                        seed=i
                    )
                ))
            else:
                layers.append(keras.layers.Dense(
                    units=len(policy_states),
                    activation=layer['output']['activation'],
                    kernel_initializer=keras.initializers.VarianceScaling(
                        scale=layer['output'].get('init_scale', 1.0),
                        mode=cfg.net.get('net_initializer_mode', 'fan_in'),
                        distribution=cfg.net.get('net_initializer_distribution', 'truncated_normal'),
                        seed=i
                    )
                ))
        policy_net = keras.models.Sequential(layers)
        policy_net.build(input_shape=(None, len(states)))

    policy_net.summary()

    # ------------------------------------------------------------------
    # OPTIMIZER
    learning_rate_multiplier = 1 if not horovod else hvd.size()
    if "lr_scheduler" in cfg.optimizer:
        if cfg.optimizer.lr_scheduler == 'ReduceLROnPlateau':
            from optim.reducelr import ReduceLROnPlateau
            lr_scheduler = ReduceLROnPlateau(**cfg.optimizer.lr_scheduler_kwargs)
            setattr(sys.modules[__name__], "use_reduce_lr_on_plateau", True)
        else:
            lr_scheduler = getattr(keras.optimizers.schedules, cfg.optimizer.lr_scheduler)(
                **cfg.optimizer.lr_scheduler_kwargs
            )
            setattr(sys.modules[__name__], "use_reduce_lr_on_plateau", False)
        setattr(sys.modules[__name__], "lr_scheduler", lr_scheduler)
    else:
        lr_scheduler = None
        setattr(sys.modules[__name__], "use_reduce_lr_on_plateau", False)

    if cfg.optimizer.optimizer == 'mao':
        from optim.mao_optimizer import MAOOptimizer
        num_tasks = cfg.optimizer.get('num_tasks', 18)
        optim = MAOOptimizer(
            policy_net.trainable_variables,
            num_tasks,
            global_lr=cfg.optimizer.get('global_lr', cfg.optimizer.learning_rate),
            beta1=cfg.optimizer.get('beta1', 0.9),
            beta2=cfg.optimizer.get('beta2', 0.999),
            epsilon=cfg.optimizer.get('epsilon', 1e-8)
        )
    elif cfg.optimizer.optimizer == 'NGD':
        from optim.NGD import NGDOptimizer
        optim = NGDOptimizer(
            policy_net.trainable_variables,
            learning_rate=cfg.optimizer.learning_rate,
            damping=cfg.optimizer.damping,
            fisher_decay=cfg.optimizer.fisher_decay,
            use_bias_correction=cfg.optimizer.use_bias_correction
        )
    elif cfg.optimizer.optimizer == 'WARM_NGD':
        """
        Warm start with some standard TF optimizer, then switch to NGD.
        """
        warm_start_n_episodes = cfg.optimizer.get('warm_start_n_episodes', 5)
        warm_optimizer_name   = cfg.optimizer.get('warm_optimizer', 'Adam')
        warm_lr               = cfg.optimizer.get('warm_learning_rate', 0.001)
        warm_clipvalue        = cfg.optimizer.get('warm_clipvalue', 1.0)
        warm_kwargs           = cfg.optimizer.get('warm_optimizer_kwargs', {})

        # 1) Create the warm-start optimizer
        warm_opt_class = getattr(keras.optimizers, warm_optimizer_name)
        warm_opt = warm_opt_class(
            learning_rate=(
                warm_lr * learning_rate_multiplier if lr_scheduler is None else lr_scheduler
            ),
            clipvalue=warm_clipvalue,
            **warm_kwargs
        )

        # 2) Create the final NGD
        from optim.NGD import NGDOptimizer
        final_opt = NGDOptimizer(
            policy_net.trainable_variables,
            learning_rate=cfg.optimizer.get('final_learning_rate', 0.01),
            damping=cfg.optimizer.get('final_damping', 1e-4),
            fisher_decay=cfg.optimizer.get('final_fisher_decay', 0.95),
            use_bias_correction=cfg.optimizer.get('final_use_bias_correction', True)
        )

        optim = warm_opt
        setattr(sys.modules[__name__], "final_optimizer_ngd", final_opt)
        setattr(sys.modules[__name__], "warm_start_n_episodes", warm_start_n_episodes)
        setattr(sys.modules[__name__], "warm_start_in_use", True)
        setattr(sys.modules[__name__], "final_optim_name", "NGD")
    elif cfg.optimizer.optimizer != 'adahessian':
        # A standard Keras optimizer
        optim = getattr(keras.optimizers, cfg.optimizer.optimizer)(
            learning_rate=(
                cfg.optimizer.learning_rate * learning_rate_multiplier
                if lr_scheduler is None or getattr(sys.modules[__name__], "use_reduce_lr_on_plateau", False)
                else lr_scheduler
            ),
            clipvalue=cfg.optimizer.get('clipvalue', None),
            **(cfg.optimizer.get('optimizer_kwargs', {}))
        )
    else:
        # AdaHessian
        optim = AdaHessian(learning_rate=cfg.optimizer.learning_rate * learning_rate_multiplier)

    setattr(sys.modules[__name__], "optimizer", optim)
    setattr(sys.modules[__name__], "optim_name", cfg.optimizer.optimizer)

    # The policy function
    def policy(s, training=None):
        raw_policy = policy_net(s, training=training)
        # If there's "implied" activation or other custom logic in variables
        for i, pol in enumerate(config_policies):
            if 'activation' in pol.keys():
                activation_str = pol['activation']
                if pol['activation'] == 'implied':
                    if 'lower' in pol['bounds'].keys() and 'upper' in pol['bounds'].keys():
                        activation_str = ('lambda x: {l} + ({u} - {l}) * '
                                          'tf.math.sigmoid(x)').format(
                            l=str(pol['bounds']['lower']),
                            u=str(pol['bounds']['upper'])
                        )
                raw_policy = tf.tensor_scatter_nd_update(
                    raw_policy,
                    [[j, i] for j in range(s.shape[0])],
                    eval(activation_str)(raw_policy[:, i])
                )
        # Make sure final output is float32 if the net was float64
        if cfg.run.keras_precision == 'float64':
            return tf.cast(raw_policy, tf.float32)
        return raw_policy

    setattr(sys.modules[__name__], "policy", policy)
    setattr(sys.modules[__name__], "policy_net", policy_net)

    # Populate constants from config
    for (key, value) in config_constants.items():
        setattr(sys.modules[__name__], key, value)

    # Initialization of states
    def initialize_states(N_batch = N_sim_batch):
        init_val = tf.ones([N_batch, len(states)], dtype=default_dtype)
        for i, s in enumerate(config_states):
            if 'init' in s:
                dist_name = s["init"]["distribution"]
                kwargs = s["init"]["kwargs"]
                new_vals = getattr(rng, dist_name)(shape=(N_batch,), **kwargs)
                init_val = tf.tensor_scatter_nd_update(
                    init_val,
                    [[j, i] for j in range(init_val.shape[0])],
                    new_vals
                )
        return init_val

    starting_state = tf.Variable(initialize_states())
    setattr(sys.modules[__name__], "starting_state", starting_state)
    setattr(sys.modules[__name__], "initialize_states", initialize_states)
    setattr(sys.modules[__name__], "initialize_each_episode", cfg.get("initialize_each_episode", False))
    setattr(sys.modules[__name__], "N_simulated_batch_size", cfg.get("N_simulated_batch_size", None))
    setattr(sys.modules[__name__], "N_simulated_episode_length", cfg.get("N_simulated_episode_length", None))
    setattr(sys.modules[__name__], "loss_choice", cfg.get("loss_choice", "mse"))
    setattr(sys.modules[__name__], "deriv_order", cfg.get("deriv_order", 1))
    setattr(sys.modules[__name__], "exp_samples", cfg.get("exp_samples", 1000))
    setattr(sys.modules[__name__], "use_weight_adjustment", cfg.get("use_weight_adjustment", False))

    setattr(sys.modules[__name__], "LOG_DIR", os.getcwd())

    # Handle checkpointing
    if cfg.STARTING_POINT == 'NEW' and not horovod_worker:
        for file in os.scandir(os.getcwd()):
            if not ".hydra" in file.path:
                os.unlink(file.path)
    setattr(sys.modules[__name__], "writer", tf.summary.create_file_writer(os.getcwd()))

    setattr(sys.modules[__name__], "current_episode", tf.Variable(1))
    if not cfg.get("use_flex_checkpoint", False):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            current_episode=current_episode,
            optimizer=optimizer,
            policy=policy_net,
            rng_state=rng_state,
            starting_state=starting_state
        )
        manager = tf.train.CheckpointManager(
            ckpt, os.getcwd(), max_to_keep=cfg.MAX_TO_KEEP_NUMBER,
            step_counter=current_episode,
            checkpoint_interval=cfg.CHECKPOINT_INTERVAL
        )
    else:
        ckpt = FlexCheckpoint(
            current_episode=current_episode,
            optimizer=optimizer,
            policy=policy_net,
            rng_state=rng_state,
            starting_state=starting_state
        )
        manager = FlexCheckpointManager(
            ckpt, os.getcwd(),
            max_to_keep=cfg.MAX_TO_KEEP_NUMBER,
            checkpoint_interval=cfg.CHECKPOINT_INTERVAL
        )

    if cfg.STARTING_POINT == 'LATEST' and manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        if cfg.get("reload_optimizer", False):
            ckpt.restore(manager.latest_checkpoint, load_optim=True)
        else:
            ckpt.restore(manager.latest_checkpoint)
    elif cfg.STARTING_POINT != 'LATEST' and cfg.STARTING_POINT != 'NEW':
        print("Restored from {}".format(cfg.STARTING_POINT))
        if cfg.get("reload_optimizer", False):
            ckpt.restore(cfg.STARTING_POINT, load_optim=True)
        else:
            ckpt.restore(cfg.STARTING_POINT)

    # Record the iteration count at the time of checkpoint loading
    setattr(sys.modules[__name__], "optimizer_starting_iteration", optimizer.iterations.numpy())
    setattr(sys.modules[__name__], "ckpt", ckpt)
    setattr(sys.modules[__name__], "manager", manager)

    tf.print("Optimizer configuration:")
    tf.print(optimizer.get_config())
    tf.print("Starting state:")
    tf.print(starting_state)

    use_loss_reweighting = cfg.get("use_loss_reweighting", False)
    setattr(sys.modules[__name__], "use_loss_reweighting", use_loss_reweighting)
    if use_loss_reweighting:
        from optim.reweighter import LossReweighting
        reweighting_scheme = cfg.get("reweighting_scheme", "softmax")
        loss_reweighter = LossReweighting(scheme=reweighting_scheme)
        setattr(sys.modules[__name__], "loss_reweighter", loss_reweighter)

    print_weights_every_n_episodes = cfg.get("print_weights_every_n_episodes", 1)
    setattr(sys.modules[__name__], "print_weights_every_n_episodes", print_weights_every_n_episodes)


set_conf()
