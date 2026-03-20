import os
import pickle
import tensorflow as tf
import shutil
import time 
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
import numpy as np

class FlexCheckpoint:
    def __init__(self, policy, optimizer, starting_state, current_episode, rng_state, save_dir=os.getcwd()):
        self.policy_net = policy
        self.optimizer = optimizer
        self.starting_state = starting_state
        self.current_episode = current_episode
        self.save_dir = save_dir
        self.rng_state = rng_state

    def save(self):
        # Save optimizer weights
        optimizer_weights = {}
        # Use Keras optimizer API if available
        if isinstance(self.optimizer, OptimizerV2):
            raw_weights = self.optimizer.get_weights()
            for i, w in enumerate(raw_weights):
                optimizer_weights[str(i)] = w
        else:
            # If the optimizer has a save_own_variables method, use it.
            if hasattr(self.optimizer, "save_own_variables"):
                self.optimizer.save_own_variables(optimizer_weights)
            else:
                # Fallback for custom optimizers (e.g., MAOOptimizer)
                optimizer_weights["global_step"] = self.optimizer.global_step.numpy()
                optimizer_weights["num_tasks"] = self.optimizer.num_tasks
                m_list = []
                v_list = []
                for task in range(self.optimizer.num_tasks):
                    m_task = [w.numpy() for w in self.optimizer.m[task]]
                    v_task = [w.numpy() for w in self.optimizer.v[task]]
                    m_list.append(m_task)
                    v_list.append(v_task)
                optimizer_weights["m"] = m_list
                optimizer_weights["v"] = v_list

        with open(os.path.join(self.save_dir, "optimizer.pkl"), "wb") as f:
            pickle.dump(optimizer_weights, f)

        # Save policy net weights
        policy_net_weights = self.policy_net.get_weights()
        with open(os.path.join(self.save_dir, "policy_net.pkl"), "wb") as f:
            pickle.dump(policy_net_weights, f)

        # Save starting state
        with open(os.path.join(self.save_dir, "starting_state.pkl"), "wb") as f:
            pickle.dump({'state': self.starting_state.numpy(), 'batch_size': self.starting_state.shape[0]}, f)

        # Save current episode
        with open(os.path.join(self.save_dir, "current_episode.pkl"), "wb") as f:
            pickle.dump(self.current_episode.numpy(), f)
        
        # Save RNG state
        with open(os.path.join(self.save_dir, "rng_state.pkl"), "wb") as f:
            pickle.dump(self.rng_state, f)
        
    def load(self, load_optim=False):
        # Load optimizer weights
        if load_optim:
            with open(os.path.join(self.save_dir, "optimizer.pkl"), "rb") as f:
                optimizer_weights = pickle.load(f)
            grad_vars = self.policy_net.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            if isinstance(self.optimizer, OptimizerV2):
                self.optimizer.apply_gradients_hessian(zip(zero_grads, zero_grads, grad_vars))
                self.optimizer.set_weights(list(optimizer_weights.values()))
            elif hasattr(self.optimizer, "apply_gradients"):
                self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
                if hasattr(self.optimizer, "load_own_variables"):
                    self.optimizer.load_own_variables(optimizer_weights)
                else:
                    # Fallback for custom optimizers with apply_gradients
                    self.optimizer.global_step.assign(optimizer_weights["global_step"])
                    for task in range(self.optimizer.num_tasks):
                        for i, var in enumerate(self.optimizer.m[task]):
                            var.assign(optimizer_weights["m"][task][i])
                        for i, var in enumerate(self.optimizer.v[task]):
                            var.assign(optimizer_weights["v"][task][i])
            else:
                # If the optimizer does not have an apply_gradients method,
                # assume it's a custom optimizer (e.g. MAOOptimizer) and load manually.
                self.optimizer.global_step.assign(optimizer_weights["global_step"])
                for task in range(self.optimizer.num_tasks):
                    for i, var in enumerate(self.optimizer.m[task]):
                        var.assign(optimizer_weights["m"][task][i])
                    for i, var in enumerate(self.optimizer.v[task]):
                        var.assign(optimizer_weights["v"][task][i])

        # Load policy net weights
        with open(os.path.join(self.save_dir, "policy_net.pkl"), "rb") as f:
            policy_net_weights = pickle.load(f)
        self.policy_net.set_weights(policy_net_weights)

        # Load starting state
        with open(os.path.join(self.save_dir, "starting_state.pkl"), "rb") as f:
            starting_state_data = pickle.load(f)
        starting_state_value = starting_state_data['state']
        original_batch_size = starting_state_data['batch_size']
        if self.starting_state is not None and self.starting_state.shape[0] != original_batch_size:
            # Adjust starting_state to new batch size
            if self.starting_state.shape[0] > original_batch_size:
                repeats = self.starting_state.shape[0] // original_batch_size + 1
                starting_state_value = np.tile(starting_state_value, (repeats, 1))[:self.starting_state.shape[0]]
            else:
                starting_state_value = starting_state_value[:self.starting_state.shape[0]]
        self.starting_state.assign(starting_state_value)
    
        # Load current episode
        with open(os.path.join(self.save_dir, "current_episode.pkl"), "rb") as f:
            current_episode_value = pickle.load(f)
        self.current_episode.assign(current_episode_value)
    
    def restore(self, checkpoint_path, load_optim=False):
        self.save_dir = os.path.realpath(checkpoint_path)
        self.load(load_optim=load_optim)


class FlexCheckpointManager:
    def __init__(self, checkpoint, directory, max_to_keep=5, checkpoint_interval=None):
        self.checkpoint = checkpoint
        self.directory = directory
        self.max_to_keep = max_to_keep
        self.save_interval_episodes = checkpoint_interval
        self.checkpoints = []
        self.last_save_episode = self.checkpoint.current_episode.numpy()
        self.counter = 0

        if not os.path.exists(directory):
            os.makedirs(directory)
        for file in os.listdir(directory):
            if file.startswith("ckpt_"):
                self.checkpoints.append(os.path.join(directory, file))
        self.checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
        self.counter = int(self.checkpoints[-1].split("_")[-1]) + 1 if self.checkpoints else 0

    def save(self, force=False):
        current_episode = self.checkpoint.current_episode.numpy()
        
        if not force and self.save_interval_episodes is not None:
            if current_episode - self.last_save_episode < self.save_interval_episodes:
                return None  

        checkpoint_path = os.path.join(self.directory, f"ckpt_{self.counter}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.checkpoint.save_dir = checkpoint_path
        self.checkpoint.save()

        self.checkpoints.append(checkpoint_path)
        self.last_save_episode = current_episode
        self.counter += 1
        # Remove old checkpoints if necessary
        while len(self.checkpoints) > self.max_to_keep:
            path_to_remove = self.checkpoints.pop(0)
            self.remove_checkpoint(path_to_remove)

        return checkpoint_path

    def remove_checkpoint(self, checkpoint_path):
        print(f"Removing checkpoint: {checkpoint_path}")
        for file in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, file))
        try:
            os.rmdir(checkpoint_path)
        except OSError:
            pass

    @property
    def latest_checkpoint(self):
        return self.checkpoints[-1] if self.checkpoints else None

    def all_checkpoints(self):
        return self.checkpoints

    @property
    def current_episode(self):
        return self.checkpoint.current_episode.numpy()

    @property
    def current_step(self):
        return self.checkpoint.step.numpy()
