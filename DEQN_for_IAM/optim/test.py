import tensorflow as tf
from Parameters import global_default_dtype

class LossReweighting:
    def __init__(self, scheme="softmax", gamma=2.0, alpha=0.9, rho=0.9, temperature=1.0):
        """
        Initializes the loss reweighter.

        Args:
            scheme (str): The weighting scheme to use.
                          "softmax" applies a softmax to the normalized absolute accumulated losses.
                          "inverse" computes scaling factors as the inverse of the absolute loss means,
                                    then normalizes them to sum to one.
                          "focal" uses a focal-loss–inspired weighting for each task.
                                    For each task with accumulated loss L_i, define:
                                      p_i = 1/(1+|L_i|),
                                      and then f_i = (1-p_i)^gamma = (|L_i|/(1+|L_i|))^gamma.
                                    The final weight is the normalized f_i.
                          "relobralo" implements an epoch-based version of the Reinforcement
                                      Learning-based Online Balance-and-Refine Approach. It uses
                                      an EMA of balanced weights based on current, previous, and
                                      initial losses.
            gamma (float): Focusing parameter used only in the "focal" scheme. Default is 2.0.
            alpha (float): Smoothing parameter for the "relobralo" scheme. Default is 0.9.
            rho (float): Smoothing parameter for the "relobralo" scheme. Default is 0.9.
            temperature (float): Temperature for softmax in the "relobralo" scheme. Default is 1.0.
        """
        self.scheme = scheme
        self.gamma = gamma
        self.alpha = alpha
        self.rho = rho
        self.temperature = temperature
        
        self.accumulated_losses = {}
        self.current_weights = None
        self.epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._initialized = False
        self._loss_keys = None  # To maintain a consistent order of losses

        # State variables for the "relobralo" scheme
        if self.scheme == "relobralo":
            self.previous_losses = None
            self.initial_losses = None
            self.previous_lambdas = None

    def initialize_accumulators(self, losses):
        """
        Initializes state variables based on the first batch of losses.
        """
        self._loss_keys = list(losses.keys())
        num_losses = len(self._loss_keys)

        for eq_f in self._loss_keys:
            self.accumulated_losses[eq_f] = tf.Variable(
                tf.zeros_like(tf.reduce_mean(losses[eq_f])), trainable=False
            )
        
        if self.scheme == "relobralo":
            dtype = global_default_dtype
            self.previous_losses = tf.Variable(tf.zeros(num_losses, dtype=dtype), trainable=False)
            self.initial_losses = tf.Variable(tf.zeros(num_losses, dtype=dtype), trainable=False)
            # Initialize weights uniformly
            initial_lambdas = tf.ones(num_losses, dtype=dtype) / tf.cast(num_losses, dtype=dtype)
            self.previous_lambdas = tf.Variable(initial_lambdas, trainable=False)

        self._initialized = True

    def update_weights(self, losses):
        """
        Accumulates the mean loss for each task from a training step.
        """
        if not self._initialized:
            self.initialize_accumulators(losses)
        
        for eq_f, loss in losses.items():
            # Compute the mean loss for the batch and add it to the accumulator
            mean_loss = tf.reduce_mean(loss)
            self.accumulated_losses[eq_f].assign_add(mean_loss)

    def compute_epoch_weights(self):
        """
        Computes the final loss weights for the next epoch based on the selected scheme.
        """
        # Get the accumulated losses in a fixed order
        loss_means = [self.accumulated_losses[key].value() for key in self._loss_keys]
        self._last_raw_losses = {key: var.numpy() for key, var in self.accumulated_losses.items()}
        
        if self.scheme == "softmax":
            loss_means_tensor = tf.stack(loss_means)
            abs_losses = tf.abs(loss_means_tensor)
            # Temperature parameter for softmax
            temp = 0.3
            weights = tf.nn.softmax(abs_losses / temp)
        
        elif self.scheme == "inverse":
            loss_means_tensor = tf.stack(loss_means)
            abs_losses = tf.abs(loss_means_tensor)
            scaling_factors = 1.0 / (abs_losses + 1e-8)
            weights = scaling_factors / tf.reduce_sum(scaling_factors)
        
        elif self.scheme == "focal":
            focal_weights = []
            for L in loss_means:
                absL = tf.abs(L)
                p = 1.0 / (1.0 + absL)
                focal = tf.pow(1.0 - p, self.gamma)
                focal_weights.append(focal)
            weights = tf.stack(focal_weights)
            weights = weights / tf.reduce_sum(weights)
            
        elif self.scheme == "relobralo":
            current_losses = tf.stack(loss_means)
            num_losses = tf.cast(len(loss_means), dtype=global_default_dtype)

            # After the first epoch (epoch 0), store the accumulated losses as initial losses.
            if self.epoch.numpy() == 0:
                self.initial_losses.assign(current_losses)
                # For the first update, assume previous losses are the initial ones.
                self.previous_losses.assign(current_losses)

            # λ_bal(t, t-1): balanced weights based on previous epoch's losses
            ratio_t_minus_1 = current_losses / (self.previous_losses * self.temperature + 1e-12)
            lambs_hat = tf.nn.softmax(ratio_t_minus_1) * num_losses

            # λ_bal(t, 0): balanced weights based on initial epoch's losses
            ratio_t_0 = current_losses / (self.initial_losses * self.temperature + 1e-12)
            lambs0_hat = tf.nn.softmax(ratio_t_0) * num_losses

            # Final weight combination: λ(t) = α*ρ*λ(t-1) + α*(1-ρ)*λ_bal(t,0) + (1-α)*λ_bal(t, t-1)
            weights = (self.alpha * self.rho * self.previous_lambdas +
                       self.alpha * (1.0 - self.rho) * lambs0_hat +
                       (1.0 - self.alpha) * lambs_hat)

            # Update state variables for the next epoch's calculation
            self.previous_losses.assign(current_losses)
            self.previous_lambdas.assign(weights)

        else:
            raise ValueError(f"Unknown weighting scheme: {self.scheme}")

        self.current_weights = tf.cast(weights, dtype=global_default_dtype)
        
        # Reset accumulators for the next epoch
        for var in self.accumulated_losses.values():
            var.assign(tf.zeros_like(var))
        
        self.epoch.assign_add(1)
        return self.current_weights

    def print_status(self):
        """
        Prints the latest raw accumulated losses and the computed weights.
        """
        if not self._initialized:
            print("LossReweighter has not been initialized yet.")
            return
            
        print(f"\nEpoch {self.epoch.numpy()} Loss Reweighting Status:")
        print("-" * 50)
        print("Accumulated Losses (Epoch Total):")
        for eq_f, loss in self._last_raw_losses.items():
            print(f"{eq_f:20s}: {loss:10.4f}")
        
        if self.current_weights is not None:
            print("\nComputed Weights for Next Epoch:")
            weights_np = self.current_weights.numpy()
            for eq_f, weight in zip(self._loss_keys, weights_np):
                print(f"{eq_f:20s}: {weight:10.4f}")
        print("-" * 50)