import tensorflow as tf
from Parameters import global_default_dtype

class LossReweighting:
    def __init__(self, scheme="softmax", gamma=2.0):
        """
        Initializes the loss reweighter.

        Args:
            scheme (str): The weighting scheme to use.
                          "softmax" applies a softmax to the (absolute) accumulated losses.
                          "inverse" computes scaling factors as the inverse of the absolute loss means,
                                    then normalizes them to sum to one.
                          "focal" uses a focal-loss–inspired weighting for each task.
                                    For each task with accumulated loss L_i, define:
                                      p_i = 1/(1+|L_i|),
                                      and then f_i = (1-p_i)^gamma = (|L_i|/(1+|L_i|))^gamma.
                                    The final weight is the normalized f_i.
            gamma (float): Focusing parameter used only in the "focal" scheme. Default is 2.0.
        """
        self.scheme = scheme
        self.gamma = gamma
        self.accumulated_losses = {}
        self.current_weights = None
        self.epoch = tf.Variable(0, dtype=tf.int32)
        self._initialized = False

    def initialize_accumulators(self, losses):
        for eq_f, loss in losses.items():
            self.accumulated_losses[eq_f] = tf.Variable(
                tf.zeros_like(tf.reduce_mean(loss)), trainable=False
            )
        self._initialized = True

    def update_weights(self, losses):
        if not self._initialized:
            self.initialize_accumulators(losses)
        for eq_f, loss in losses.items():
            # Compute the mean loss and ensure non-zero magnitude with a small epsilon.
            mean_loss = tf.reduce_mean(loss)
            safe_loss = tf.sign(mean_loss) * tf.maximum(tf.abs(mean_loss), 1e-8)
            self.accumulated_losses[eq_f].assign_add(safe_loss)

    def compute_epoch_weights(self):
        # Get the raw accumulated loss means.
        loss_means = [self.accumulated_losses[eq_f].value() for eq_f in self.accumulated_losses.keys()]
        # Store raw values for logging.
        self._last_raw_losses = { eq_f: var.numpy() for eq_f, var in self.accumulated_losses.items() }
        
        if self.scheme == "softmax":
            loss_means = tf.stack(loss_means)
            abs_losses = tf.abs(loss_means)
            max_abs = tf.reduce_max(abs_losses)
            normalized = loss_means / (max_abs + 1e-8)
            abs_normalized = tf.abs(normalized)
            temperature = 0.3  # adjustable parameter
            weights = tf.nn.softmax(abs_normalized / temperature)
        elif self.scheme == "inverse":
            loss_means = tf.stack(loss_means)
            abs_losses = tf.abs(loss_means)
            scaling_factors = 1.0 / (abs_losses + 1e-8)
            weights = scaling_factors / tf.reduce_sum(scaling_factors)
        elif self.scheme == "focal":
            # For each task, compute p_i = 1/(1+|L_i|) and focal weight f_i = (1-p_i)^gamma.
            focal_weights = []
            for L in loss_means:
                absL = tf.abs(L)
                p = 1.0 / (1.0 + absL)
                focal = tf.pow(1.0 - p, self.gamma)
                focal_weights.append(focal)
            weights = tf.stack(focal_weights)
            weights = weights / tf.reduce_sum(weights)
        else:
            raise ValueError("Unknown weighting scheme: " + self.scheme)

        self.current_weights = tf.cast(weights, dtype=global_default_dtype)
        # Reset accumulators for the next epoch.
        for var in self.accumulated_losses.values():
            var.assign(tf.zeros_like(var))
        self.epoch.assign_add(1)
        return self.current_weights

    def print_status(self):
        print(f"\nEpoch {self.epoch.numpy()} Loss Reweighting Status:")
        print("-" * 50)
        print("Accumulated Losses:")
        for eq_f, loss in self._last_raw_losses.items():
            print(f"{eq_f:20s}: {loss:10.4f}")
        if self.current_weights is not None:
            print("\nCurrent Weights:")
            weights_np = self.current_weights.numpy()
            for eq_f, weight in zip(self.accumulated_losses.keys(), weights_np):
                print(f"{eq_f:20s}: {weight:10.4f}")
