"""
===============================================================================
 Multi-Adaptive Optimization (MAO) with Adam for Multiple Tasks
===============================================================================

This script demonstrates a general approach for multi-task learning using an
Adam-based Multi-Adaptive Optimization (MAO). The key idea is that each task
maintains its own set of first- and second-moment accumulators (m and v) for
each trainable parameter. Then, at each training step:

  1) We compute a distinct loss for each task (e.g., regression, classification).
  2) We calculate gradients per task. (Any parameter not used by a given task
     will have a gradient of None, which we treat as zero.)
  3) For each task n, we update the Adam moment accumulators m_n and v_n:
         m_n <- beta1 * m_n + (1 - beta1) * g_n
         v_n <- beta2 * v_n + (1 - beta2) * (g_n)^2
     and apply bias correction:
         m_hat_n = m_n / (1 - beta1^t)
         v_hat_n = v_n / (1 - beta2^t)
  4) We compute the per-task effective learning rate:
         lr_n = global_lr / (sqrt(v_hat_n) + epsilon)
  5) Each task contributes an update:
         update_n = lr_n * m_hat_n
     which we SUM across tasks for each parameter:
         total_update = sum( update_n for n in tasks )
  6) We subtract total_update from the parameter to perform the MAO step.

By giving each task its own moment estimates, MAO allows multiple tasks to
adapt at different rates and effectively share parameters, improving
multi-task training stability and speed. Here, we illustrate this with three
tasks on a toy function f(x) = x^2:
  - Task 1 (Regression):   y = x^2
  - Task 2 (Classification): class 0 if x^2 < 2 else 1
  - Task 3 (Classification): class 0 if x^2>3 else 1

At the end, we evaluate the model on 100 new points, computing MSE for the
regression and accuracy for each classification task.
===============================================================================
"""

import numpy as np
import tensorflow as tf

# NEW CLASS: MAOOptimizer
class MAOOptimizer:
    def __init__(self, variables, num_tasks, global_lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the MAO optimizer.
        
        Args:
            variables: list of tf.Variable, the trainable parameters.
            num_tasks: int, number of tasks (components) in the loss.
            global_lr: float, global learning rate.
            beta1: float, exponential decay rate for the first moment estimates.
            beta2: float, exponential decay rate for the second moment estimates.
            epsilon: float, small constant to avoid division by zero.
        """
        self.num_tasks = num_tasks
        self.global_lr = global_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.variables = variables
        
        # List of lists: one (m,v) pair per task, each covering all variables
        self.m = []
        self.v = []
        for _ in range(num_tasks):
            self.m.append([tf.Variable(tf.zeros_like(var), trainable=False) for var in variables])
            self.v.append([tf.Variable(tf.zeros_like(var), trainable=False) for var in variables])
        
        # A global step to mimic 'iterations' in TF's built-in optimizers
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    @property
    def iterations(self):
        """Mimic the Keras optimizer attribute for iteration count."""
        return self.global_step

    def get_config(self):
        """
        Minimal get_config to avoid AttributeError in code that tries 
        to print optimizer.get_config().
        """
        return {
            "num_tasks": self.num_tasks,
            "global_lr": self.global_lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }

    @tf.function
    def apply_mao_gradients(self, task_gradients):
        """
        Applies the MAO update using the gradients from each task.
        
        Args:
            task_gradients: list of gradients per task. Each element is a list 
                            of gradients for each variable.
        """
        self.global_step.assign_add(1)
        t = tf.cast(self.global_step, tf.float32)  # bias correction exponent

        for i, var in enumerate(self.variables):
            total_update = 0.0
            for n in range(self.num_tasks):
                g_n = task_gradients[n][i]
                if g_n is None:
                    g_n = tf.zeros_like(var)
                # Update first and second moments for this task
                self.m[n][i].assign(self.beta1 * self.m[n][i] + (1. - self.beta1) * g_n)
                self.v[n][i].assign(self.beta2 * self.v[n][i] + (1. - self.beta2) * tf.square(g_n))

                # Bias-corrected moments
                m_hat = self.m[n][i] / (1.0 - tf.pow(self.beta1, t))
                v_hat = self.v[n][i] / (1.0 - tf.pow(self.beta2, t))

                # Per-task effective learning rate
                lr_n = self.global_lr / (tf.sqrt(v_hat) + self.epsilon)
                update_n = lr_n * m_hat
                total_update += update_n

            # Apply combined update from all tasks
            var.assign_sub(total_update)


###############################################################################
# Below is the original illustrative code example showing how to use MAO on a 
# simple 3-task problem. We wrap it in an `if __name__ == "__main__":` so it 
# does NOT run automatically when imported.
###############################################################################
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 300
    x_vals = np.random.uniform(low=0.0, high=3.0, size=(n_samples, 1))
    y_reg_vals  = x_vals**2
    y_clf2_vals = (y_reg_vals >= 2.0).astype(np.float32)  # 0 if x^2<2 else 1
    y_clf3_vals = (y_reg_vals <= 3.0).astype(np.float32)  # 0 if x^2>3 else 1

    x_tensor   = tf.convert_to_tensor(x_vals,      dtype=tf.float32)
    y_reg_t    = tf.convert_to_tensor(y_reg_vals,  dtype=tf.float32)
    y_clf2_t   = tf.convert_to_tensor(y_clf2_vals, dtype=tf.float32)
    y_clf3_t   = tf.convert_to_tensor(y_clf3_vals, dtype=tf.float32)

    inputs   = tf.keras.Input(shape=(1,))
    hidden   = tf.keras.layers.Dense(16, activation='relu')(inputs)

    # 3 separate "heads" for the 3 tasks
    out_reg  = tf.keras.layers.Dense(1, name='out_reg')(hidden)
    out_clf2 = tf.keras.layers.Dense(1, activation='sigmoid', name='out_clf2')(hidden)
    out_clf3 = tf.keras.layers.Dense(1, activation='sigmoid', name='out_clf3')(hidden)

    model    = tf.keras.Model(inputs=inputs, outputs=[out_reg, out_clf2, out_clf3])

    N = 3  # number of tasks
    global_lr = 0.01
    beta1     = 0.9
    beta2     = 0.999
    epsilon   = 1e-8

    trainable_vars = model.trainable_variables

    # We'll just create plain lists for moment accumulators to show the idea:
    m = []
    v = []
    for _ in range(N):
        m.append([tf.Variable(tf.zeros_like(var), trainable=False) for var in trainable_vars])
        v.append([tf.Variable(tf.zeros_like(var), trainable=False) for var in trainable_vars])

    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    @tf.function
    def mao_adam_train_step(x_batch, y_reg_batch, y_clf2_batch, y_clf3_batch):
        # Increment global step
        global_step.assign_add(1)
        t = tf.cast(global_step, tf.float32)  # bias correction exponent

        with tf.GradientTape(persistent=True) as tape:
            pred_reg, pred_clf2, pred_clf3 = model(x_batch, training=True)

            # Task 1 (Regression): MSE
            loss_t1 = tf.reduce_mean(tf.square(pred_reg - y_reg_batch))

            # Task 2 (Classification): Binary Crossentropy
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss_t2 = bce(y_clf2_batch, pred_clf2)

            # Task 3 (Classification): Another Binary Crossentropy
            loss_t3 = bce(y_clf3_batch, pred_clf3)

        grads_t1 = tape.gradient(loss_t1, trainable_vars)
        grads_t2 = tape.gradient(loss_t2, trainable_vars)
        grads_t3 = tape.gradient(loss_t3, trainable_vars)
        del tape

        grads_all = [grads_t1, grads_t2, grads_t3]

        for i, var in enumerate(trainable_vars):
            total_update = 0.0
            for n in range(N):
                g_n = grads_all[n][i]
                if g_n is None:
                    g_n = tf.zeros_like(var)
                m[n][i].assign(beta1 * m[n][i] + (1. - beta1) * g_n)
                v[n][i].assign(beta2 * v[n][i] + (1. - beta2) * tf.square(g_n))

                m_hat = m[n][i] / (1.0 - tf.pow(beta1, t))
                v_hat = v[n][i] / (1.0 - tf.pow(beta2, t))

                lr_n = global_lr / (tf.sqrt(v_hat) + epsilon)
                update_n = lr_n * m_hat
                total_update += update_n

            var.assign_sub(total_update)

        return [loss_t1, loss_t2, loss_t3]

    num_epochs = 200
    batch_size = 32
    indices = np.arange(n_samples)

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        x_shuff    = x_tensor.numpy()[indices]
        y_reg_shuf = y_reg_t.numpy()[indices]
        y2_shuff   = y_clf2_t.numpy()[indices]
        y3_shuff   = y_clf3_t.numpy()[indices]

        avg_losses = np.zeros(N)
        num_batches = n_samples // batch_size

        for b in range(num_batches):
            start = b * batch_size
            end   = start + batch_size
            x_batch  = tf.constant(x_shuff[start:end],   dtype=tf.float32)
            y1_batch = tf.constant(y_reg_shuf[start:end], dtype=tf.float32)
            y2_batch = tf.constant(y2_shuff[start:end],   dtype=tf.float32)
            y3_batch = tf.constant(y3_shuff[start:end],   dtype=tf.float32)

            losses = mao_adam_train_step(x_batch, y1_batch, y2_batch, y3_batch)
            for i in range(N):
                avg_losses[i] += losses[i].numpy()

        avg_losses /= num_batches
        if (epoch+1) % 20 == 0:
            print(
                f"Epoch {epoch+1:03d} | "
                f"T1 (reg) loss: {avg_losses[0]:.4f} | "
                f"T2 (clf2) loss: {avg_losses[1]:.4f} | "
                f"T3 (clf3) loss: {avg_losses[2]:.4f}"
            )

    test_size = 100
    x_test_np    = np.random.uniform(0.0, 3.0, size=(test_size, 1)).astype(np.float32)
    y_test_reg_np  = x_test_np**2
    y_test_clf2_np = (y_test_reg_np >= 2.0).astype(np.float32)
    y_test_clf3_np = (y_test_reg_np <= 3.0).astype(np.float32)

    x_test_tf     = tf.constant(x_test_np,     dtype=tf.float32)
    y_test_reg_tf = tf.constant(y_test_reg_np, dtype=tf.float32)
    y_test_c2_tf  = tf.constant(y_test_clf2_np,dtype=tf.float32)
    y_test_c3_tf  = tf.constant(y_test_clf3_np,dtype=tf.float32)

    pred_reg, pred_clf2, pred_clf3 = model(x_test_tf, training=False)

    mse_reg = tf.reduce_mean(tf.square(pred_reg - y_test_reg_tf))
    pred_classes_2 = tf.cast(pred_clf2 > 0.5, tf.float32)
    acc_clf2 = tf.reduce_mean(tf.cast(tf.equal(pred_classes_2, y_test_c2_tf), tf.float32))
    pred_classes_3 = tf.cast(pred_clf3 > 0.5, tf.float32)
    acc_clf3 = tf.reduce_mean(tf.cast(tf.equal(pred_classes_3, y_test_c3_tf), tf.float32))

    print("\n-- FINAL EVALUATION on 100 random test points --")
    print(f"Task 1 (Regression) MSE        : {mse_reg.numpy():.4f}")
    print(f"Task 2 (Classification) Accuracy: {acc_clf2.numpy():.4f}")
    print(f"Task 3 (Classification) Accuracy: {acc_clf3.numpy():.4f}")
