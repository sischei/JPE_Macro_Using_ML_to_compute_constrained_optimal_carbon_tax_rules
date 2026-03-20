# adapted from https://gist.github.com/cloneofsimo/d6edd7ad44d9bb3f33bc09098e0655f5

import tensorflow as tf
import numpy as np


def compute_activation_std(model, dataset, batch_size=32, layer_names=None):
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

    activations = {name: [] for name in layer_names}
    for inputs in dataset:
        outputs = activation_model(inputs, training=False)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for name, output in zip(layer_names, outputs):
            activations[name].append(output.numpy())

    layer_activation_std = {}
    for name in layer_names:
        act = np.concatenate(activations[name], axis=0)
        act_std = np.std(act)
        layer_activation_std[name] = act_std

    return layer_activation_std


def adjust_weight_init(
    model, dataset, batch_size=32, tol=0.2, max_iters=10, exclude_layers=None
):
    if exclude_layers is None:
        exclude_layers = []

    layers_to_adjust = []
    for layer in model.layers:
        if (
            isinstance(layer, (tf.keras.layers.Dense))
            and layer.name not in exclude_layers
        ):
            layers_to_adjust.append(layer)

    print(f"Layers to adjust: {[layer.name for layer in layers_to_adjust]}")
    initial_std = {}
    layer_weight_std = {}

    for layer in layers_to_adjust:
        print(f"Adjusting layer: {layer.name}")

        weights = layer.get_weights()[0]
        initial_std[layer.name] = np.std(weights)

        fan_in = np.prod(weights.shape[:-1]) 
        weight_std = np.sqrt(1 / fan_in)

        for i in range(max_iters):
            new_initializer = tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=weight_std
            )
            layer.set_weights([new_initializer(weights.shape), layer.get_weights()[1]])
            activation_std = compute_activation_std(
                model, dataset, batch_size, layer_names=[layer.name]
            )[layer.name]
            print(f"Iteration {i + 1}: Activation std = {activation_std:.4f}")

            if abs(activation_std - 1.0) < tol:
                print(
                    f"Layer {layer.name} achieved near unit activation of {activation_std:.4f} with weight std = {weight_std:.4f}"
                )
                layer_weight_std[layer.name] = weight_std / activation_std
                break
            else:
                weight_std = weight_std / activation_std
        else:
            print(f"Layer {layer.name} did not converge within {max_iters} iterations.")
            layer_weight_std[layer.name] = weight_std

    return initial_std, layer_weight_std
