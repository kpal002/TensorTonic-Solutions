import numpy as np

def forward_pass(x, weights, biases):
    """
    Returns: Dict with "activations" and "pre_activations", values rounded to 4 decimals.
    """
    a = np.array(x, dtype=float)

    activations = [np.round(a, 4).tolist()]
    pre_activations = []

    num_layers = len(weights)

    for layer in range(num_layers):
        W = np.array(weights[layer], dtype=float)
        b = np.array(biases[layer], dtype=float)

        z = W @ a + b

        pre_activations.append(np.round(z, 4).tolist())

        # ReLU for hidden layers, linear for output layer
        if layer == num_layers - 1:
            a = z
        else:
            a = np.maximum(0, z)

        activations.append(np.round(a, 4).tolist())

    return {
        "activations": activations,
        "pre_activations": pre_activations
    }