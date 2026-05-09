import numpy as np

def mini_batch_training(X, y, weights, biases, lr, epochs, batch_size):
    """
    Returns: list of floats
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    weights = [np.array(w, dtype=float) for w in weights]
    biases = [np.array(b, dtype=float) for b in biases]

    n_layers = len(weights)
    n_samples = X.shape[0]

    epoch_losses = []

    for _ in range(epochs):
        for start in range(0, n_samples, batch_size):
            end = start + batch_size

            X_batch = X[start:end]
            y_batch = y[start:end]

            batch_n = X_batch.shape[0]

            # --------------------
            # Forward pass
            # --------------------
            activations = [X_batch]
            pre_activations = []

            a = X_batch
            for l in range(n_layers):

                z = a @ weights[l].T + biases[l]
                pre_activations.append(z)
                if l == n_layers - 1:
                    # output layer: identity
                    a = z
                else:
                    # hidden layers: ReLU
                    a = np.maximum(0, z)

                activations.append(a)

            y_pred = activations[-1]

            # --------------------
            # Backward pass
            # --------------------

            # MSE loss: L = 1/(2B) * sum ||y_pred - y||^2
            # dL/dy_pred = (y_pred - y) / B

            delta = (y_pred - y_batch) / batch_n
            grad_weights = [None] * n_layers
            grad_biases = [None] * n_layers

            for l in reversed(range(n_layers)):
                # z_l = a_{l-1} @ W_l.T + b_l
                # dW_l shape should match W_l: (out_dim, in_dim)

                grad_weights[l] = delta.T @ activations[l]
                grad_biases[l] = np.sum(delta, axis=0)
                if l > 0:

                    delta = delta @ weights[l]

                    # derivative of ReLU
                    relu_derivative = (pre_activations[l - 1] > 0).astype(float)
                    delta = delta * relu_derivative

            # --------------------
            # Parameter update
            # --------------------
            for l in range(n_layers):

                weights[l] -= lr * grad_weights[l]
                biases[l] -= lr * grad_biases[l]
        # --------------------
        # Full-dataset loss after epoch
        # --------------------

        a = X
        for l in range(n_layers):
            z = a @ weights[l].T + biases[l]
            if l == n_layers - 1:
                a = z
            else:
                a = np.maximum(0, z)

        full_pred = a
        loss = np.sum((full_pred - y) ** 2) / (2 * n_samples)
        epoch_losses.append(round(float(loss), 4))

    return epoch_losses
            
        

        