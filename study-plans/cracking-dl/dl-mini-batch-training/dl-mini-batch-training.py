import numpy as np


def mini_batch_training(X, y, weights, biases, lr, epochs, batch_size):
    """
    Train a fully-connected network using mini-batch gradient descent.

    Uses ReLU for hidden layers, identity on the output layer, and MSE
    loss L = ‖ŷ - y‖² / (2B). Batches are iterated in fixed order each
    epoch so results are fully deterministic.

    Args:
        X:          Input array of shape (n_samples, n_features).
        y:          Target array of shape (n_samples, n_outputs).
        weights:    List of weight matrices W_l, each shape (out_dim, in_dim).
        biases:     List of bias vectors b_l, each shape (out_dim,).
        lr:         Learning rate η.
        epochs:     Number of full passes over the data.
        batch_size: Number of samples per mini-batch.

    Returns:
        List of per-epoch MSE losses (rounded to 4 decimal places),
        computed on the full dataset after each epoch.
    """
    X       = np.asarray(X, dtype=np.float64)
    y       = np.asarray(y, dtype=np.float64)
    weights = [np.asarray(w, dtype=np.float64) for w in weights]
    biases  = [np.asarray(b, dtype=np.float64) for b in biases]

    n_layers  = len(weights)
    n_samples = X.shape[0]
    epoch_losses = []

    for _ in range(epochs):

        for start in range(0, n_samples, batch_size):
            X_b = X[start : start + batch_size]
            y_b = y[start : start + batch_size]
            B   = X_b.shape[0]

            # ── Forward pass ──────────────────────────────────────────────
            # z_l = a_{l-1} W_l^T + b_l,  a_l = ReLU(z_l) or z_l (output)
            activations     = [X_b]
            pre_activations = []
            a = X_b
            for l in range(n_layers):
                z = a @ weights[l].T + biases[l]
                pre_activations.append(z)
                a = z if l == n_layers - 1 else np.maximum(0.0, z)
                activations.append(a)

            # ── Backward pass ─────────────────────────────────────────────
            # δ_L   = (ŷ − y) / B
            # δ_l−1 = (δ_l W_l) ⊙ 𝟙[z_{l-1} > 0]
            # ∂L/∂W_l = δ_l^T a_{l-1},   ∂L/∂b_l = Σ_batch δ_l
            delta  = (activations[-1] - y_b) / B
            grad_W = [None] * n_layers
            grad_b = [None] * n_layers
            for l in reversed(range(n_layers)):
                grad_W[l] = delta.T @ activations[l]
                grad_b[l] = delta.sum(axis=0)
                if l > 0:
                    delta = delta @ weights[l]
                    delta *= pre_activations[l - 1] > 0  # ReLU derivative, in-place

            # ── Parameter update ──────────────────────────────────────────
            for l in range(n_layers):
                weights[l] -= lr * grad_W[l]
                biases[l]  -= lr * grad_b[l]

        # ── Full-dataset loss after each epoch ────────────────────────────
        a = X
        for l in range(n_layers):
            z = a @ weights[l].T + biases[l]
            a = z if l == n_layers - 1 else np.maximum(0.0, z)
        loss = np.sum((a - y) ** 2) / (2 * n_samples)
        epoch_losses.append(round(float(loss), 4))

    return epoch_losses