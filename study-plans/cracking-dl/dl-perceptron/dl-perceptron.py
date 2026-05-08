import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float).ravel()

    m, n = X_arr.shape

    weights = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        for i in range(m):
            z = np.dot(X_arr[i], weights) + b
            y_pred = 1.0 if z >= 0 else 0.0

            error = y_arr[i] - y_pred

            weights += lr * error * X_arr[i]
            b += lr * error

    return weights.tolist(), float(b)