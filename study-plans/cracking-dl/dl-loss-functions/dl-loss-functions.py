import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred, dtype=float)

    eps = 1e-15

    if loss_type == "mse":
        loss = np.mean((y_true_arr - y_pred_arr) ** 2)

    elif loss_type == "bce":
        y_pred_safe = np.clip(y_pred_arr, eps, 1 - eps)
        loss = -np.mean(
            y_true_arr * np.log(y_pred_safe)
            + (1 - y_true_arr) * np.log(1 - y_pred_safe)
        )

    elif loss_type == "cce":
        y_true_indices = y_true_arr.astype(int)

        shifted = y_pred_arr - np.max(y_pred_arr, axis=1, keepdims=True)

        correct_logits = shifted[np.arange(len(y_true_indices)), y_true_indices]
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))

        loss = -np.mean(correct_logits - log_sum_exp)

    else:
        loss = np.mean(np.maximum(0, 1 - y_true_arr * y_pred_arr))

    return round(float(loss), 4)