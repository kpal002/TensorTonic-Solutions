import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    loss = - np.mean(np.log(y_pred_arr[np.arange(len(y_true_arr)), y_true_arr]))
    return loss    