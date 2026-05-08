import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X_arr = np.array(X, dtype=float)
    mask_arr = np.array(mask)
    
    if mode == "test":
        return X_arr
    else:
        
        return X_arr * mask_arr / (1 - drop_prob)
    