import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    sigma = 1.0/(1.0 + np.exp(-x_arr))
    return x_arr*sigma