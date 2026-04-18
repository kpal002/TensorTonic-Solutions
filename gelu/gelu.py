import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    v_erf = np.vectorize(math.erf)
    return 0.5*x_arr*(1 + v_erf(x/np.sqrt(2)))
