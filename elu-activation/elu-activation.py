import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    return np.where(x_arr>0,x_arr,alpha*(np.exp(x_arr)-1)).tolist()