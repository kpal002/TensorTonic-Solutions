import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim > 1:
        return np.exp(x_arr -np.max(x_arr,axis=1,keepdims=True))/np.sum(np.exp(x_arr -np.max(x_arr,axis=1,keepdims=True)), axis = 1,keepdims=True)

    else:
        return np.exp(x_arr -np.max(x_arr))/np.sum(np.exp(x_arr -np.max(x_arr)))