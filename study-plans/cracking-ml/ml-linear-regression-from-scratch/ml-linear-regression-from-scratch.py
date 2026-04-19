import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X_arr = np.array(X)
    y_arr = np.array(y)
    n = X_arr.shape[1] 
    m = X_arr.shape[0]
    w = np.zeros(n)
    b = 0.0

    for i in range(epochs):
        y_pred = np.matmul(X_arr,w) + b
        loss = np.mean(np.square(y_pred - y_arr))
        w = w - lr*2*np.matmul(X_arr.T,y_pred - y_arr)/m
        b = b - lr*2*np.mean(y_pred - y_arr)

    return w,b
