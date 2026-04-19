import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    m = X_arr.shape[0]
    n = X_arr.shape[1]
    w = np.zeros((n,n_classes))
    b = np.zeros(n_classes)
    
    Y = np.zeros((m, n_classes))
    Y[np.arange(m), y_arr] = 1

    
    for i in range(n_iters):
        y_pred = softmax(np.matmul(X_arr,w) + b)
        w = w - lr*np.matmul(X_arr.T,y_pred - Y)/m
        b = b - lr*np.mean(y_pred - Y, axis =0)

    return w,b