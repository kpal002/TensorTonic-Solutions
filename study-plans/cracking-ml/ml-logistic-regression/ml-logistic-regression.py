import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """

    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
        
    X_arr = np.array(X)
    y_arr = np.array(y)
    m = X_arr.shape[0]
    n = X_arr.shape[1]
    w = np.zeros(n)
    b = 0.0


    
    for i in range(n_iters):
        y_pred = sigmoid(np.matmul(X_arr,w) + b)
        loss = -np.mean(y_arr*np.log(y_pred) + (1-y_arr)*np.log(1 - y_pred))
        w = w - lr*np.matmul(X_arr.T,y_pred - y_arr)/m
        b = b - lr*np.mean(y_pred - y_arr)

    return w,b
        
