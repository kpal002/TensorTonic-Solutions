import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    if activation == "relu":

        value = max(0, x)

        derivative = 1.0 if x > 0 else 0.0

        return float(value), derivative

    elif activation == "leaky_relu":

        value = x if x > 0 else 0.01*x

        derivative = 1.0 if x > 0 else 0.01

        return value, derivative

    elif activation == "sigmoid":

        value = 1.0/(1.0 + np.exp(-x))

        derivative = value * (1-value)

        return value, derivative
    
    elif activation == "tanh":

        value = (np.exp(x) - np.exp(-x))/(np.exp(x)+ np.exp(-x))

        derivative = 1 - value**2

        return value, derivative

    elif activation == "gelu":
        a = np.sqrt(2 / np.pi)

        u = a * (x + 0.044715 * x**3)
        
        value = 0.5 * x * (1 + np.tanh(u))

        derivative = 0.5 * (1 + np.tanh(u)) + 0.5 * x * (1 - np.tanh(u)**2) * a * (1 + 3 * 0.044715 * x**2)

        return value, derivative

    elif activation == "swish":
        sig = 1.0/(1.0 + np.exp(-x))
        value = x * sig

        derivative = sig + x*sig*(1-sig)

        return value, derivative

    