import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    stats = {}
    stats['mean'] = np.mean(x)
    stats['median'] = np.median(x)

    values, counts = np.unique(x, return_counts=True)
    stats['mode'] = values[np.argmax(counts)]
    
    return stats