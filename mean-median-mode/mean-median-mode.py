from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    x = np.asarray(x)
    tmp = Counter(x).most_common(1)[0]
    return { "mean": float(np.mean(x)), "median": float(np.median(x)), "mode": float(tmp[0])}
    pass