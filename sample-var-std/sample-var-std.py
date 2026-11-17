import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x)
    x_mean = np.mean(x)
    variance = 0.0
    for _x in x:
        variance += (_x - x_mean)**2
    variance /= (len(x)-1)
    return {"variance": float(variance), "standard_deviation": float(variance**0.5)}
    pass