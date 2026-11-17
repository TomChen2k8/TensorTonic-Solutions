import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    ans = float((np.dot(a,b))/(np.linalg.norm(a) * np.linalg.norm(b)))
    return (ans if np.dot(a,b) != 0.0 else 0.0)
    pass