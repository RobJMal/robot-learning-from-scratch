import numpy as np

DIV_ZERO_GUARD = 1e-6

def normalize(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std + DIV_ZERO_GUARD)

def unnormalize(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    return z * (std + DIV_ZERO_GUARD)+ mean