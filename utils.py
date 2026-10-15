import glob
import numpy as np
import pandas as pd

DIV_ZERO_GUARD = 1e-6

def normalize(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std + DIV_ZERO_GUARD)

def unnormalize(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    return z * (std + DIV_ZERO_GUARD)+ mean

def load_dataframe(pattern: str) -> pd.DataFrame:
    print(f"[load_dataframe] Reading parquet files matching {pattern}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern}")
    df = pd.read_parquet(files[0])
    print(f"[load_dataframe] Loaded {files[0]}: {df.shape[0]} rows, "
          f"{df['episode_index'].nunique()} episodes")
    return df