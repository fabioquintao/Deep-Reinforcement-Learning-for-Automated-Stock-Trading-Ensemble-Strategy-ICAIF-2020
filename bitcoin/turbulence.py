import numpy as np
import pandas as pd


def add_turbulence(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Mahalanobis-distance turbulence index for a single asset (uses rolling vol)."""
    df = df.copy()
    returns = df["close"].pct_change().fillna(0)
    rolling_mean = returns.rolling(lookback).mean()
    rolling_std  = returns.rolling(lookback).std()

    turbulence = np.where(
        rolling_std > 0,
        ((returns - rolling_mean) / rolling_std) ** 2,
        0.0,
    )
    df["turbulence"] = turbulence
    df["turbulence"] = df["turbulence"].fillna(0)
    return df
