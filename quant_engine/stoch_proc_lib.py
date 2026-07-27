# quant_engine/stoch_proc_lib.py
import numpy as np
import pandas as pd
from . import stats_lib

def brownian_motion(T: float, N: int, dt: float) -> np.ndarray:
    """Generate a Brownian Motion path."""
    dW = stats_lib.sample_normal(0.0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    # Start at 0
    W = np.insert(W, 0, 0)
    return W

def geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, N: int, dt: float) -> np.ndarray:
    """Generate a Geometric Brownian Motion path."""
    t = np.linspace(0, T, N+1)
    W = brownian_motion(T, N, dt)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S

def random_walk(start_price: float, steps: int, step_size: float) -> np.ndarray:
    """Generate a simple random walk path."""
    increments = stats_lib.sample_normal(0.0, step_size, steps)
    path = np.concatenate(([start_price], start_price + np.cumsum(increments)))
    return path

# Add more stochastic processes as needed...
