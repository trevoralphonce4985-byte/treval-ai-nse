# quant_engine/stats_lib.py
import numpy as np
import pandas as pd
from scipy import stats
import math

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from a series of prices."""
    return np.log(prices / prices.shift(1)).dropna()

def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from a series of prices."""
    return (prices / prices.shift(1) - 1).dropna()

def calculate_mean(data: pd.Series) -> float:
    """Calculate the mean of a series."""
    return data.mean()

def calculate_variance(data: pd.Series) -> float:
    """Calculate the sample variance of a series."""
    return data.var(ddof=1) # ddof=1 for sample variance

def calculate_std_dev(data: pd.Series) -> float:
    """Calculate the sample standard deviation of a series."""
    return data.std(ddof=1) # ddof=1 for sample std dev

def calculate_correlation(x: pd.Series, y: pd.Series) -> float:
    """Calculate the Pearson correlation coefficient between two series."""
    return x.corr(y)

def calculate_covariance(x: pd.Series, y: pd.Series) -> float:
    """Calculate the sample covariance between two series."""
    return x.cov(y)

def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate the simple moving average."""
    return data.rolling(window=window).mean()

def calculate_exponential_moving_average(data: pd.Series, span: int) -> pd.Series:
    """Calculate the exponential moving average."""
    return data.ewm(span=span).mean()

def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
    """Calculate the Z-score for a value given mean and std dev."""
    if std_dev == 0:
        return 0.0 # Avoid division by zero
    return (value - mean) / std_dev

def sample_normal(mean: float = 0.0, std_dev: float = 1.0, size: int = 1) -> np.ndarray:
    """Sample from a normal distribution."""
    return np.random.normal(loc=mean, scale=std_dev, size=size)

# Add more statistical functions as needed...
