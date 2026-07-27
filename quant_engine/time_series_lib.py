# quant_engine/time_series_lib.py
import pandas as pd
import numpy as np
from scipy import stats # For potential stationarity tests if using scipy

def calculate_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Calculate the autocorrelation at a specific lag."""
    return series.autocorr(lag=lag)

def calculate_partial_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Calculate the partial autocorrelation at a specific lag (placeholder using correlation of residuals)."""
    # This is a simplified approximation. Proper PACF calculation is more complex (e.g., using Yule-Walker equations).
    if lag <= 1:
        return calculate_autocorrelation(series, lag)
    # Fit AR(lag-1) model to y[:-1] and y[1:]
    # Fit AR(lag-1) model to y[:-1] and y[1:] shifted by lag
    # Calculate residuals
    # Correlate residuals
    # For now, just return the regular autocorrelation as a fallback
    print(f"Warning: Simplified PACF calculation used for lag {lag}. Consider using statsmodels for precise PACF.")
    return calculate_autocorrelation(series, lag)

def difference_series(series: pd.Series, order: int = 1) -> pd.Series:
    """Difference a series a specified number of times."""
    diff_series = series.diff(periods=order).dropna()
    return diff_series

def calculate_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Calculate the rolling mean."""
    return series.rolling(window=window).mean()

def calculate_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Calculate the rolling standard deviation."""
    return series.rolling(window=window).std()

def calculate_lagged_values(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    """Calculate multiple lagged versions of a series."""
    df = pd.DataFrame()
    for lag in lags:
         df[f'lag_{lag}'] = series.shift(lag)
    return df.dropna()

# Placeholder for stationarity tests (would typically use statsmodels)
def adf_test(series: pd.Series) -> Dict:
    """Augmented Dickey-Fuller test (placeholder - requires statsmodels)."""
    # from statsmodels.tsa.stattools import adfuller
    # result = adfuller(series)
    # return {"statistic": result[0], "p_value": result[1], "is_stationary": result[1] < 0.05}
    print("ADF Test requires statsmodels. Skipping.")
    return {"statistic": None, "p_value": None, "is_stationary": None}


# Add more time series functions as needed...
