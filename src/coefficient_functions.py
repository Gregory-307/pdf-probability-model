import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LinearRegression

def calculate_mean(pct_changes):
    """
    Calculate the mean percentage change (\(\mu_0\)).
    """
    return np.nanmean(pct_changes)  # Handles NaNs safely

def calculate_volatility(pct_changes):
    """
    Calculate the volatility (\(\sigma_0\)), defined as the standard deviation of percentage changes.
    """
    return np.nanstd(pct_changes)  # Handles NaNs safely

def calculate_skewness(pct_changes):
    """
    Calculate the skewness (\(\alpha\)), representing asymmetry in the distribution of percentage changes.
    """
    if isinstance(pct_changes, np.ndarray):  # If input is NumPy array
        pct_changes = pct_changes[~np.isnan(pct_changes)]  # Remove NaNs
    return skew(pct_changes)

def calculate_mean_rate(prices):
    """
    Calculate the mean rate (\(\delta\)) as the slope of a linear regression of prices over time.
    """
    time_indices = np.arange(len(prices)).reshape(-1, 1)  # Time indices
    model = LinearRegression().fit(time_indices, prices)
    return model.coef_[0]  # Slope of the regression line

def calculate_volatility_growth(pct_changes, window=7):
    """
    Calculate the volatility growth (\(\beta\)) as the slope of a linear regression of rolling volatility over time.
    """
    if isinstance(pct_changes, np.ndarray):
        pct_changes = pd.Series(pct_changes)  # Convert to Pandas Series for rolling calculation
    
    # Compute rolling volatility
    rolling_volatility = pct_changes.rolling(window=window).std().dropna()
    
    # Perform linear regression on rolling volatility
    time_indices = np.arange(len(rolling_volatility)).reshape(-1, 1)
    model = LinearRegression().fit(time_indices, rolling_volatility)
    return model.coef_[0]  # Slope of the regression line
