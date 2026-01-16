import pandas as pd
import numpy as np
from coefficient_functions import (
    calculate_mean,
    calculate_volatility,
    calculate_skewness,
    calculate_mean_rate,
    calculate_volatility_growth
)

def calculate_observed_coefficients(data, column, horizon=60):
    """
    Calculate observed coefficients over a rolling horizon for a given column in a DataFrame.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data.
        column (str): The column to calculate percentage changes and coefficients for (e.g., 'price').
        horizon (int): The rolling window size (e.g., 60 days).
    
    Returns:
        pd.DataFrame: DataFrame with observed coefficients added as new columns.
    """
    # Ensure the column exists in the DataFrame
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Calculate percentage changes
    data['pct_change'] = data[column].pct_change() * 100
    
    # Define rolling functions for each coefficient
    data['mean'] = data['pct_change'].rolling(window=horizon).apply(calculate_mean, raw=True)
    data['volatility'] = data['pct_change'].rolling(window=horizon).apply(calculate_volatility, raw=True)
    data['skewness'] = data['pct_change'].rolling(window=horizon).apply(calculate_skewness, raw=True)
    data['mean_rate'] = data[column].rolling(window=horizon).apply(calculate_mean_rate, raw=False)
    data['volatility_growth'] = data['pct_change'].rolling(window=horizon).apply(
        lambda x: calculate_volatility_growth(pd.Series(x)), raw=False
    )
    
    # Drop NA values created by rolling windows
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    # Example usage
    example_data = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "close": [100, 102, 101, 105, 110, 108, 112, 115, 118, 120]
    })

    try:
        processed_data = calculate_observed_coefficients(example_data, column="close", horizon=5)
        print("Processed Data:")
        print(processed_data)
    except Exception as e:
        print(f"Error: {e}")
