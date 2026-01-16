from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, ifft
import pywt
import pandas as pd
import numpy as np

def decompose_stl(data, column="Close", seasonal_period=None):
    """
    Decompose a time series using STL with optional seasonal component.

    Parameters:
        data (pd.DataFrame): Input dataframe containing the time series data.
        column (str): Column name for the time series to decompose.
        seasonal_period (int, optional): Seasonal period for the decomposition. If None, disables seasonal component.

    Returns:
        pd.DataFrame: Dataframe with added columns for 'trend' and 'residual'.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    if seasonal_period is not None and len(data) < seasonal_period:
        raise ValueError(f"Insufficient data for STL decomposition: requires at least {seasonal_period} points.")

    stl = STL(data[column], period=seasonal_period) if seasonal_period else STL(data[column])
    result = stl.fit()

    data["trend"] = result.trend
    data["residual"] = result.resid

    return data

def decompose_stl_with_seasonality(data, column="Close", seasonal_period=7):
    """
    Decompose a time series using STL with a fixed seasonal component (under construction).

    Parameters:
        data (pd.DataFrame): Input dataframe containing the time series data.
        column (str): Column name for the time series to decompose.
        seasonal_period (int): Seasonal period for the decomposition. Default is 7.

    Returns:
        pd.DataFrame: Dataframe with added columns for 'trend', 'seasonal', and 'residual'.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    if len(data) < seasonal_period:
        raise ValueError(f"Insufficient data for STL decomposition: requires at least {seasonal_period} points.")

    stl = STL(data[column], period=seasonal_period)
    result = stl.fit()

    data["trend"] = result.trend
    data["seasonal"] = result.seasonal
    data["residual"] = result.resid

    return data

def decompose_fourier(data, column="Close"):
    """
    Decompose a time series using Fourier Transform.

    Parameters:
        data (pd.DataFrame): Input dataframe containing the time series data.
        column (str): Column name for the time series to decompose.

    Returns:
        pd.DataFrame: Dataframe with added columns for 'real_part' and 'imag_part'.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    if len(data) < 2:
        raise ValueError("Insufficient data for Fourier decomposition: requires at least 2 points.")

    signal = data[column].values
    fourier_transform = fft(signal)

    data["real_part"] = np.real(fourier_transform)
    data["imag_part"] = np.imag(fourier_transform)

    return data

def decompose_wavelet(data, column="Close", wavelet="db4", level=1):
    """
    Decompose a time series using Wavelet Transform.

    Parameters:
        data (pd.DataFrame): Input dataframe containing the time series data.
        column (str): Column name for the time series to decompose.
        wavelet (str): Type of wavelet to use for decomposition. Default is 'db4'.
        level (int): Level of decomposition. Default is 1.

    Returns:
        pd.DataFrame: Dataframe with added columns for wavelet approximation and details.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    if len(data) < pywt.dwt_max_level(len(data), wavelet):
        raise ValueError("Insufficient data for Wavelet decomposition: data length is too short for the specified level.")

    signal = data[column].values
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Add wavelet coefficients to the dataframe
    for i, coeff in enumerate(coeffs):
        data[f"wavelet_level_{i}"] = coeff

    return data

def decompose_moving_average(data, column="Close", window=7):
    """
    Decompose a time series using a simple moving average.

    Parameters:
        data (pd.DataFrame): Input dataframe containing the time series data.
        column (str): Column name for the time series to decompose.
        window (int): Moving average window size. Default is 7.

    Returns:
        pd.DataFrame: Dataframe with added columns for 'moving_average' and 'residual'.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    if data["date"].max() - data["date"].min() < pd.Timedelta(days=window):
        raise ValueError(f"Insufficient data for moving average decomposition: requires at least {window} days.")

    data["moving_average"] = data[column].rolling(window=window, min_periods=1).mean()
    data["residual"] = data[column] - data["moving_average"]

    return data

if __name__ == "__main__":
    example_data = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
        "Close": np.random.rand(30) * 100
    })

    print("STL Decomposition (without seasonality):")
    try:
        print(decompose_stl(example_data.copy(), column="Close", seasonal_period=None).head())
    except ValueError as e:
        print(f"Error: {e}")

    print("STL Decomposition (with seasonality):")
    try:
        print(decompose_stl_with_seasonality(example_data.copy(), column="Close", seasonal_period=7).head())
    except ValueError as e:
        print(f"Error: {e}")
