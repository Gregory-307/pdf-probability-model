"""Time series decomposition utilities."""

from typing import Any
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft
import pywt


def decompose_stl(
    data: pd.DataFrame,
    column: str,
    seasonal_period: int | None = None,
    robust: bool = False,
) -> pd.DataFrame:
    """
    Decompose a time series using STL (Seasonal-Trend decomposition using LOESS).

    STL decomposes a time series into trend, seasonal, and residual components.
    When seasonal_period is None, only trend and residual are computed.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to decompose
        seasonal_period: Period for seasonal component (None to disable)
        robust: Use robust fitting (less sensitive to outliers)

    Returns:
        DataFrame with added 'trend' and 'residual' columns

    Raises:
        ValueError: If column not found or insufficient data
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    df = data.copy()
    series = df[column].values

    if seasonal_period is not None and len(series) < seasonal_period * 2:
        raise ValueError(
            f"Insufficient data for STL: need at least {seasonal_period * 2} points"
        )

    stl = STL(series, period=seasonal_period, robust=robust) if seasonal_period else STL(series, robust=robust)
    result = stl.fit()

    df["trend"] = result.trend
    df["residual"] = result.resid

    return df


def decompose_stl_with_seasonality(
    data: pd.DataFrame,
    column: str,
    seasonal_period: int = 7,
    robust: bool = False,
) -> pd.DataFrame:
    """
    Decompose a time series using STL with seasonal component.

    Returns trend, seasonal, and residual components.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to decompose
        seasonal_period: Period for seasonal component (default 7 for weekly)
        robust: Use robust fitting

    Returns:
        DataFrame with 'trend', 'seasonal', and 'residual' columns

    Raises:
        ValueError: If column not found or insufficient data
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    df = data.copy()
    series = df[column].values

    if len(series) < seasonal_period * 2:
        raise ValueError(
            f"Insufficient data for STL: need at least {seasonal_period * 2} points"
        )

    stl = STL(series, period=seasonal_period, robust=robust)
    result = stl.fit()

    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["residual"] = result.resid

    return df


def decompose_fourier(
    data: pd.DataFrame,
    column: str,
    n_components: int | None = None,
) -> pd.DataFrame:
    """
    Decompose a time series using Fourier Transform.

    The Fourier Transform converts the signal to frequency domain,
    revealing periodic components.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to decompose
        n_components: Number of frequency components to keep (None for all)

    Returns:
        DataFrame with 'fft_magnitude' and 'fft_phase' columns,
        plus 'fft_reconstructed' if n_components is specified

    Raises:
        ValueError: If column not found or insufficient data
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    df = data.copy()
    signal = df[column].values

    if len(signal) < 2:
        raise ValueError("Insufficient data: need at least 2 points")

    fourier_transform = fft(signal)

    df["fft_magnitude"] = np.abs(fourier_transform)
    df["fft_phase"] = np.angle(fourier_transform)

    # Optionally reconstruct with limited components
    if n_components is not None:
        ft_filtered = fourier_transform.copy()
        # Keep only top n_components (symmetric for real signal)
        n = len(ft_filtered)
        mask = np.zeros(n, dtype=bool)
        mask[:n_components] = True
        mask[-(n_components - 1):] = True
        ft_filtered[~mask] = 0
        from scipy.fft import ifft
        df["fft_reconstructed"] = np.real(ifft(ft_filtered))

    return df


def decompose_wavelet(
    data: pd.DataFrame,
    column: str,
    wavelet: str = "db4",
    level: int = 1,
) -> pd.DataFrame:
    """
    Decompose a time series using Wavelet Transform.

    Wavelet decomposition captures both frequency and time information,
    making it useful for non-stationary signals.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to decompose
        wavelet: Wavelet type (default 'db4' - Daubechies 4)
        level: Decomposition level (higher = more detailed)

    Returns:
        DataFrame with wavelet coefficient columns (wavelet_approx, wavelet_detail_N)

    Raises:
        ValueError: If column not found or insufficient data

    Note:
        Common wavelets: 'haar', 'db4', 'sym5', 'coif3'
        Higher levels require more data points.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    df = data.copy()
    signal = df[column].values

    max_level = pywt.dwt_max_level(len(signal), wavelet)
    if level > max_level:
        raise ValueError(
            f"Level {level} too high for data length {len(signal)}. Max level: {max_level}"
        )

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # First coefficient is approximation, rest are details
    # Pad coefficients to match original length for DataFrame storage
    for i, coeff in enumerate(coeffs):
        if i == 0:
            col_name = "wavelet_approx"
        else:
            col_name = f"wavelet_detail_{i}"

        # Upsample to match original length (approximate)
        if len(coeff) != len(signal):
            ratio = len(signal) / len(coeff)
            indices = np.floor(np.arange(len(signal)) / ratio).astype(int)
            indices = np.clip(indices, 0, len(coeff) - 1)
            df[col_name] = coeff[indices]
        else:
            df[col_name] = coeff

    return df


def decompose_moving_average(
    data: pd.DataFrame,
    column: str,
    window: int = 7,
    center: bool = True,
) -> pd.DataFrame:
    """
    Decompose a time series using simple moving average.

    The simplest decomposition: trend as smoothed series, residual as remainder.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to decompose
        window: Moving average window size
        center: Whether to center the window (default True)

    Returns:
        DataFrame with 'moving_average' and 'residual' columns

    Raises:
        ValueError: If column not found
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    df = data.copy()

    df["moving_average"] = df[column].rolling(
        window=window, min_periods=1, center=center
    ).mean()
    df["residual"] = df[column] - df["moving_average"]

    return df


def decompose_exponential_smoothing(
    data: pd.DataFrame,
    column: str,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """
    Decompose a time series using exponential smoothing.

    Exponential smoothing gives more weight to recent observations.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to decompose
        alpha: Smoothing factor (0 < alpha <= 1). Higher = more responsive.

    Returns:
        DataFrame with 'exp_smooth' and 'residual' columns

    Raises:
        ValueError: If column not found or invalid alpha
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be between 0 (exclusive) and 1 (inclusive)")

    df = data.copy()

    df["exp_smooth"] = df[column].ewm(alpha=alpha, adjust=False).mean()
    df["residual"] = df[column] - df["exp_smooth"]

    return df


def get_dominant_frequencies(
    data: pd.DataFrame,
    column: str,
    n_frequencies: int = 5,
    sampling_rate: float = 1.0,
) -> list[dict[str, float]]:
    """
    Get the dominant frequencies from FFT analysis.

    Useful for identifying periodic patterns in the data.

    Args:
        data: Input DataFrame containing the time series
        column: Column name for the series to analyze
        n_frequencies: Number of top frequencies to return
        sampling_rate: Samples per time unit (e.g., 1.0 for daily data)

    Returns:
        List of dicts with 'frequency', 'period', and 'amplitude' keys

    Raises:
        ValueError: If column not found
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    signal = data[column].values
    n = len(signal)

    fourier_transform = fft(signal)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)
    amplitudes = np.abs(fourier_transform)

    # Only consider positive frequencies (first half, excluding DC)
    positive_mask = frequencies > 0
    pos_frequencies = frequencies[positive_mask]
    pos_amplitudes = amplitudes[positive_mask]

    # Get top n frequencies by amplitude
    top_indices = np.argsort(pos_amplitudes)[-n_frequencies:][::-1]

    results = []
    for idx in top_indices:
        freq = pos_frequencies[idx]
        results.append({
            "frequency": float(freq),
            "period": float(1 / freq) if freq > 0 else float("inf"),
            "amplitude": float(pos_amplitudes[idx]),
        })

    return results
