"""Rolling window coefficient extractor."""

from typing import Any
import pandas as pd
import numpy as np

from .config import ExtractionConfig
from .functions import (
    calculate_mean,
    calculate_volatility,
    calculate_skewness,
    calculate_mean_rate,
    calculate_volatility_growth,
)
from ..core.parameters import GeneralizedLaplaceParameters


class RollingCoefficientExtractor:
    """
    Extracts distribution coefficients using rolling window calculations.

    This extractor computes the five core coefficients needed for
    the GeneralizedLaplace distribution from time series data:

    - mu_0 (mean): Average percentage/fractional change
    - sigma_0 (volatility): Standard deviation of changes
    - alpha (skewness): Asymmetry in the distribution
    - delta (mean_rate): Trend in the underlying values
    - beta (volatility_growth): Rate of volatility expansion

    The extractor is data-agnostic - it uses the ExtractionConfig to
    determine which columns to use and how to process the data.

    Example:
        >>> extractor = RollingCoefficientExtractor()
        >>> config = ExtractionConfig(
        ...     value_column="close",
        ...     group_column="ticker",
        ...     horizon=60
        ... )
        >>> result = extractor.extract(stock_data, config)
        >>> print(result[['mean', 'volatility', 'skewness']].head())
    """

    # Column names added by extraction
    COEFFICIENT_COLUMNS = [
        "pct_change",
        "mean",
        "volatility",
        "skewness",
        "mean_rate",
        "volatility_growth",
    ]

    def extract(
        self,
        data: pd.DataFrame,
        config: ExtractionConfig,
    ) -> pd.DataFrame:
        """
        Extract coefficients with rolling windows.

        Args:
            data: DataFrame containing time series data
            config: Extraction configuration

        Returns:
            DataFrame with added coefficient columns

        Raises:
            ValueError: If required columns are missing
        """
        if config.value_column not in data.columns:
            raise ValueError(f"Column '{config.value_column}' not found in DataFrame")

        df = data.copy()

        # Handle grouping
        if config.group_column and config.group_column in df.columns:
            result = df.groupby(config.group_column, group_keys=False).apply(
                lambda g: self._extract_single(g, config)
            )
            return result.reset_index(drop=True) if isinstance(result.index, pd.MultiIndex) else result
        else:
            return self._extract_single(df, config)

    def _extract_single(
        self,
        data: pd.DataFrame,
        config: ExtractionConfig,
    ) -> pd.DataFrame:
        """Extract coefficients for a single group."""
        df = data.copy()
        col = config.value_column
        h = config.horizon

        # Calculate percentage/fractional changes
        df["pct_change"] = df[col].pct_change() * config.pct_change_multiplier

        # Rolling coefficient calculations
        df["mean"] = df["pct_change"].rolling(window=h).apply(
            calculate_mean, raw=True
        )
        df["volatility"] = df["pct_change"].rolling(window=h).apply(
            calculate_volatility, raw=True
        )
        df["skewness"] = df["pct_change"].rolling(window=h).apply(
            calculate_skewness, raw=True
        )
        df["mean_rate"] = df[col].rolling(window=h).apply(
            calculate_mean_rate, raw=False
        )
        df["volatility_growth"] = df["pct_change"].rolling(window=h).apply(
            lambda x: calculate_volatility_growth(
                pd.Series(x), window=config.volatility_window
            ),
            raw=False,
        )

        # Handle NaN values
        if config.dropna:
            df = df.dropna(subset=self.COEFFICIENT_COLUMNS)

        return df

    def extract_single_window(
        self,
        values: pd.Series | np.ndarray,
        config: ExtractionConfig,
    ) -> dict[str, float]:
        """
        Extract coefficients from a single window of data.

        Useful when you have a fixed window of historical data
        and want to compute coefficients without rolling.

        Args:
            values: Series or array of values
            config: Extraction configuration

        Returns:
            Dictionary of coefficient values
        """
        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        # Calculate percentage changes
        pct_changes = values.pct_change() * config.pct_change_multiplier
        pct_changes = pct_changes.dropna()

        return {
            "mean": calculate_mean(pct_changes),
            "volatility": calculate_volatility(pct_changes),
            "skewness": calculate_skewness(pct_changes),
            "mean_rate": calculate_mean_rate(values),
            "volatility_growth": calculate_volatility_growth(
                pct_changes, window=config.volatility_window
            ),
        }

    def to_parameters(
        self,
        coefficients: dict[str, float],
        k: float = 1.0,
        lambda_decay: float = 0.0,
    ) -> GeneralizedLaplaceParameters:
        """
        Convert extracted coefficients to distribution parameters.

        Args:
            coefficients: Dictionary with keys: mean, volatility, skewness, mean_rate, volatility_growth
            k: Tail sharpness parameter (default 1.0)
            lambda_decay: Time decay parameter (default 0.0)

        Returns:
            GeneralizedLaplaceParameters instance
        """
        return GeneralizedLaplaceParameters(
            mu_0=coefficients["mean"],
            sigma_0=max(coefficients["volatility"], 1e-6),  # Ensure positive
            alpha=coefficients["skewness"],
            delta=coefficients["mean_rate"],
            beta=coefficients["volatility_growth"],
            k=k,
            lambda_decay=lambda_decay,
        )

    def extract_and_convert(
        self,
        values: pd.Series | np.ndarray,
        config: ExtractionConfig,
        k: float = 1.0,
        lambda_decay: float = 0.0,
    ) -> GeneralizedLaplaceParameters:
        """
        Extract coefficients and convert to parameters in one step.

        Convenience method that combines extract_single_window and to_parameters.

        Args:
            values: Series or array of values
            config: Extraction configuration
            k: Tail sharpness parameter
            lambda_decay: Time decay parameter

        Returns:
            GeneralizedLaplaceParameters instance
        """
        coefficients = self.extract_single_window(values, config)
        return self.to_parameters(coefficients, k=k, lambda_decay=lambda_decay)

    @staticmethod
    def coefficients_from_row(
        row: pd.Series | dict[str, Any],
        k: float = 1.0,
        lambda_decay: float = 0.0,
    ) -> GeneralizedLaplaceParameters:
        """
        Create parameters from a DataFrame row with coefficient columns.

        Useful when you have already extracted coefficients into a DataFrame
        and want to convert individual rows to parameters.

        Args:
            row: Series or dict with coefficient columns
            k: Tail sharpness parameter
            lambda_decay: Time decay parameter

        Returns:
            GeneralizedLaplaceParameters instance
        """
        return GeneralizedLaplaceParameters(
            mu_0=float(row["mean"]),
            sigma_0=max(float(row["volatility"]), 1e-6),
            alpha=float(row["skewness"]),
            delta=float(row["mean_rate"]),
            beta=float(row["volatility_growth"]),
            k=k,
            lambda_decay=lambda_decay,
        )
