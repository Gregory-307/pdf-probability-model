"""Configuration for coefficient extraction."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ExtractionConfig:
    """
    Configuration for coefficient extraction from time series data.

    This is the key abstraction that makes the library data-agnostic.
    Users specify their column names and parameters, removing all
    hardcoded assumptions about data structure.

    Attributes:
        value_column: Column containing values to analyze (e.g., 'price', 'temperature')
        time_column: Column containing timestamps (optional, uses index if None)
        group_column: Column for grouping data (optional, e.g., 'ticker', 'sensor_id')
        horizon: Rolling window size for coefficient calculation (number of periods)
        volatility_window: Inner window for volatility growth calculation
        pct_change_multiplier: Multiplier for percentage changes (100 for %, 1 for fractions)
        dropna: Whether to drop rows with NaN values after calculation

    Example:
        >>> # Stock data configuration
        >>> stock_config = ExtractionConfig(
        ...     value_column="close",
        ...     time_column="date",
        ...     group_column="ticker",
        ...     horizon=60
        ... )
        >>>
        >>> # Sensor data configuration
        >>> sensor_config = ExtractionConfig(
        ...     value_column="temperature",
        ...     time_column="timestamp",
        ...     group_column="sensor_id",
        ...     horizon=24,
        ...     pct_change_multiplier=1.0  # Use fractions, not percentages
        ... )
    """

    value_column: str
    time_column: str | None = None
    group_column: str | None = None
    horizon: int = 60
    volatility_window: int = 7
    pct_change_multiplier: float = 100.0
    dropna: bool = True
    nan_strategy: Literal["drop", "keep"] = "drop"

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.volatility_window >= self.horizon:
            raise ValueError("volatility_window must be less than horizon")
