"""Parameter tracking over rolling windows."""

from dataclasses import dataclass, fields
from typing import Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...utilities import fit


@dataclass
class ParameterTracker:
    """
    Track distribution parameters over rolling windows.

    Fits a distribution at each time step using a rolling window,
    producing a time series of parameter values.

    Args:
        distribution: Which distribution to fit ('nig', 'student_t', 'normal', 'skew_normal')
        window: Number of observations in the rolling window
        step: Step size between fits (1 = fit at every observation)
        min_window: Minimum window size to start fitting (default = window)

    Example:
        >>> tracker = ParameterTracker(distribution='nig', window=60, step=1)
        >>> param_history = tracker.fit(returns)
        >>> # param_history is DataFrame with columns: position, mu, delta, alpha, beta
    """
    distribution: Literal["nig", "student_t", "normal", "skew_normal"]
    window: int
    step: int = 1
    min_window: int | None = None

    def fit(
        self,
        data: NDArray[np.float64],
        index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Roll through data, fitting distribution at each step.

        Args:
            data: Array of observations (e.g., returns)
            index: Optional datetime index for the data

        Returns:
            DataFrame with columns for each parameter, indexed by position/date.
        """
        data = np.asarray(data)
        min_w = self.min_window if self.min_window is not None else self.window

        results = []

        for i in range(min_w, len(data) + 1, self.step):
            window_data = data[max(0, i - self.window):i]

            try:
                params = fit(window_data, distribution=self.distribution)

                row = {"position": i}
                if index is not None and i - 1 < len(index):
                    row["date"] = index[i - 1]

                # Add all params as columns
                for field in fields(params):
                    value = getattr(params, field.name)
                    # Skip volatility_model and other non-numeric fields
                    if isinstance(value, (int, float)):
                        row[field.name] = value

                results.append(row)

            except Exception:
                # Skip this window if fitting fails
                continue

        return pd.DataFrame(results)

    def fit_with_expanding(
        self,
        data: NDArray[np.float64],
        index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Fit with expanding window (grow window from min_window to full data).

        Useful for initial periods where rolling window isn't yet full.

        Args:
            data: Array of observations
            index: Optional datetime index

        Returns:
            DataFrame with parameter history
        """
        data = np.asarray(data)
        min_w = self.min_window if self.min_window is not None else self.window

        results = []

        for i in range(min_w, len(data) + 1, self.step):
            # Use expanding window up to max window size
            window_start = max(0, i - self.window)
            window_data = data[window_start:i]

            try:
                params = fit(window_data, distribution=self.distribution)

                row = {"position": i, "window_size": len(window_data)}
                if index is not None and i - 1 < len(index):
                    row["date"] = index[i - 1]

                for field in fields(params):
                    value = getattr(params, field.name)
                    if isinstance(value, (int, float)):
                        row[field.name] = value

                results.append(row)

            except Exception:
                continue

        return pd.DataFrame(results)
