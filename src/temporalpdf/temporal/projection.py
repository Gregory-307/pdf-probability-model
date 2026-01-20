"""
Projection classes for parameter forecasting.

Projection contains the result of projecting parameters forward
using dynamics models, including uncertainty quantification.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ParamDistribution:
    """
    Distribution of parameters at a specific time point.

    Contains Monte Carlo samples of each parameter at time t.
    """
    values: dict[str, NDArray[np.float64]]  # {param_name: samples}
    t: int

    def mean(self) -> dict[str, float]:
        """Mean value of each parameter."""
        return {k: float(np.mean(v)) for k, v in self.values.items()}

    def std(self) -> dict[str, float]:
        """Standard deviation of each parameter."""
        return {k: float(np.std(v)) for k, v in self.values.items()}

    def quantile(self, q: float) -> dict[str, float]:
        """Quantile q of each parameter."""
        return {k: float(np.quantile(v, q)) for k, v in self.values.items()}

    def confidence_interval(self, level: float = 0.90) -> dict[str, tuple[float, float]]:
        """Confidence interval for each parameter."""
        alpha = (1 - level) / 2
        return {
            k: (float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha)))
            for k, v in self.values.items()
        }


@dataclass
class Projection:
    """
    Container for parameter projection results.

    Stores Monte Carlo paths of each parameter over the projection horizon,
    enabling uncertainty quantification at any future time point.

    Attributes:
        param_paths: Dictionary mapping parameter names to (n_paths, horizon) arrays
        horizon: Number of time steps projected
        n_paths: Number of Monte Carlo paths
        confidence_levels: Default confidence levels for summaries
    """
    param_paths: dict[str, NDArray[np.float64]]  # {param_name: (n_paths, horizon)}
    horizon: int
    n_paths: int
    confidence_levels: Sequence[float] = (0.5, 0.9)

    def mean(self, t: int | None = None) -> dict[str, float] | pd.DataFrame:
        """
        Mean parameters at time t (or all times if t is None).

        Args:
            t: Time step (1-indexed). None returns all times.

        Returns:
            Dictionary of mean values at time t, or DataFrame of all times.
        """
        if t is not None:
            return {k: float(v[:, t-1].mean()) for k, v in self.param_paths.items()}
        else:
            return pd.DataFrame({
                k: v.mean(axis=0) for k, v in self.param_paths.items()
            })

    def std(self, t: int | None = None) -> dict[str, float] | pd.DataFrame:
        """Standard deviation at time t (or all times)."""
        if t is not None:
            return {k: float(v[:, t-1].std()) for k, v in self.param_paths.items()}
        else:
            return pd.DataFrame({
                k: v.std(axis=0) for k, v in self.param_paths.items()
            })

    def quantile(self, q: float, t: int | None = None) -> dict[str, float] | pd.DataFrame:
        """Quantile q at time t (or all times)."""
        if t is not None:
            return {k: float(np.quantile(v[:, t-1], q)) for k, v in self.param_paths.items()}
        else:
            return pd.DataFrame({
                k: np.quantile(v, q, axis=0) for k, v in self.param_paths.items()
            })

    def at(self, t: int) -> ParamDistribution:
        """Get parameter distribution at specific time."""
        return ParamDistribution(
            values={k: v[:, t-1] for k, v in self.param_paths.items()},
            t=t,
        )

    def confidence_interval(
        self,
        level: float = 0.90,
        t: int | None = None,
    ) -> dict[str, tuple[float, float]] | pd.DataFrame:
        """
        Confidence interval at time t (or all times).

        Args:
            level: Confidence level (e.g., 0.90 for 90% CI)
            t: Time step (None = all times)

        Returns:
            For single t: dict of (lower, upper) per param
            For all t: DataFrame with multi-level columns
        """
        alpha = (1 - level) / 2
        if t is not None:
            return {
                k: (
                    float(np.quantile(v[:, t-1], alpha)),
                    float(np.quantile(v[:, t-1], 1 - alpha))
                )
                for k, v in self.param_paths.items()
            }
        else:
            result = {}
            for k, v in self.param_paths.items():
                result[(k, 'lower')] = np.quantile(v, alpha, axis=0)
                result[(k, 'upper')] = np.quantile(v, 1 - alpha, axis=0)
            return pd.DataFrame(result)

    def summary(self, t: int) -> str:
        """Generate summary string for time t."""
        mean_vals = self.mean(t)
        ci_vals = self.confidence_interval(0.90, t)

        lines = [f"Parameter Projection at t={t}"]
        lines.append("-" * 40)

        for param in mean_vals:
            m = mean_vals[param]
            lo, hi = ci_vals[param]
            lines.append(f"{param:12s}: {m:8.4f} [{lo:8.4f}, {hi:8.4f}]")

        return "\n".join(lines)
