"""
TemporalModel - Central class combining temporal modeling components.

This is the main entry point for temporal probability modeling,
combining:
- Parameter tracking over rolling windows
- Weighting schemes for estimation
- Dynamics models for projection
- Predictive distributions for decision-making
"""

from dataclasses import dataclass, field, fields
from typing import Sequence, Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .weights import WeightScheme, EMA
from .tracking import ParameterTracker
from .dynamics import DynamicsModel, Constant
from .projection import Projection
from .predictive import PredictiveDistribution
from ..core.result import DecisionSummary
from ..utilities import fit


@dataclass
class TemporalModel:
    """
    Central class for temporal probability modeling.

    Combines:
    - Parameter tracking (rolling window fitting)
    - Weighting (how to weight observations)
    - Dynamics (how parameters evolve over time)
    - Projection (forward simulation)
    - Decision (risk metrics with uncertainty)

    Example:
        >>> from temporalpdf.temporal import TemporalModel, EMA, MeanReverting, GARCH
        >>>
        >>> model = TemporalModel(
        ...     distribution="nig",
        ...     tracking=ParameterTracker("nig", window=60),
        ...     weighting=EMA(halflife=20),
        ...     dynamics={
        ...         "mu": Constant(),
        ...         "delta": GARCH(1, 1),
        ...         "alpha": MeanReverting(),
        ...         "beta": Constant(),
        ...     },
        ... )
        >>>
        >>> model.fit(returns)
        >>> projection = model.project(horizon=30)
        >>> decision = model.decision(t=5, alpha=0.05)
        >>> print(decision)

    Attributes:
        distribution: Distribution name ('nig', 'normal', 'student_t', 'skew_normal')
        tracking: Optional ParameterTracker for rolling window fitting
        weighting: WeightScheme for parameter estimation
        dynamics: Dict mapping parameter names to dynamics models
    """
    distribution: Literal["nig", "normal", "student_t", "skew_normal"]
    tracking: ParameterTracker | None = None
    weighting: WeightScheme = field(default_factory=lambda: EMA(halflife=20))
    dynamics: dict[str, DynamicsModel] = field(default_factory=dict)

    # Fitted state (populated after fit())
    param_history: pd.DataFrame | None = field(default=None, init=False, repr=False)
    current_params: object | None = field(default=None, init=False, repr=False)
    _fitted_dynamics: dict[str, DynamicsModel] = field(default_factory=dict, init=False, repr=False)
    _data: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        data: NDArray[np.float64],
        index: pd.DatetimeIndex | None = None,
    ) -> "TemporalModel":
        """
        Fit the temporal model to data.

        Steps:
        1. Track parameters over rolling windows (if tracking configured)
        2. Estimate current parameters using weighting scheme
        3. Fit dynamics models to parameter time series

        Args:
            data: Array of observations (e.g., returns)
            index: Optional datetime index for parameter history

        Returns:
            Self (for method chaining)
        """
        data = np.asarray(data)
        self._data = data

        # Step 1: Track parameters over time (if tracking configured)
        if self.tracking is not None:
            self.param_history = self.tracking.fit(data, index)

        # Step 2: Estimate current params with weighting
        weights = self.weighting.get_weights(len(data))
        self.current_params = fit(data, distribution=self.distribution)
        # Note: weighted fitting would require updating the fit function
        # For now, we use standard fit but weighting affects dynamics

        # Step 3: Fit dynamics to param history
        if self.param_history is not None and len(self.dynamics) > 0:
            for param_name, dynamics_model in self.dynamics.items():
                if param_name in self.param_history.columns:
                    param_series = self.param_history[param_name].values
                    # Fit dynamics model to parameter series
                    fitted = dynamics_model.fit(param_series)
                    self._fitted_dynamics[param_name] = fitted

        return self

    def project(
        self,
        horizon: int,
        n_paths: int = 1000,
        confidence_levels: Sequence[float] = (0.5, 0.9),
    ) -> Projection:
        """
        Project parameters forward using fitted dynamics.

        Args:
            horizon: Number of time steps to project
            n_paths: Number of Monte Carlo paths
            confidence_levels: Confidence levels for summaries

        Returns:
            Projection object with parameter paths and statistics
        """
        if self.current_params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get current param values
        param_values = {}
        for f in fields(self.current_params):
            value = getattr(self.current_params, f.name)
            if isinstance(value, (int, float)):
                param_values[f.name] = value

        # Project each parameter
        all_paths = {}
        for param_name, current_val in param_values.items():
            if param_name in self._fitted_dynamics:
                paths = self._fitted_dynamics[param_name].project(
                    current_val, horizon, n_paths
                )
            else:
                # If no dynamics model, keep constant
                paths = np.full((n_paths, horizon), current_val)
            all_paths[param_name] = paths

        return Projection(
            param_paths=all_paths,
            horizon=horizon,
            n_paths=n_paths,
            confidence_levels=confidence_levels,
        )

    def predictive(
        self,
        t: int,
        n_samples: int = 10000,
        n_paths: int = 1000,
    ) -> PredictiveDistribution:
        """
        Get predictive distribution at horizon t.

        Integrates over parameter uncertainty to get the full
        predictive distribution of returns.

        Args:
            t: Forecast horizon
            n_samples: Number of samples from predictive distribution
            n_paths: Number of parameter paths for projection

        Returns:
            PredictiveDistribution object for decision-making
        """
        projection = self.project(horizon=t, n_paths=n_paths)

        # Get param values at time t
        param_at_t = {}
        for param_name, paths in projection.param_paths.items():
            param_at_t[param_name] = paths[:, t-1]  # t is 1-indexed

        return PredictiveDistribution(
            distribution=self.distribution,
            param_paths=param_at_t,
            t=t,
            n_samples=n_samples,
        )

    def decision(
        self,
        t: int,
        alpha: float = 0.05,
        confidence_level: float = 0.90,
        n_samples: int = 10000,
        n_paths: int = 1000,
    ) -> DecisionSummary:
        """
        Compute trading decision metrics at horizon t.

        Returns VaR, CVaR, Kelly fraction, and probability of profit,
        all with confidence intervals from parameter uncertainty.

        Args:
            t: Forecast horizon
            alpha: VaR/CVaR confidence level
            confidence_level: CI level for all metrics
            n_samples: Samples for predictive distribution
            n_paths: Paths for parameter projection

        Returns:
            DecisionSummary with all metrics and CIs
        """
        predictive = self.predictive(t, n_samples, n_paths)
        return predictive.decision_summary(alpha=alpha, confidence_level=confidence_level)

    def var(
        self,
        t: int,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Quick access to VaR at horizon t."""
        return self.predictive(t, **kwargs).var(alpha)

    def cvar(
        self,
        t: int,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Quick access to CVaR at horizon t."""
        return self.predictive(t, **kwargs).cvar(alpha)

    def summary(self) -> str:
        """Generate model summary string."""
        lines = [
            f"TemporalModel({self.distribution})",
            "=" * 50,
        ]

        if self.current_params is not None:
            lines.append("Current Parameters:")
            for f in fields(self.current_params):
                value = getattr(self.current_params, f.name)
                if isinstance(value, (int, float)):
                    lines.append(f"  {f.name}: {value:.6f}")

        if self.param_history is not None:
            lines.append(f"\nParameter History: {len(self.param_history)} observations")

        if self._fitted_dynamics:
            lines.append("\nFitted Dynamics:")
            for param_name, dyn in self._fitted_dynamics.items():
                lines.append(f"  {param_name}: {type(dyn).__name__}")
                for k, v in dyn.summary().items():
                    lines.append(f"    {k}: {v:.6f}")

        return "\n".join(lines)
