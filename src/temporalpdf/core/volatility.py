"""
Volatility evolution models for time-evolving distributions.

These models define how the scale/volatility parameter evolves over time.

References:
    Engle, R.F. (1982). Autoregressive Conditional Heteroskedasticity with
    Estimates of the Variance of United Kingdom Inflation. Econometrica.

    Bollerslev, T. (1986). Generalized Autoregressive Conditional
    Heteroskedasticity. Journal of Econometrics.

    Nelson, D.B. (1991). Conditional Heteroskedasticity in Asset Returns:
    A New Approach. Econometrica. (EGARCH)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class VolatilityModel(ABC):
    """
    Abstract base class for volatility evolution models.

    Defines how volatility (scale parameter) evolves over time.
    """

    @abstractmethod
    def at_time(self, sigma_0: float, t: float) -> float:
        """
        Compute volatility at time t.

        Args:
            sigma_0: Initial volatility at t=0
            t: Time point

        Returns:
            Volatility at time t
        """
        pass

    @abstractmethod
    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        """
        Compute volatility at multiple time points (vectorized).

        Args:
            sigma_0: Initial volatility at t=0
            t: Array of time points

        Returns:
            Array of volatilities at each time point
        """
        pass


@dataclass(frozen=True)
class LinearGrowth(VolatilityModel):
    """
    Linear volatility growth model.

    sigma(t) = sigma_0 * (1 + growth_rate * t)

    This is the simplest model but unrealistic for long horizons
    as it grows without bound.

    Attributes:
        growth_rate: Multiplicative growth rate per time unit
    """

    growth_rate: float = 0.0

    def at_time(self, sigma_0: float, t: float) -> float:
        return sigma_0 * (1 + self.growth_rate * t)

    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        return sigma_0 * (1 + self.growth_rate * t)


@dataclass(frozen=True)
class ExponentialDecay(VolatilityModel):
    """
    Exponential decay to long-run volatility.

    sigma(t) = sigma_long + (sigma_0 - sigma_long) * exp(-kappa * t)

    Models mean-reversion where volatility decays from its initial
    level toward a long-run equilibrium. Common in term structure
    models for implied volatility.

    As t -> infinity, sigma(t) -> sigma_long
    At t=0, sigma(0) = sigma_0

    Attributes:
        sigma_long: Long-run equilibrium volatility
        kappa: Mean-reversion speed (higher = faster reversion)

    Example:
        If current vol is 30% (elevated due to news) and long-run vol
        is 20%, with kappa=0.1:
        - t=0: vol = 30%
        - t=10: vol = 20% + 10% * exp(-1) = 23.7%
        - t=20: vol = 20% + 10% * exp(-2) = 21.4%
    """

    sigma_long: float
    kappa: float = 0.1

    def __post_init__(self) -> None:
        if self.sigma_long <= 0:
            raise ValueError("sigma_long must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")

    def at_time(self, sigma_0: float, t: float) -> float:
        return self.sigma_long + (sigma_0 - self.sigma_long) * np.exp(-self.kappa * t)

    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        return self.sigma_long + (sigma_0 - self.sigma_long) * np.exp(-self.kappa * t)


@dataclass(frozen=True)
class SquareRootDiffusion(VolatilityModel):
    """
    Square-root diffusion (CIR-style) volatility model.

    Expected volatility under the CIR process:
    E[sigma(t)] = sigma_long + (sigma_0 - sigma_long) * exp(-kappa * t)

    Same expected trajectory as ExponentialDecay, but this is derived
    from the Cox-Ingersoll-Ross (CIR) stochastic process:
    d(sigma^2) = kappa * (sigma_long^2 - sigma^2) * dt + xi * sigma * dW

    The CIR process is the standard for stochastic volatility in finance
    because it guarantees positive volatility (given Feller condition).

    For distributional forecasting without simulation, we use the
    expected path which matches ExponentialDecay.

    Reference:
        Cox, J.C., Ingersoll, J.E., & Ross, S.A. (1985). A Theory of the
        Term Structure of Interest Rates. Econometrica.

    Attributes:
        sigma_long: Long-run mean volatility
        kappa: Mean-reversion speed
    """

    sigma_long: float
    kappa: float = 0.1

    def __post_init__(self) -> None:
        if self.sigma_long <= 0:
            raise ValueError("sigma_long must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")

    def at_time(self, sigma_0: float, t: float) -> float:
        # Expected value under CIR dynamics
        return self.sigma_long + (sigma_0 - self.sigma_long) * np.exp(-self.kappa * t)

    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        return self.sigma_long + (sigma_0 - self.sigma_long) * np.exp(-self.kappa * t)


@dataclass(frozen=True)
class GARCHForecast(VolatilityModel):
    """
    GARCH(1,1) volatility forecast model.

    For forecasting h steps ahead from current variance sigma^2:
    E[sigma^2(t)] = omega/(1-alpha-beta) + (alpha+beta)^t * (sigma_0^2 - omega/(1-alpha-beta))

    This is the expected variance at time t under GARCH(1,1) dynamics
    when alpha + beta < 1 (stationarity condition).

    The unconditional (long-run) variance is:
    sigma_long^2 = omega / (1 - alpha - beta)

    Reference:
        Bollerslev, T. (1986). Generalized Autoregressive Conditional
        Heteroskedasticity. Journal of Econometrics.

    Attributes:
        omega: Constant term (intercept), must be positive
        alpha: ARCH coefficient (shock impact), typically 0.05-0.15
        beta: GARCH coefficient (persistence), typically 0.8-0.95

    Note:
        alpha + beta < 1 is required for stationarity
        alpha + beta close to 1 = highly persistent volatility
    """

    omega: float
    alpha: float = 0.1
    beta: float = 0.85

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        if self.alpha < 0 or self.alpha >= 1:
            raise ValueError("alpha must be in [0, 1)")
        if self.beta < 0 or self.beta >= 1:
            raise ValueError("beta must be in [0, 1)")
        if self.alpha + self.beta >= 1:
            raise ValueError(
                f"alpha + beta must be < 1 for stationarity, got {self.alpha + self.beta}"
            )

    @property
    def long_run_variance(self) -> float:
        """Unconditional (long-run) variance."""
        return self.omega / (1 - self.alpha - self.beta)

    @property
    def long_run_vol(self) -> float:
        """Unconditional (long-run) volatility (standard deviation)."""
        return np.sqrt(self.long_run_variance)

    @property
    def persistence(self) -> float:
        """Volatility persistence (alpha + beta)."""
        return self.alpha + self.beta

    @property
    def half_life(self) -> float:
        """Half-life of volatility shocks in time units."""
        return np.log(2) / (-np.log(self.persistence))

    def at_time(self, sigma_0: float, t: float) -> float:
        sigma_0_sq = sigma_0 ** 2
        long_run_var = self.long_run_variance
        persistence = self.persistence

        # Forecast variance
        forecast_var = long_run_var + (persistence ** t) * (sigma_0_sq - long_run_var)
        return np.sqrt(max(forecast_var, 1e-10))

    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        sigma_0_sq = sigma_0 ** 2
        long_run_var = self.long_run_variance
        persistence = self.persistence

        # Forecast variance (vectorized)
        forecast_var = long_run_var + (persistence ** t) * (sigma_0_sq - long_run_var)
        return np.sqrt(np.maximum(forecast_var, 1e-10))


@dataclass(frozen=True)
class TermStructure(VolatilityModel):
    """
    Term structure volatility model (piecewise constant).

    Uses a predefined term structure of volatilities at specific
    time points, with linear interpolation between points.

    This is useful when you have volatility forecasts from an
    external model (e.g., options-implied vol surface).

    Attributes:
        times: Array of time points (must start with 0)
        vols: Array of volatilities at each time point
    """

    times: tuple[float, ...]
    vols: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.vols):
            raise ValueError("times and vols must have same length")
        if len(self.times) < 2:
            raise ValueError("Need at least 2 time points")
        if self.times[0] != 0:
            raise ValueError("times must start with 0")
        if not all(t1 < t2 for t1, t2 in zip(self.times[:-1], self.times[1:])):
            raise ValueError("times must be strictly increasing")
        if not all(v > 0 for v in self.vols):
            raise ValueError("all vols must be positive")

    def at_time(self, sigma_0: float, t: float) -> float:
        # sigma_0 is ignored - we use the term structure directly
        # but the first vol should typically match sigma_0
        times = np.array(self.times)
        vols = np.array(self.vols)
        return float(np.interp(t, times, vols))

    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        times = np.array(self.times)
        vols = np.array(self.vols)
        return np.interp(t, times, vols)


# Convenience factory functions


def constant_volatility() -> LinearGrowth:
    """Constant volatility (no time evolution)."""
    return LinearGrowth(growth_rate=0.0)


def linear_growth(growth_rate: float) -> LinearGrowth:
    """Linear volatility growth: sigma(t) = sigma_0 * (1 + rate * t)."""
    return LinearGrowth(growth_rate=growth_rate)


def mean_reverting(sigma_long: float, kappa: float = 0.1) -> ExponentialDecay:
    """
    Mean-reverting volatility with exponential decay.

    Args:
        sigma_long: Long-run equilibrium volatility
        kappa: Mean-reversion speed (higher = faster)

    Returns:
        ExponentialDecay model
    """
    return ExponentialDecay(sigma_long=sigma_long, kappa=kappa)


def garch_forecast(
    omega: float,
    alpha: float = 0.1,
    beta: float = 0.85,
) -> GARCHForecast:
    """
    GARCH(1,1) volatility forecast.

    Args:
        omega: Constant term (intercept)
        alpha: ARCH coefficient (shock impact)
        beta: GARCH coefficient (persistence)

    Returns:
        GARCHForecast model

    Example:
        Typical equity values: omega=0.00001, alpha=0.1, beta=0.85
        This gives long-run vol of ~3.2% with persistence of 0.95
    """
    return GARCHForecast(omega=omega, alpha=alpha, beta=beta)
