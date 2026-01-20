"""
Parameter dynamics model implementations.

Each model describes how a parameter evolves over time and provides:
- fit(): Estimate model parameters from historical data
- project(): Monte Carlo simulation of future paths
- summary(): Dictionary of fitted parameters
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class Constant:
    """
    Constant dynamics - parameter stays at long-run average.

    The simplest dynamics model: parameter doesn't change.
    Useful as a baseline or for stable parameters.

    Example:
        >>> const = Constant()
        >>> const.fit(param_series)
        >>> paths = const.project(current_value=0.02, horizon=30)
        >>> # All paths are flat lines at long_run_value
    """
    long_run_value: float | None = None

    def fit(self, param_series: NDArray[np.float64]) -> "Constant":
        """Fit to mean of historical series."""
        self.long_run_value = float(np.mean(param_series))
        return self

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """Project constant value forward."""
        value = self.long_run_value if self.long_run_value is not None else current_value
        return np.full((n_paths, horizon), value)

    def summary(self) -> dict[str, float]:
        return {"long_run_value": self.long_run_value or 0.0}


@dataclass
class RandomWalk:
    """
    Random walk dynamics.

    Parameter follows a random walk with drift:
        theta[t+1] = theta[t] + drift + sigma * epsilon
        epsilon ~ N(0, 1)

    Args:
        drift: Drift term (estimated from data if None)
        sigma: Volatility of shocks (estimated from data if None)
        estimate_drift: Whether to estimate drift (False = assume zero drift)

    Example:
        >>> rw = RandomWalk()
        >>> rw.fit(param_series)
        >>> paths = rw.project(current_value=0.02, horizon=30)
    """
    drift: float | None = None
    sigma: float | None = None
    estimate_drift: bool = True

    def fit(self, param_series: NDArray[np.float64]) -> "RandomWalk":
        """Fit drift and sigma from differences."""
        diffs = np.diff(param_series)
        self.drift = float(np.mean(diffs)) if self.estimate_drift else 0.0
        self.sigma = float(np.std(diffs))
        return self

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """Simulate random walk paths."""
        drift = self.drift if self.drift is not None else 0.0
        sigma = self.sigma if self.sigma is not None else 0.01

        paths = np.zeros((n_paths, horizon))
        paths[:, 0] = current_value + drift + sigma * np.random.randn(n_paths)

        for t in range(1, horizon):
            paths[:, t] = paths[:, t-1] + drift + sigma * np.random.randn(n_paths)

        return paths

    def summary(self) -> dict[str, float]:
        return {
            "drift": self.drift or 0.0,
            "sigma": self.sigma or 0.0,
        }


@dataclass
class MeanReverting:
    """
    Mean-reverting dynamics (Ornstein-Uhlenbeck process).

    Parameter reverts to a long-run mean:
        d_theta = kappa * (long_run - theta) * dt + sigma * dW

    Discretized:
        theta[t+1] = theta[t] + kappa * (long_run - theta[t]) + sigma * epsilon

    Args:
        kappa: Mean reversion speed (higher = faster reversion)
        long_run: Long-run equilibrium value
        sigma: Volatility of shocks

    Example:
        >>> mr = MeanReverting()
        >>> mr.fit(param_series)
        >>> print(f"Half-life: {mr.half_life():.1f} periods")
        >>> paths = mr.project(current_value=0.05, horizon=30)
    """
    kappa: float | None = None
    long_run: float | None = None
    sigma: float | None = None

    def fit(self, param_series: NDArray[np.float64]) -> "MeanReverting":
        """
        Fit via AR(1) regression.

        theta[t+1] = c + phi * theta[t] + error
        Then: kappa = 1 - phi, long_run = c / kappa
        """
        y = param_series[1:]
        x = param_series[:-1]

        slope, intercept, _, _, _ = stats.linregress(x, y)

        self.kappa = float(1 - slope)
        if self.kappa > 0.001:
            self.long_run = float(intercept / self.kappa)
        else:
            self.long_run = float(np.mean(param_series))

        residuals = y - (intercept + slope * x)
        self.sigma = float(np.std(residuals))

        return self

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """Simulate mean-reverting paths."""
        kappa = self.kappa if self.kappa is not None else 0.1
        long_run = self.long_run if self.long_run is not None else current_value
        sigma = self.sigma if self.sigma is not None else 0.01

        paths = np.zeros((n_paths, horizon))
        theta = np.full(n_paths, current_value)

        for t in range(horizon):
            theta = theta + kappa * (long_run - theta) + sigma * np.random.randn(n_paths)
            paths[:, t] = theta

        return paths

    def half_life(self) -> float:
        """Time for mean reversion to reduce deviation by 50%."""
        if self.kappa is not None and self.kappa > 0:
            return np.log(2) / self.kappa
        return np.inf

    def summary(self) -> dict[str, float]:
        return {
            "kappa": self.kappa or 0.0,
            "long_run": self.long_run or 0.0,
            "sigma": self.sigma or 0.0,
            "half_life": self.half_life(),
        }


@dataclass
class AR:
    """
    Autoregressive process of order p.

    theta[t] = c + phi_1 * theta[t-1] + ... + phi_p * theta[t-p] + sigma * epsilon

    Args:
        order: AR order (number of lags)
        coefficients: Fitted coefficients [c, phi_1, ..., phi_p]
        sigma: Residual standard deviation

    Example:
        >>> ar = AR(order=2)
        >>> ar.fit(param_series)
        >>> paths = ar.project(current_value=0.02, horizon=30)

    Note:
        Requires statsmodels for fitting. If not available, falls back to AR(1).
    """
    order: int = 1
    coefficients: NDArray[np.float64] | None = None
    sigma: float | None = None

    def fit(self, param_series: NDArray[np.float64]) -> "AR":
        """Fit AR model using statsmodels or fallback to simple AR(1)."""
        try:
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(param_series, lags=self.order).fit()
            self.coefficients = np.concatenate([[model.params[0]], model.params[1:]])
            self.sigma = float(np.std(model.resid))
        except ImportError:
            # Fallback to simple AR(1) via linear regression
            if self.order >= 1:
                y = param_series[1:]
                x = param_series[:-1]
                slope, intercept, _, _, _ = stats.linregress(x, y)
                self.coefficients = np.array([intercept, slope])
                residuals = y - (intercept + slope * x)
                self.sigma = float(np.std(residuals))
            else:
                self.coefficients = np.array([np.mean(param_series)])
                self.sigma = float(np.std(param_series))

        return self

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """Simulate AR paths."""
        if self.coefficients is None:
            return np.full((n_paths, horizon), current_value)

        sigma = self.sigma if self.sigma is not None else 0.01
        c = self.coefficients[0]
        phis = self.coefficients[1:]

        paths = np.zeros((n_paths, horizon))

        # Initialize history with current value
        order = len(phis)
        history = np.full((n_paths, order), current_value)

        for t in range(horizon):
            new_val = c + np.sum(phis * history, axis=1) + sigma * np.random.randn(n_paths)
            paths[:, t] = new_val

            # Shift history
            if order > 1:
                history = np.column_stack([new_val, history[:, :-1]])
            else:
                history = new_val.reshape(-1, 1)

        return paths

    def summary(self) -> dict[str, float]:
        result = {"order": float(self.order), "sigma": self.sigma or 0.0}
        if self.coefficients is not None:
            result["intercept"] = float(self.coefficients[0])
            for i, phi in enumerate(self.coefficients[1:], 1):
                result[f"phi_{i}"] = float(phi)
        return result


@dataclass
class GARCH:
    """
    GARCH(p, q) for volatility parameters.

    Models time-varying variance:
        sigma^2[t] = omega + sum(alpha_i * epsilon^2[t-i]) + sum(beta_j * sigma^2[t-j])

    Args:
        p: Order of ARCH terms (lagged squared residuals)
        q: Order of GARCH terms (lagged conditional variance)
        omega: Constant term
        alpha: ARCH coefficients
        beta: GARCH coefficients

    Example:
        >>> garch = GARCH(p=1, q=1)
        >>> garch.fit(volatility_series)
        >>> paths = garch.project(current_value=0.02, horizon=30)

    Note:
        Requires arch library for fitting. If not available, falls back to simple model.
    """
    p: int = 1
    q: int = 1
    omega: float | None = None
    alpha: NDArray[np.float64] | None = None
    beta: NDArray[np.float64] | None = None
    _last_variance: float = field(default=0.0001, repr=False)
    _last_residual: float = field(default=0.0, repr=False)

    def fit(self, param_series: NDArray[np.float64]) -> "GARCH":
        """Fit GARCH model using arch library or fallback."""
        # Changes in the parameter
        changes = np.diff(param_series)

        try:
            from arch import arch_model
            # Scale for numerical stability
            scaled = changes * 100
            model = arch_model(scaled, vol='Garch', p=self.p, q=self.q)
            result = model.fit(disp='off')

            self.omega = result.params['omega'] / 10000
            self.alpha = np.array([result.params[f'alpha[{i+1}]'] for i in range(self.p)])
            self.beta = np.array([result.params[f'beta[{i+1}]'] for i in range(self.q)])
            self._last_variance = result.conditional_volatility[-1]**2 / 10000
            self._last_residual = changes[-1]

        except ImportError:
            # Fallback: assume constant variance
            self.omega = float(np.var(changes))
            self.alpha = np.array([0.1])
            self.beta = np.array([0.85])
            self._last_variance = float(np.var(changes))
            self._last_residual = changes[-1] if len(changes) > 0 else 0.0

        return self

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """Simulate GARCH paths."""
        omega = self.omega if self.omega is not None else 0.0001
        alpha = self.alpha if self.alpha is not None else np.array([0.1])
        beta = self.beta if self.beta is not None else np.array([0.85])

        paths = np.zeros((n_paths, horizon))
        sigma2 = np.full(n_paths, self._last_variance)
        value = np.full(n_paths, current_value)

        for t in range(horizon):
            epsilon = np.random.randn(n_paths)
            # Update variance: GARCH(1,1) simplified
            sigma2 = omega + alpha[0] * sigma2 * epsilon**2 + beta[0] * sigma2
            # Update value
            value = value + np.sqrt(np.maximum(sigma2, 1e-10)) * epsilon
            paths[:, t] = value

        return paths

    def persistence(self) -> float:
        """Alpha + beta sum. Should be < 1 for stationarity."""
        a = np.sum(self.alpha) if self.alpha is not None else 0.1
        b = np.sum(self.beta) if self.beta is not None else 0.85
        return a + b

    def summary(self) -> dict[str, float]:
        return {
            "omega": self.omega or 0.0,
            "alpha": float(np.sum(self.alpha)) if self.alpha is not None else 0.0,
            "beta": float(np.sum(self.beta)) if self.beta is not None else 0.0,
            "persistence": self.persistence(),
        }
