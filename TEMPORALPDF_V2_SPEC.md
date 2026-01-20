# TEMPORALPDF V2: IMPLEMENTATION SPECIFICATION

**Status**: AUTHORITATIVE - This document governs all implementation decisions.
**Usage**: Claude Code must reference this file before implementing any module.

---

## SECTION 1: EXISTING CODE STATE

### Files That Exist (DO NOT RECREATE)
```
src/temporalpdf/
├── core/
│   ├── distribution.py      # Distribution Protocol - EXISTS, may need update
│   ├── parameters.py        # Parameter dataclasses - EXISTS, may need update
│   ├── volatility.py        # VolatilityModel classes - EXISTS, needs refactor
│   └── result.py            # Result wrappers - EXISTS
├── distributions/
│   ├── nig.py               # NIGDistribution - EXISTS, KEEP
│   ├── normal.py            # NormalDistribution - EXISTS, REPLACE with scipy wrapper
│   ├── student_t.py         # StudentTDistribution - EXISTS, REPLACE with scipy wrapper
│   ├── skew_normal.py       # EXISTS, REPLACE with scipy wrapper
│   ├── generalized_laplace.py # EXISTS, KEEP (custom)
│   └── registry.py          # EXISTS
├── scoring/
│   └── rules.py             # CRPS, log_score - EXISTS, swap internals to scoringrules
├── decision/
│   ├── risk.py              # var(), cvar() - EXISTS, needs update for CIs
│   ├── kelly.py             # kelly_fraction() - EXISTS, needs update for CIs
│   └── probability.py       # prob_greater_than() - EXISTS
├── validation/
│   ├── validator.py         # EXISTS
│   └── metrics.py           # EXISTS
├── utilities.py             # fit_nig, fit_student_t, select_best_distribution, etc. - EXISTS
└── visualization/           # EXISTS
```

### Files To Create (DO NOT EXIST YET)
```
src/temporalpdf/
├── api.py                   # High-level facade: discover(), forecast()
├── discovery/
│   ├── __init__.py
│   ├── selection.py         # select_best_distribution logic (move from utilities.py)
│   ├── scoring.py           # Wrapper over scoringrules
│   └── significance.py      # Paired t-tests, multiple comparison
├── fitting/
│   ├── __init__.py
│   ├── mle.py               # Unified fit() function
│   └── weighted.py          # Weighted likelihood
├── temporal/                # THE NEW CORE MODULE
│   ├── __init__.py
│   ├── model.py             # TemporalModel class
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py       # ParameterTracker
│   ├── weights/
│   │   ├── __init__.py
│   │   ├── base.py          # WeightScheme Protocol
│   │   ├── sma.py
│   │   ├── ema.py
│   │   ├── linear.py
│   │   ├── power.py
│   │   ├── gaussian.py
│   │   ├── regime.py
│   │   └── custom.py
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py          # DynamicsModel Protocol
│   │   ├── constant.py
│   │   ├── random_walk.py
│   │   ├── mean_reverting.py
│   │   ├── ar.py
│   │   ├── garch.py
│   │   └── regime_switching.py
│   ├── projection.py        # Forward projection
│   └── predictive.py        # PredictiveDistribution
├── backtest/
│   ├── __init__.py
│   ├── runner.py            # Backtest class
│   ├── tests.py             # Kupiec, Christoffersen
│   └── comparison.py        # Compare backtests
└── conditional/             # PHASE 6 - DEFER
    ├── __init__.py
    └── model.py             # ConditionalModel wrapper
```

---

## SECTION 2: CODING STANDARDS (MANDATORY)

### 2.1 Rich Return Pattern
**NEVER return raw floats for metrics. ALWAYS return objects with uncertainty.**

```python
# WRONG
def var(self, alpha: float) -> float:
    return -self.ppf(alpha)

# CORRECT
@dataclass(frozen=True)
class RiskMetric:
    value: float
    confidence_interval: tuple[float, float] | None = None
    standard_error: float | None = None

def var(self, alpha: float, confidence_level: float = 0.90) -> RiskMetric:
    # Compute value
    # Compute CI via Monte Carlo over param paths
    return RiskMetric(value=..., confidence_interval=..., standard_error=...)
```

### 2.2 Distribution Protocol
**All distributions MUST implement this interface exactly.**

```python
from typing import Protocol, runtime_checkable
from numpy.typing import ArrayLike

@runtime_checkable
class Distribution(Protocol):
    """Protocol that all distributions must implement."""
    
    def pdf(self, x: ArrayLike, t: float, params: Parameters) -> ArrayLike:
        """Probability density function."""
        ...
    
    def cdf(self, x: ArrayLike, t: float, params: Parameters) -> ArrayLike:
        """Cumulative distribution function."""
        ...
    
    def ppf(self, q: ArrayLike, t: float, params: Parameters) -> ArrayLike:
        """Percent point function (inverse CDF)."""
        ...
    
    def sample(self, n: int, t: float, params: Parameters) -> ArrayLike:
        """Generate random samples."""
        ...
    
    def fit(self, data: ArrayLike, weights: WeightScheme | None = None) -> Parameters:
        """Fit parameters to data via MLE."""
        ...
```

### 2.3 Parameter Immutability
**All parameter objects MUST be frozen dataclasses.**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class NIGParameters:
    mu: float      # Location
    delta: float   # Scale (must be > 0)
    alpha: float   # Tail heaviness (must be > |beta|)
    beta: float    # Skewness (must be |beta| < alpha)
    
    def __post_init__(self):
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.alpha <= abs(self.beta):
            raise ValueError("alpha must be > |beta|")
    
    def with_delta(self, new_delta: float) -> "NIGParameters":
        """Return new params with updated delta."""
        return NIGParameters(mu=self.mu, delta=new_delta, alpha=self.alpha, beta=self.beta)
```

### 2.4 Enum for Configurations
**NO magic strings. Use Enums.**

```python
from enum import Enum, auto

class WeightSchemeType(Enum):
    SMA = auto()
    EMA = auto()
    LINEAR = auto()
    POWER = auto()
    GAUSSIAN = auto()
    REGIME = auto()
    CUSTOM = auto()

class DynamicsType(Enum):
    CONSTANT = auto()
    RANDOM_WALK = auto()
    MEAN_REVERTING = auto()
    AR = auto()
    GARCH = auto()
    REGIME_SWITCHING = auto()
```

### 2.5 Type Hints
**ALL functions MUST have complete type hints.**

```python
# WRONG
def fit(data, distribution):
    ...

# CORRECT
def fit(
    data: np.ndarray,
    distribution: Literal["nig", "student_t", "normal", "skew_normal", "gen_laplace"],
    weights: WeightScheme | None = None,
) -> NIGParameters | StudentTParameters | NormalParameters | SkewNormalParameters | GenLaplaceParameters:
    ...
```

---

## SECTION 3: STAGE 1 - DISCOVERY

### 3.1 Module: `discovery/selection.py`

**Function**: `discover`

```python
def discover(
    data: np.ndarray,
    candidates: Sequence[str] = ("normal", "student_t", "nig", "skew_normal"),
    scoring: Sequence[Literal["crps", "log_score"]] = ("crps",),
    test_fraction: float = 0.2,
    cv_folds: int = 5,
    significance_level: float = 0.05,
) -> DiscoveryResult:
    """
    Determine best distribution family with statistical rigor.
    
    Process:
    1. For each candidate distribution:
       a. Fit to training data (1 - test_fraction)
       b. Compute CRPS and/or log_score on test data
    2. Run k-fold cross-validation
    3. Perform pairwise paired t-tests between best and others
    4. Assign confidence level based on p-values
    
    Returns:
        DiscoveryResult with:
        - best: str (name of best distribution)
        - confidence: Literal["high", "medium", "low"]
        - scores: dict[str, float] (mean score per distribution)
        - pairwise_pvalues: dict[tuple[str, str], float]
        - best_params: Parameters (fitted params for best distribution)
    """
```

**Dataclass**: `DiscoveryResult`

```python
@dataclass(frozen=True)
class DiscoveryResult:
    best: str
    confidence: Literal["high", "medium", "low"]
    scores: dict[str, float]
    std_scores: dict[str, float]
    pairwise_pvalues: dict[tuple[str, str], float]
    best_params: Parameters
    
    def summary(self) -> str:
        """Return formatted summary table."""
        ...
```

### 3.2 Module: `discovery/scoring.py`

**Use**: `scoringrules` library (NOT `properscoring`)

```python
import scoringrules as sr

def crps_from_samples(y_true: float, samples: np.ndarray) -> float:
    """Compute CRPS from samples."""
    return sr.crps_ensemble(y_true, samples)

def crps_normal(y_true: float, mu: float, sigma: float) -> float:
    """Closed-form CRPS for Normal."""
    return sr.crps_normal(y_true, mu, sigma)

def log_score(y_true: float, pdf_value: float) -> float:
    """Negative log likelihood."""
    return -np.log(max(pdf_value, 1e-300))
```

### 3.3 Module: `discovery/significance.py`

```python
from scipy import stats

def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Paired t-test for comparing two distributions.
    
    Returns: p-value
    """
    _, pvalue = stats.ttest_rel(scores_a, scores_b)
    return pvalue

def determine_confidence(
    best_scores: np.ndarray,
    second_scores: np.ndarray,
    significance_level: float = 0.05,
) -> Literal["high", "medium", "low"]:
    """
    Determine confidence in selection.
    
    - high: p < significance_level AND gap > 10%
    - medium: p < significance_level
    - low: p >= significance_level
    """
    pvalue = paired_t_test(best_scores, second_scores)
    gap = (np.mean(second_scores) - np.mean(best_scores)) / np.mean(best_scores)
    
    if pvalue < significance_level and gap > 0.10:
        return "high"
    elif pvalue < significance_level:
        return "medium"
    else:
        return "low"
```

---

## SECTION 4: STAGE 2 - FITTING

### 4.1 Module: `fitting/mle.py`

**Function**: `fit`

```python
def fit(
    data: np.ndarray,
    distribution: Literal["nig", "student_t", "normal", "skew_normal", "gen_laplace"],
    weights: WeightScheme | None = None,
    compute_std_errors: bool = False,
    n_bootstrap: int = 500,
) -> Parameters:
    """
    Fit distribution to data via maximum likelihood.
    
    Args:
        data: Array of observations
        distribution: Which distribution to fit
        weights: Optional weighting scheme (affects likelihood)
        compute_std_errors: Whether to bootstrap standard errors
        n_bootstrap: Number of bootstrap samples for std errors
    
    Returns:
        Fitted parameters (type depends on distribution)
    """
```

### 4.2 Distribution-Specific Fitting

**NIG**: Use custom optimizer (scipy.optimize.minimize with Nelder-Mead)
```python
def fit_nig(data: np.ndarray, weights: np.ndarray | None = None) -> NIGParameters:
    """
    Fit NIG via MLE.
    
    Parameterization:
    - mu: location (unconstrained)
    - delta: scale (log-transform for positivity)
    - alpha: tail heaviness (log-transform for positivity)
    - beta: skewness (arctanh-transform to satisfy |beta| < alpha)
    
    Optimization:
    - Method: Nelder-Mead
    - Initial guess: mu=mean(data), delta=std(data), alpha=5, beta=0
    - Max iterations: 1000
    """
```

**Normal, Student-t, Skew-Normal**: Use scipy.stats wrappers
```python
def fit_normal(data: np.ndarray, weights: np.ndarray | None = None) -> NormalParameters:
    """Fit Normal via scipy.stats.norm.fit or weighted MLE."""
    if weights is None:
        mu, sigma = np.mean(data), np.std(data)
    else:
        w = weights / weights.sum()
        mu = np.average(data, weights=w)
        sigma = np.sqrt(np.average((data - mu)**2, weights=w))
    return NormalParameters(mu_0=mu, sigma_0=sigma)

def fit_student_t(data: np.ndarray, weights: np.ndarray | None = None) -> StudentTParameters:
    """Fit Student-t via scipy.stats.t.fit."""
    nu, loc, scale = stats.t.fit(data)
    return StudentTParameters(nu=nu, mu_0=loc, sigma_0=scale)

def fit_skew_normal(data: np.ndarray, weights: np.ndarray | None = None) -> SkewNormalParameters:
    """Fit Skew-Normal via scipy.stats.skewnorm.fit."""
    a, loc, scale = stats.skewnorm.fit(data)
    return SkewNormalParameters(alpha=a, xi=loc, omega=scale)
```

---

## SECTION 5: STAGE 3 - TEMPORAL DYNAMICS

### 5.1 Module: `temporal/tracking/tracker.py`

**Class**: `ParameterTracker`

```python
@dataclass
class ParameterTracker:
    """
    Track distribution parameters over rolling windows.
    
    Usage:
        tracker = ParameterTracker(distribution="nig", window=60, step=1)
        param_history = tracker.fit(returns)
        # param_history is DataFrame with columns: date, mu, delta, alpha, beta
    """
    distribution: str
    window: int
    step: int = 1
    min_window: int | None = None
    
    def fit(self, data: np.ndarray, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        """
        Roll through data, fitting distribution at each step.
        
        Returns:
            DataFrame with columns for each parameter, indexed by date/position.
        """
        results = []
        min_w = self.min_window or self.window
        
        for i in range(min_w, len(data), self.step):
            window_data = data[max(0, i - self.window):i]
            params = fit(window_data, distribution=self.distribution)
            
            row = {"position": i}
            if index is not None:
                row["date"] = index[i - 1]
            
            # Add all params as columns
            for field in dataclasses.fields(params):
                row[field.name] = getattr(params, field.name)
            
            results.append(row)
        
        return pd.DataFrame(results)
```

### 5.2 Module: `temporal/weights/`

**Protocol**: `WeightScheme`

```python
from typing import Protocol

class WeightScheme(Protocol):
    """Protocol for weighting schemes."""
    
    def get_weights(self, n: int) -> np.ndarray:
        """
        Return array of weights for n observations.
        
        Index 0 = most recent observation.
        Weights sum to 1.
        """
        ...
    
    def effective_sample_size(self, n: int) -> float:
        """Return effective sample size given n observations."""
        w = self.get_weights(n)
        return 1.0 / np.sum(w ** 2)
```

**Implementation**: `SMA`

```python
@dataclass
class SMA:
    """Simple Moving Average - equal weights."""
    window: int
    
    def get_weights(self, n: int) -> np.ndarray:
        effective_n = min(n, self.window)
        weights = np.zeros(n)
        weights[:effective_n] = 1.0 / effective_n
        return weights
    
    def effective_sample_size(self, n: int) -> float:
        return float(min(n, self.window))
```

**Implementation**: `EMA`

```python
@dataclass
class EMA:
    """
    Exponential Moving Average.
    
    Formula: weight[i] = (1 - alpha) * alpha^i
    Where: alpha = exp(-ln(2) / halflife)
    
    Index 0 = most recent.
    """
    halflife: float
    
    def get_weights(self, n: int) -> np.ndarray:
        alpha = np.exp(-np.log(2) / self.halflife)
        weights = (1 - alpha) * (alpha ** np.arange(n))
        return weights / weights.sum()
    
    def effective_sample_size(self, n: int) -> float:
        alpha = np.exp(-np.log(2) / self.halflife)
        # ESS = (1 + alpha) / (1 - alpha) for infinite series
        # For finite n, compute exactly
        w = self.get_weights(n)
        return 1.0 / np.sum(w ** 2)
```

**Implementation**: `Linear`

```python
@dataclass
class Linear:
    """
    Linear decay weights.
    
    Formula: weight[i] = (window - i) for i < window, else 0
    Normalized to sum to 1.
    """
    window: int
    
    def get_weights(self, n: int) -> np.ndarray:
        effective_n = min(n, self.window)
        raw = np.maximum(self.window - np.arange(n), 0).astype(float)
        return raw / raw.sum()
```

**Implementation**: `PowerDecay`

```python
@dataclass
class PowerDecay:
    """
    Power decay weights.
    
    Formula: weight[i] = 1 / (i + 1)^power
    
    power=0.5 gives sqrt decay.
    power=1.0 gives 1/n decay.
    """
    power: float
    window: int | None = None
    
    def get_weights(self, n: int) -> np.ndarray:
        effective_n = n if self.window is None else min(n, self.window)
        raw = 1.0 / ((np.arange(n) + 1) ** self.power)
        if self.window is not None:
            raw[self.window:] = 0
        return raw / raw.sum()
```

**Implementation**: `Gaussian`

```python
@dataclass
class Gaussian:
    """
    Gaussian decay weights.
    
    Formula: weight[i] = exp(-0.5 * (i / sigma)^2)
    """
    sigma: float
    
    def get_weights(self, n: int) -> np.ndarray:
        raw = np.exp(-0.5 * (np.arange(n) / self.sigma) ** 2)
        return raw / raw.sum()
```

**Implementation**: `Custom`

```python
@dataclass
class Custom:
    """
    Custom weight function.
    
    func takes (i, n) -> weight for observation i out of n total.
    i=0 is most recent.
    """
    func: Callable[[int, int], float]
    
    def get_weights(self, n: int) -> np.ndarray:
        raw = np.array([self.func(i, n) for i in range(n)])
        return raw / raw.sum()
```

### 5.3 Module: `temporal/dynamics/`

**Protocol**: `DynamicsModel`

```python
class DynamicsModel(Protocol):
    """Protocol for parameter dynamics models."""
    
    def fit(self, param_series: np.ndarray) -> "DynamicsModel":
        """Fit model to historical parameter series."""
        ...
    
    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> np.ndarray:
        """
        Project parameter forward.
        
        Returns:
            Array of shape (n_paths, horizon) with simulated paths.
        """
        ...
    
    def summary(self) -> dict[str, float]:
        """Return fitted model parameters."""
        ...
```

**Implementation**: `Constant`

```python
@dataclass
class Constant:
    """Parameter stays at long-run average."""
    long_run_value: float | None = None
    
    def fit(self, param_series: np.ndarray) -> "Constant":
        self.long_run_value = np.mean(param_series)
        return self
    
    def project(self, current_value: float, horizon: int, n_paths: int = 1000) -> np.ndarray:
        return np.full((n_paths, horizon), self.long_run_value)
    
    def summary(self) -> dict[str, float]:
        return {"long_run_value": self.long_run_value}
```

**Implementation**: `RandomWalk`

```python
@dataclass
class RandomWalk:
    """
    Random walk dynamics.
    
    theta[t+1] = theta[t] + drift + sigma * epsilon
    epsilon ~ N(0, 1)
    """
    drift: float | None = None
    sigma: float | None = None
    estimate_drift: bool = True
    
    def fit(self, param_series: np.ndarray) -> "RandomWalk":
        diffs = np.diff(param_series)
        self.drift = np.mean(diffs) if self.estimate_drift else 0.0
        self.sigma = np.std(diffs)
        return self
    
    def project(self, current_value: float, horizon: int, n_paths: int = 1000) -> np.ndarray:
        paths = np.zeros((n_paths, horizon))
        paths[:, 0] = current_value + self.drift + self.sigma * np.random.randn(n_paths)
        for t in range(1, horizon):
            paths[:, t] = paths[:, t-1] + self.drift + self.sigma * np.random.randn(n_paths)
        return paths
    
    def summary(self) -> dict[str, float]:
        return {"drift": self.drift, "sigma": self.sigma}
```

**Implementation**: `MeanReverting`

```python
@dataclass
class MeanReverting:
    """
    Ornstein-Uhlenbeck process.
    
    d_theta = kappa * (long_run - theta) * dt + sigma * dW
    
    Discretized:
    theta[t+1] = theta[t] + kappa * (long_run - theta[t]) + sigma * epsilon
    """
    kappa: float | None = None       # Mean reversion speed
    long_run: float | None = None    # Long-run value
    sigma: float | None = None       # Volatility of shocks
    
    def fit(self, param_series: np.ndarray) -> "MeanReverting":
        # AR(1) regression: theta[t+1] = c + phi * theta[t] + error
        # Then: kappa = 1 - phi, long_run = c / kappa
        from scipy import stats
        y = param_series[1:]
        x = param_series[:-1]
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        self.kappa = 1 - slope
        self.long_run = intercept / self.kappa if self.kappa > 0.001 else np.mean(param_series)
        residuals = y - (intercept + slope * x)
        self.sigma = np.std(residuals)
        return self
    
    def project(self, current_value: float, horizon: int, n_paths: int = 1000) -> np.ndarray:
        paths = np.zeros((n_paths, horizon))
        theta = np.full(n_paths, current_value)
        for t in range(horizon):
            theta = theta + self.kappa * (self.long_run - theta) + self.sigma * np.random.randn(n_paths)
            paths[:, t] = theta
        return paths
    
    def half_life(self) -> float:
        """Time for mean reversion to reduce deviation by 50%."""
        return np.log(2) / self.kappa if self.kappa > 0 else np.inf
    
    def summary(self) -> dict[str, float]:
        return {
            "kappa": self.kappa,
            "long_run": self.long_run,
            "sigma": self.sigma,
            "half_life": self.half_life(),
        }
```

**Implementation**: `AR`

```python
@dataclass
class AR:
    """
    Autoregressive process of order p.
    
    theta[t] = c + phi_1 * theta[t-1] + ... + phi_p * theta[t-p] + sigma * epsilon
    """
    order: int = 1
    coefficients: np.ndarray | None = None  # [c, phi_1, ..., phi_p]
    sigma: float | None = None
    
    def fit(self, param_series: np.ndarray) -> "AR":
        from statsmodels.tsa.ar_model import AutoReg
        model = AutoReg(param_series, lags=self.order).fit()
        self.coefficients = np.concatenate([[model.params[0]], model.params[1:]])
        self.sigma = np.std(model.resid)
        return self
    
    def project(self, current_value: float, horizon: int, n_paths: int = 1000) -> np.ndarray:
        # Need last `order` values for initialization
        # For simplicity, assume current_value is the last value
        # and use long-run mean for padding
        paths = np.zeros((n_paths, horizon))
        c = self.coefficients[0]
        phis = self.coefficients[1:]
        
        # Initialize with current value replicated
        history = np.full((n_paths, self.order), current_value)
        
        for t in range(horizon):
            new_val = c + np.sum(phis * history, axis=1) + self.sigma * np.random.randn(n_paths)
            paths[:, t] = new_val
            history = np.column_stack([new_val, history[:, :-1]])
        
        return paths
```

**Implementation**: `GARCH`

```python
@dataclass
class GARCH:
    """
    GARCH(p, q) for volatility parameters.
    
    sigma^2[t] = omega + sum(alpha_i * epsilon^2[t-i]) + sum(beta_j * sigma^2[t-j])
    
    Uses arch library for fitting.
    """
    p: int = 1
    q: int = 1
    omega: float | None = None
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None
    
    def fit(self, param_series: np.ndarray) -> "GARCH":
        from arch import arch_model
        # GARCH fits to returns, so we need the changes in the parameter
        changes = np.diff(param_series) * 100  # Scale for numerical stability
        model = arch_model(changes, vol='Garch', p=self.p, q=self.q)
        result = model.fit(disp='off')
        
        self.omega = result.params['omega'] / 10000
        self.alpha = np.array([result.params[f'alpha[{i+1}]'] for i in range(self.p)])
        self.beta = np.array([result.params[f'beta[{i+1}]'] for i in range(self.q)])
        self._last_variance = result.conditional_volatility[-1]**2 / 10000
        self._last_residual = changes[-1] / 100
        return self
    
    def project(self, current_value: float, horizon: int, n_paths: int = 1000) -> np.ndarray:
        paths = np.zeros((n_paths, horizon))
        sigma2 = np.full(n_paths, self._last_variance)
        value = np.full(n_paths, current_value)
        
        for t in range(horizon):
            epsilon = np.random.randn(n_paths)
            sigma2 = self.omega + self.alpha[0] * sigma2 * epsilon**2 + self.beta[0] * sigma2
            value = value + np.sqrt(sigma2) * epsilon
            paths[:, t] = value
        
        return paths
    
    def persistence(self) -> float:
        """Alpha + beta. Should be < 1 for stationarity."""
        return np.sum(self.alpha) + np.sum(self.beta)
    
    def summary(self) -> dict[str, float]:
        return {
            "omega": self.omega,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "persistence": self.persistence(),
        }
```

### 5.4 Module: `temporal/model.py`

**Class**: `TemporalModel`

```python
@dataclass
class TemporalModel:
    """
    Central class that combines tracking, weighting, and dynamics.
    
    Usage:
        temporal = TemporalModel(
            distribution="nig",
            tracking=Rolling(window=60, step=1),
            weighting=EMA(halflife=20),
            dynamics={
                "mu": RandomWalk(),
                "delta": GARCH(1, 1),
                "alpha": MeanReverting(),
                "beta": Constant(),
            },
        )
        temporal.fit(returns)
        projection = temporal.project(horizon=30)
        decision = temporal.decision(t=5, alpha=0.05)
    """
    distribution: str
    tracking: ParameterTracker | None = None
    weighting: WeightScheme = field(default_factory=lambda: EMA(halflife=20))
    dynamics: dict[str, DynamicsModel] = field(default_factory=dict)
    
    # Fitted state
    param_history: pd.DataFrame | None = field(default=None, init=False)
    current_params: Parameters | None = field(default=None, init=False)
    _fitted_dynamics: dict[str, DynamicsModel] = field(default_factory=dict, init=False)
    
    def fit(self, data: np.ndarray, index: pd.DatetimeIndex | None = None) -> "TemporalModel":
        """
        Fit the temporal model to data.
        
        Steps:
        1. Track parameters over rolling windows (if tracking configured)
        2. Estimate current parameters using weighting scheme
        3. Fit dynamics models to parameter time series
        """
        # Step 1: Track parameters
        if self.tracking is not None:
            self.param_history = self.tracking.fit(data, index)
        
        # Step 2: Estimate current params with weighting
        weights = self.weighting.get_weights(len(data))
        self.current_params = fit(data, self.distribution, weights=weights)
        
        # Step 3: Fit dynamics to param history
        if self.param_history is not None:
            for param_name, dynamics_model in self.dynamics.items():
                if param_name in self.param_history.columns:
                    param_series = self.param_history[param_name].values
                    self._fitted_dynamics[param_name] = dynamics_model.fit(param_series)
        
        return self
    
    def project(
        self,
        horizon: int,
        n_paths: int = 1000,
        confidence_levels: Sequence[float] = (0.5, 0.9),
    ) -> Projection:
        """
        Project parameters forward using fitted dynamics.
        
        Returns:
            Projection object with paths, mean, quantiles at each horizon.
        """
        # Get current param values
        param_values = {}
        for field in dataclasses.fields(self.current_params):
            param_values[field.name] = getattr(self.current_params, field.name)
        
        # Project each parameter
        all_paths = {}
        for param_name, current_val in param_values.items():
            if param_name in self._fitted_dynamics:
                paths = self._fitted_dynamics[param_name].project(
                    current_val, horizon, n_paths
                )
            else:
                # If no dynamics, keep constant
                paths = np.full((n_paths, horizon), current_val)
            all_paths[param_name] = paths
        
        return Projection(
            param_paths=all_paths,
            horizon=horizon,
            n_paths=n_paths,
            confidence_levels=confidence_levels,
        )
    
    def predictive(self, t: int, n_samples: int = 10000) -> PredictiveDistribution:
        """
        Get predictive distribution at horizon t, integrating over parameter uncertainty.
        """
        projection = self.project(horizon=t, n_paths=1000)
        return PredictiveDistribution(
            distribution=self.distribution,
            param_paths=projection.param_paths,
            t=t,
            n_samples=n_samples,
        )
    
    def decision(
        self,
        t: int,
        alpha: float = 0.05,
        confidence_level: float = 0.90,
    ) -> DecisionSummary:
        """
        Compute trading decision metrics at horizon t with confidence intervals.
        """
        predictive = self.predictive(t)
        return predictive.decision_summary(alpha=alpha, confidence_level=confidence_level)
```

### 5.5 Module: `temporal/projection.py`

```python
@dataclass
class Projection:
    """Container for parameter projection results."""
    param_paths: dict[str, np.ndarray]  # {param_name: (n_paths, horizon)}
    horizon: int
    n_paths: int
    confidence_levels: Sequence[float]
    
    def mean(self, t: int | None = None) -> dict[str, float] | pd.DataFrame:
        """Mean params at time t (or all times if t is None)."""
        if t is not None:
            return {k: v[:, t-1].mean() for k, v in self.param_paths.items()}
        else:
            return pd.DataFrame({
                k: v.mean(axis=0) for k, v in self.param_paths.items()
            })
    
    def quantile(self, q: float, t: int | None = None) -> dict[str, float] | pd.DataFrame:
        """Quantile q of params at time t."""
        if t is not None:
            return {k: np.quantile(v[:, t-1], q) for k, v in self.param_paths.items()}
        else:
            return pd.DataFrame({
                k: np.quantile(v, q, axis=0) for k, v in self.param_paths.items()
            })
    
    def at(self, t: int) -> "ParamDistribution":
        """Get param distribution at specific time."""
        return ParamDistribution(
            values={k: v[:, t-1] for k, v in self.param_paths.items()},
            t=t,
        )
```

### 5.6 Module: `temporal/predictive.py`

```python
@dataclass
class PredictiveDistribution:
    """
    Predictive distribution integrating over parameter uncertainty.
    
    P(r_t) = integral P(r | theta) * P(theta | t) d_theta
    
    Approximated via Monte Carlo:
    - Sample param paths
    - For each path, sample from the conditional distribution
    - Aggregate samples
    """
    distribution: str
    param_paths: dict[str, np.ndarray]  # At specific t
    t: int
    n_samples: int = 10000
    _samples: np.ndarray | None = field(default=None, init=False)
    
    def _ensure_samples(self):
        if self._samples is None:
            dist = get_distribution(self.distribution)
            samples = []
            n_paths = list(self.param_paths.values())[0].shape[0]
            samples_per_path = self.n_samples // n_paths
            
            for i in range(n_paths):
                # Get params for this path at time t
                param_dict = {k: v[i, self.t - 1] for k, v in self.param_paths.items()}
                params = create_params(self.distribution, **param_dict)
                
                # Sample from distribution
                path_samples = dist.sample(samples_per_path, t=0, params=params)
                samples.extend(path_samples)
            
            self._samples = np.array(samples)
    
    def mean(self) -> float:
        self._ensure_samples()
        return np.mean(self._samples)
    
    def std(self) -> float:
        self._ensure_samples()
        return np.std(self._samples)
    
    def quantile(self, q: float) -> float:
        self._ensure_samples()
        return np.quantile(self._samples, q)
    
    def var(self, alpha: float = 0.05) -> float:
        """Value at Risk (loss threshold exceeded with probability alpha)."""
        return -self.quantile(alpha)
    
    def cvar(self, alpha: float = 0.05) -> float:
        """Conditional VaR (expected loss given we're in the tail)."""
        self._ensure_samples()
        threshold = self.quantile(alpha)
        tail_samples = self._samples[self._samples <= threshold]
        return -np.mean(tail_samples) if len(tail_samples) > 0 else -threshold
    
    def decision_summary(
        self,
        alpha: float = 0.05,
        confidence_level: float = 0.90,
    ) -> "DecisionSummary":
        """
        Compute full decision summary with confidence intervals.
        
        CIs are computed by bootstrapping over the parameter paths.
        """
        self._ensure_samples()
        
        # Point estimates
        var_val = self.var(alpha)
        cvar_val = self.cvar(alpha)
        kelly_val = self._compute_kelly()
        prob_profit = np.mean(self._samples > 0)
        
        # Bootstrap CIs
        n_bootstrap = 1000
        var_boots, cvar_boots, kelly_boots, prob_boots = [], [], [], []
        
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(self._samples), len(self._samples), replace=True)
            boot_samples = self._samples[boot_idx]
            
            var_boots.append(-np.quantile(boot_samples, alpha))
            threshold = np.quantile(boot_samples, alpha)
            tail = boot_samples[boot_samples <= threshold]
            cvar_boots.append(-np.mean(tail) if len(tail) > 0 else -threshold)
            kelly_boots.append(np.mean(boot_samples) / (np.var(boot_samples) + 1e-10))
            prob_boots.append(np.mean(boot_samples > 0))
        
        ci_low = (1 - confidence_level) / 2
        ci_high = 1 - ci_low
        
        return DecisionSummary(
            var=RiskMetric(
                value=var_val,
                confidence_interval=(np.quantile(var_boots, ci_low), np.quantile(var_boots, ci_high)),
            ),
            cvar=RiskMetric(
                value=cvar_val,
                confidence_interval=(np.quantile(cvar_boots, ci_low), np.quantile(cvar_boots, ci_high)),
            ),
            kelly=RiskMetric(
                value=kelly_val,
                confidence_interval=(np.quantile(kelly_boots, ci_low), np.quantile(kelly_boots, ci_high)),
            ),
            prob_profit=RiskMetric(
                value=prob_profit,
                confidence_interval=(np.quantile(prob_boots, ci_low), np.quantile(prob_boots, ci_high)),
            ),
            expected_return=self.mean(),
            volatility=self.std(),
            t=self.t,
            alpha=alpha,
        )
    
    def _compute_kelly(self) -> float:
        """Kelly fraction = mean / variance."""
        return self.mean() / (self.std()**2 + 1e-10)
```

---

## SECTION 6: STAGE 4 - DECISION LAYER

### 6.1 Dataclasses

```python
@dataclass(frozen=True)
class RiskMetric:
    """Container for a risk metric with uncertainty."""
    value: float
    confidence_interval: tuple[float, float] | None = None
    standard_error: float | None = None

@dataclass(frozen=True)
class DecisionSummary:
    """Complete decision output."""
    var: RiskMetric
    cvar: RiskMetric
    kelly: RiskMetric
    prob_profit: RiskMetric
    expected_return: float
    volatility: float
    t: int
    alpha: float
    
    def __str__(self) -> str:
        lines = [
            f"Decision Summary (t={self.t}, alpha={self.alpha})",
            "-" * 50,
            f"Expected Return: {self.expected_return:+.2%}",
            f"Volatility:      {self.volatility:.2%}",
            f"VaR ({1-self.alpha:.0%}):       {self.var.value:.2%} [{self.var.confidence_interval[0]:.2%}, {self.var.confidence_interval[1]:.2%}]",
            f"CVaR ({1-self.alpha:.0%}):      {self.cvar.value:.2%} [{self.cvar.confidence_interval[0]:.2%}, {self.cvar.confidence_interval[1]:.2%}]",
            f"Kelly:           {self.kelly.value:.2f} [{self.kelly.confidence_interval[0]:.2f}, {self.kelly.confidence_interval[1]:.2f}]",
            f"P(profit):       {self.prob_profit.value:.1%} [{self.prob_profit.confidence_interval[0]:.1%}, {self.prob_profit.confidence_interval[1]:.1%}]",
        ]
        return "\n".join(lines)
```

---

## SECTION 7: STAGE 5 - BACKTESTING

### 7.1 Module: `backtest/runner.py`

```python
@dataclass
class Backtest:
    """
    Rolling backtest framework.
    
    Usage:
        bt = Backtest(
            distribution="nig",
            lookback=252,
            forecast_horizon=5,
            alpha=0.05,
            weighting=EMA(halflife=20),
        )
        bt.run(returns)
        print(bt.summary())
    """
    distribution: str
    lookback: int
    forecast_horizon: int = 1
    alpha: float = 0.05
    weighting: WeightScheme | None = None
    dynamics: dict[str, DynamicsModel] | None = None
    step: int = 1
    
    # Results (populated after run)
    var_forecasts: np.ndarray | None = field(default=None, init=False)
    actual_returns: np.ndarray | None = field(default=None, init=False)
    exceedances: np.ndarray | None = field(default=None, init=False)
    crps_scores: np.ndarray | None = field(default=None, init=False)
    
    def run(self, data: np.ndarray) -> "Backtest":
        """Execute the backtest."""
        n = len(data)
        var_list, actual_list, exc_list, crps_list = [], [], [], []
        
        for i in range(self.lookback, n - self.forecast_horizon + 1, self.step):
            window = data[i - self.lookback:i]
            actual = data[i + self.forecast_horizon - 1]
            
            # Fit model
            if self.dynamics is not None:
                temporal = TemporalModel(
                    distribution=self.distribution,
                    weighting=self.weighting or EMA(halflife=20),
                    dynamics=self.dynamics,
                )
                temporal.fit(window)
                predictive = temporal.predictive(t=self.forecast_horizon)
                var_forecast = predictive.var(self.alpha)
            else:
                # Simple fit without dynamics
                params = fit(window, self.distribution)
                dist = get_distribution(self.distribution)
                var_forecast = -dist.ppf(self.alpha, t=0, params=params)
            
            var_list.append(var_forecast)
            actual_list.append(actual)
            exc_list.append(-actual > var_forecast)
        
        self.var_forecasts = np.array(var_list)
        self.actual_returns = np.array(actual_list)
        self.exceedances = np.array(exc_list)
        
        return self
    
    @property
    def exceedance_rate(self) -> float:
        return np.mean(self.exceedances)
    
    @property
    def n_exceedances(self) -> int:
        return np.sum(self.exceedances)
    
    @property
    def n_total(self) -> int:
        return len(self.exceedances)
    
    def kupiec_test(self) -> tuple[float, float, bool]:
        """
        Kupiec unconditional coverage test.
        
        Returns: (statistic, p-value, reject_null)
        """
        return kupiec_test(self.exceedances, self.alpha)
    
    def christoffersen_test(self) -> tuple[float, float, bool]:
        """
        Christoffersen independence test.
        
        Returns: (statistic, p-value, reject_null)
        """
        return christoffersen_test(self.exceedances)
    
    def summary(self) -> str:
        kup_stat, kup_p, kup_reject = self.kupiec_test()
        chr_stat, chr_p, chr_reject = self.christoffersen_test()
        
        status = "PASS" if not kup_reject and not chr_reject else "FAIL"
        
        lines = [
            "Backtest Summary",
            "=" * 50,
            f"Distribution:        {self.distribution}",
            f"Lookback:            {self.lookback}",
            f"Forecast horizon:    {self.forecast_horizon}",
            f"Alpha:               {self.alpha}",
            "",
            f"Expected exc. rate:  {self.alpha:.1%}",
            f"Actual exc. rate:    {self.exceedance_rate:.1%}",
            f"Exceedances:         {self.n_exceedances} / {self.n_total}",
            "",
            f"Kupiec p-value:      {kup_p:.4f} ({'PASS' if not kup_reject else 'FAIL'})",
            f"Christoffersen p:    {chr_p:.4f} ({'PASS' if not chr_reject else 'FAIL'})",
            "",
            f"Overall Status:      {status}",
        ]
        return "\n".join(lines)
```

### 7.2 Module: `backtest/tests.py`

```python
def kupiec_test(exceedances: np.ndarray, alpha: float, significance: float = 0.05) -> tuple[float, float, bool]:
    """
    Kupiec unconditional coverage test.
    
    H0: True exceedance rate equals alpha.
    
    Returns: (LR statistic, p-value, reject_null)
    """
    n = len(exceedances)
    n_exc = np.sum(exceedances)
    
    if n_exc == 0 or n_exc == n:
        return 0.0, 0.0, True  # Edge case
    
    p_hat = n_exc / n
    
    # Likelihood ratio
    lr = -2 * (
        n_exc * np.log(alpha / p_hat) + 
        (n - n_exc) * np.log((1 - alpha) / (1 - p_hat))
    )
    
    p_value = 1 - stats.chi2.cdf(lr, 1)
    reject = p_value < significance
    
    return lr, p_value, reject


def christoffersen_test(exceedances: np.ndarray, significance: float = 0.05) -> tuple[float, float, bool]:
    """
    Christoffersen independence test.
    
    H0: Exceedances are independent (no clustering).
    
    Returns: (LR statistic, p-value, reject_null)
    """
    # Count transitions
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(exceedances)):
        prev, curr = int(exceedances[i-1]), int(exceedances[i])
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1
    
    # Edge cases
    if n00 + n01 == 0 or n10 + n11 == 0:
        return 0.0, 1.0, False
    
    # Transition probabilities
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Likelihood ratio
    if pi01 == 0 or pi01 == 1 or pi11 == 0 or pi11 == 1 or pi == 0 or pi == 1:
        return 0.0, 1.0, False
    
    lr = -2 * (
        n00 * np.log(1 - pi) + n01 * np.log(pi) + 
        n10 * np.log(1 - pi) + n11 * np.log(pi) -
        n00 * np.log(1 - pi01) - n01 * np.log(pi01) -
        n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    
    p_value = 1 - stats.chi2.cdf(abs(lr), 1)
    reject = p_value < significance
    
    return lr, p_value, reject
```

---

## SECTION 8: HIGH-LEVEL API

### Module: `api.py`

```python
"""
High-level API facade.

Usage:
    import temporalpdf as tpdf
    
    # Discovery
    result = tpdf.discover(returns)
    
    # Simple fit
    params = tpdf.fit(returns, distribution="nig")
    
    # Full temporal model
    temporal = tpdf.TemporalModel(...)
    temporal.fit(returns)
    decision = temporal.decision(t=5)
    
    # Backtest
    bt = tpdf.backtest(returns, distribution="nig", lookback=252)
"""

from .discovery.selection import discover, DiscoveryResult
from .fitting.mle import fit
from .temporal.model import TemporalModel
from .temporal.weights import SMA, EMA, Linear, PowerDecay, Gaussian, Custom
from .temporal.dynamics import Constant, RandomWalk, MeanReverting, AR, GARCH
from .backtest.runner import Backtest

def backtest(
    data: np.ndarray,
    distribution: str = "nig",
    lookback: int = 252,
    forecast_horizon: int = 1,
    alpha: float = 0.05,
    **kwargs,
) -> Backtest:
    """Convenience function to create and run a backtest."""
    bt = Backtest(
        distribution=distribution,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        alpha=alpha,
        **kwargs,
    )
    bt.run(data)
    return bt
```

---

## SECTION 9: IMPLEMENTATION PHASES

### Phase 1: Foundations (FIRST)
1. Update `core/parameters.py` - ensure all param classes are frozen dataclasses
2. Update `core/distribution.py` - ensure Protocol is correct
3. Create `core/result.py` - add RiskMetric, DecisionSummary dataclasses
4. Replace `distributions/normal.py`, `student_t.py`, `skew_normal.py` with scipy wrappers
5. Keep `distributions/nig.py` and `generalized_laplace.py` as-is

### Phase 2: Temporal Core (SECOND)
1. Create `temporal/weights/` module with all weighting schemes
2. Create `temporal/tracking/tracker.py` - ParameterTracker
3. Write tests for weighting schemes

### Phase 3: Dynamics (THIRD)
1. Create `temporal/dynamics/` module
2. Implement: Constant, RandomWalk, MeanReverting, AR, GARCH
3. Create `temporal/projection.py`
4. Create `temporal/predictive.py`
5. Create `temporal/model.py` - TemporalModel class

### Phase 4: Decision (FOURTH)
1. Update `decision/risk.py` - return RiskMetric instead of float
2. Update `decision/kelly.py` - return RiskMetric
3. Integrate with TemporalModel

### Phase 5: Discovery & Backtest (FIFTH)
1. Move selection logic from `utilities.py` to `discovery/selection.py`
2. Create `discovery/significance.py`
3. Create `backtest/runner.py`
4. Create `backtest/tests.py` - Kupiec, Christoffersen

### Phase 6: Conditional (DEFER)
1. Create `conditional/model.py` - wrapper for XGBoostLSS
2. Only implement when Phases 1-5 are complete

---

## SECTION 10: DEPENDENCIES

### Required
```toml
[project]
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "scipy>=1.7",
    "scoringrules>=0.1",  # NOT properscoring
]
```

### Optional
```toml
[project.optional-dependencies]
dynamics = ["arch>=5.0", "statsmodels>=0.13"]
conditional = ["xgboostlss>=0.4", "shap>=0.40"]
visualization = ["matplotlib>=3.4", "plotly>=5.0"]
```

---

## SECTION 11: TESTING REQUIREMENTS

Every module MUST have corresponding tests in `tests/`.

```
tests/
├── test_discovery.py      # Test discover(), significance tests
├── test_fitting.py        # Test fit() for all distributions
├── test_weights.py        # Test all weighting schemes
├── test_dynamics.py       # Test all dynamics models
├── test_temporal.py       # Test TemporalModel, Projection, Predictive
├── test_decision.py       # Test VaR, CVaR, Kelly with CIs
└── test_backtest.py       # Test Backtest, Kupiec, Christoffersen
```

Test pattern:
```python
def test_ema_weights_sum_to_one():
    ema = EMA(halflife=20)
    weights = ema.get_weights(100)
    assert np.isclose(weights.sum(), 1.0)

def test_ema_most_recent_has_highest_weight():
    ema = EMA(halflife=20)
    weights = ema.get_weights(100)
    assert weights[0] > weights[1] > weights[2]

def test_mean_reverting_halflife():
    mr = MeanReverting(kappa=0.1, long_run=10.0, sigma=0.5)
    assert np.isclose(mr.half_life(), np.log(2) / 0.1)
```

---

## SECTION 12: FORBIDDEN PATTERNS

### DO NOT
- Return raw floats from risk functions (use RiskMetric)
- Use magic strings for distribution names (use Literal types)
- Create mutable parameter classes (use frozen=True)
- Implement CRPS from scratch (use scoringrules)
- Implement Normal/Student-t/Skew-Normal from scratch (use scipy wrappers)
- Skip confidence intervals on decision outputs
- Ignore the weighting scheme in fitting

### ALWAYS
- Type hint every function
- Use dataclasses for structured data
- Return uncertainty with point estimates
- Test edge cases (empty data, single observation, etc.)
- Follow the 5-stage pipeline structure

---

END OF SPECIFICATION
