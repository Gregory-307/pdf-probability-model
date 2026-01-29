# Extending temporalpdf

How to add new distributions, volatility models, dynamics models, weighting schemes, and scoring rules to the library. Each section gives the exact interface to satisfy, what works automatically after implementation, what requires manual wiring, and a minimal working example.

---

## Extension Points at a Glance

| Extension Point | Interface Type | Location | What It Controls |
|---|---|---|---|
| **Distribution** | Protocol (V2) or ABC (V1) | `core/distribution.py` | What probability distributions are available |
| **Volatility Model** | ABC | `core/volatility.py` | How the scale parameter evolves within a forecast horizon |
| **Dynamics Model** | Protocol | `temporal/dynamics/base.py` | How fitted parameters evolve across sequential forecasts |
| **Weighting Scheme** | Protocol | `temporal/weights/base.py` | How observations are weighted during parameter estimation |
| **Scoring Rule** | Protocol | `scoring/rules.py` | How distributional predictions are evaluated |

The library uses two interface patterns:

- **Protocol** (structural typing): Any class with the right methods works. No inheritance required.
- **ABC** (abstract base class): Must explicitly inherit and implement abstract methods.

For distribution classes specifically, there are two API generations. V2 (Protocol) is the minimal interface for new code. V1 (ABC) adds properties and a `pdf_matrix` method needed for visualisation and some legacy code paths.

---

## Adding a New Distribution

This is the most involved extension. A distribution touches multiple subsystems, some of which accept any protocol-satisfying object and some of which use hard-coded name lookups.

### Step 1: Define the Parameter Class

Create a frozen dataclass inheriting from `DistributionParameters`:

```python
# distributions/logistic.py

from dataclasses import dataclass
from ..core.distribution import DistributionParameters


@dataclass(frozen=True)
class LogisticParameters(DistributionParameters):
    """Parameters for the Logistic distribution."""

    mu_0: float        # Location parameter
    s_0: float         # Scale parameter (> 0)
    delta: float = 0.0 # Location drift per unit time
    beta: float = 0.0  # Scale growth rate

    def __post_init__(self) -> None:
        if self.s_0 <= 0:
            raise ValueError("s_0 must be positive")

    def with_mu_0(self, new_mu_0: float) -> "LogisticParameters":
        return LogisticParameters(
            mu_0=new_mu_0, s_0=self.s_0, delta=self.delta, beta=self.beta
        )

    def with_s_0(self, new_s_0: float) -> "LogisticParameters":
        return LogisticParameters(
            mu_0=self.mu_0, s_0=new_s_0, delta=self.delta, beta=self.beta
        )
```

**Required conventions:**

- `frozen=True` — parameters are never mutated; create new objects instead.
- `__post_init__` — validate constraints (positive scale, bounded shape parameters, etc.).
- `with_*()` methods — return a new instance with one field changed. The temporal pipeline uses these to reconstruct parameter objects from projected float arrays.
- `delta` and `beta` fields — if your distribution supports time evolution, include these with `0.0` defaults. They control `mu(t) = mu_0 + delta * t` and `sigma(t) = sigma_0 * (1 + beta * t)`.

**Note on NIGParameters:** The NIG parameter class intentionally does *not* inherit from `DistributionParameters`. This is a historical anomaly — it predates the base class and has a different field naming convention (`mu` instead of `mu_0`, `delta` instead of `sigma_0`). New distributions should inherit from `DistributionParameters`.

### Step 2: Implement the Distribution Class

You can implement either the V2 Protocol (minimal) or the V1 ABC (full). The V1 ABC is recommended for distributions that will be used with the visualisation pipeline (`pdf_matrix`, `evaluate()`).

#### V2 Protocol — Minimal Interface

Satisfy these four methods and the decision functions (`var`, `cvar`, `kelly_fraction`, `prob_greater_than`, etc.) will accept your distribution immediately:

```python
class Distribution(Protocol):
    def pdf(self, x: ArrayLike, t: float, params: DistributionParameters) -> ArrayLike: ...
    def cdf(self, x: ArrayLike, t: float, params: DistributionParameters) -> ArrayLike: ...
    def ppf(self, q: ArrayLike, t: float, params: DistributionParameters) -> ArrayLike: ...
    def sample(self, n: int, t: float, params: DistributionParameters) -> ArrayLike: ...
```

#### V1 ABC — Full Interface

Inherit from `TimeEvolvingDistribution[YourParams]` and implement:

| Member | Type | Required | Default Implementation |
|---|---|---|---|
| `name` | Property | Yes | — |
| `parameter_names` | Property | Yes | — |
| `pdf(x, t, params)` | Method | Yes | — |
| `pdf_matrix(x, time_grid, params)` | Method | Yes | — |
| `cdf(x, t, params)` | Method | Yes (V2) | — |
| `ppf(q, t, params)` | Method | Yes (V2) | — |
| `sample(n, t, params)` | Method | Yes (V2) | — |
| `expected_value(x, t, params)` | Method | No | Numerical integration via `np.trapezoid` |
| `variance(x, t, params)` | Method | No | Numerical integration via `np.trapezoid` |
| `std(x, t, params)` | Method | No | `sqrt(variance())` |

**Note:** `cdf`, `ppf`, and `sample` are required by the V2 Protocol but are not abstract methods on the V1 ABC. You must implement them for the decision functions to work. The Normal distribution (`distributions/normal.py`) is the clearest reference implementation.

#### Complete Example: Logistic Distribution

```python
# distributions/logistic.py

import numpy as np
from numpy.typing import NDArray
from scipy.stats import logistic

from ..core.distribution import TimeEvolvingDistribution
from .logistic_params import LogisticParameters  # or define in same file


class LogisticDistribution(TimeEvolvingDistribution[LogisticParameters]):
    """Time-evolving Logistic distribution."""

    @property
    def name(self) -> str:
        return "Logistic"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu_0", "s_0", "delta", "beta")

    def _params_at_t(self, t: float, params: LogisticParameters):
        mu_t = params.mu_0 + params.delta * t
        s_t = params.s_0 * (1 + params.beta * t)
        return mu_t, s_t

    def pdf(self, x: NDArray[np.float64], t: float,
            params: LogisticParameters) -> NDArray[np.float64]:
        mu_t, s_t = self._params_at_t(t, params)
        return logistic.pdf(x, loc=mu_t, scale=s_t)

    def pdf_matrix(self, x: NDArray[np.float64],
                   time_grid: NDArray[np.float64],
                   params: LogisticParameters) -> NDArray[np.float64]:
        result = np.zeros((len(time_grid), len(x)))
        for i, t in enumerate(time_grid):
            result[i, :] = self.pdf(x, float(t), params)
        return result

    def cdf(self, x: NDArray[np.float64], t: float,
            params: LogisticParameters) -> NDArray[np.float64]:
        mu_t, s_t = self._params_at_t(t, params)
        return logistic.cdf(x, loc=mu_t, scale=s_t)

    def ppf(self, q: NDArray[np.float64], t: float,
            params: LogisticParameters) -> NDArray[np.float64]:
        mu_t, s_t = self._params_at_t(t, params)
        return logistic.ppf(q, loc=mu_t, scale=s_t)

    def sample(self, n: int, t: float, params: LogisticParameters,
               rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        mu_t, s_t = self._params_at_t(t, params)
        if rng is None:
            rng = np.random.default_rng()
        return rng.logistic(loc=mu_t, scale=s_t, size=n)
```

### Step 3: Register with the Registry

```python
from temporalpdf.distributions.registry import DistributionRegistry

DistributionRegistry.register("logistic", LogisticDistribution)
```

For built-in distributions, add the registration call to `_register_builtins()` in `distributions/registry.py`. For external distributions, call `register()` at import time in your own code.

### What Works Automatically After Steps 1-3

Once your distribution satisfies the V2 Protocol, these subsystems accept it with no additional code:

| Subsystem | What Works | Why |
|---|---|---|
| `var(dist, params)` | VaR computation | Calls `dist.ppf()` |
| `cvar(dist, params)` | CVaR computation | Calls `dist.ppf()` + `dist.pdf()` |
| `kelly_fraction(dist, params)` | Kelly sizing | Requires `dist.mean(t, params)` and `dist.variance(t, params)` — see note below |
| `prob_greater_than(dist, params, threshold)` | Tail probability | Calls `dist.cdf()` |
| `var_with_ci(dist, params)` | VaR with confidence intervals | Calls `dist.sample()` |
| `CRPS()(dist, params, y)` | Scoring rule evaluation | Calls `dist.cdf()` + `dist.ppf()` |
| `LogScore()(dist, params, y)` | Scoring rule evaluation | Calls `dist.pdf()` |
| `evaluate(name, params)` | PDF surface visualisation | Calls `dist.pdf_matrix()` (V1 only) |

**Kelly criterion note:** `kelly_fraction()` uses its own local Protocol requiring `mean(t, params) -> float` and `variance(t, params) -> float`. These are **not** part of the V2 Protocol or the V1 ABC (which has `expected_value(x, t, params)` taking an `x` grid). NIG implements analytical `mean()` and `variance()` directly. For other distributions, you must add these methods if you want Kelly sizing to work. The `kelly_with_ci()` variant uses sampling instead and only requires `dist.sample()`.

### What Requires Manual Wiring

Several subsystems use hard-coded if/elif dispatch on distribution name strings. To fully integrate a new distribution, you must add branches to:

| File | Function | What to Add |
|---|---|---|
| `utilities.py` | `fit()` | An `elif distribution == "logistic"` branch calling your MLE fitter |
| `discovery/selection.py` | `_fit_distribution()` | An `elif dist_name == "logistic"` branch |
| `discovery/selection.py` | `_score_distribution()` | CRPS and log_score computation for your distribution |
| `temporal/predictive.py` | `get_distribution()` | An entry in the `distributions` dict |
| `temporal/predictive.py` | `create_params()` | An `elif distribution == "logistic"` branch |
| `ml.py` | `_apply_constraints()` | Parameter constraint logic for training |
| `ml.py` | `_sample_distribution()` | Differentiable sampling in PyTorch |
| `backtest/runner.py` | `Backtest._compute_var()` | VaR computation for your distribution |

Not all of these are required for every use case. If you only need the decision functions (`var`, `cvar`, etc.) with manually constructed parameters, steps 1-3 are sufficient. The manual wiring is only needed for the subsystems that resolve distributions by name.

### Distributions Without Analytical PPF

If your distribution has no closed-form quantile function, use numerical inversion. The NIG distribution does this via `scipy.optimize.brentq`:

```python
from scipy.optimize import brentq

def ppf(self, q, t, params):
    result = np.zeros_like(q)
    for i, qi in enumerate(q):
        def objective(x):
            return float(self.cdf(np.array([x]), t, params)[0]) - qi
        result[i] = brentq(objective, -10, 10)  # adjust bounds as needed
    return result
```

This is slower than analytical PPF but works for any distribution with a valid CDF.

### Distributions Without Closure Under Convolution

If your distribution does not have a known form for sums of random variables, the Monte Carlo fallback path handles it automatically. The `PredictiveDistribution` in `temporal/predictive.py` works by sampling from the distribution — it does not require analytical convolution. Similarly, `crps_from_samples()` in `discovery/scoring.py` computes CRPS from Monte Carlo samples rather than requiring a closed-form expression.

### Testing Checklist

For any new distribution, verify:

- [ ] **PDF integrates to 1**: `np.trapezoid(dist.pdf(x, t=0, params), x) ≈ 1.0` over a wide enough grid
- [ ] **CDF is monotone and bounded**: values in [0, 1], non-decreasing
- [ ] **CDF endpoints**: `cdf(-∞) ≈ 0`, `cdf(+∞) ≈ 1`
- [ ] **PPF inverts CDF**: `dist.cdf(dist.ppf(q, t, params), t, params) ≈ q` for q in (0, 1)
- [ ] **Sample statistics match moments**: mean and variance of `dist.sample(100000, ...)` match analytical expectations
- [ ] **Time evolution**: at `t=0` with `delta=0, beta=0`, the distribution matches its static form
- [ ] **Parameter validation**: invalid parameters (negative scale, etc.) raise `ValueError`
- [ ] **Frozen immutability**: modifying a parameter field raises `FrozenInstanceError`

---

## Adding a New Volatility Model

Volatility models control how the scale parameter evolves within a single forecast horizon. They are used by NIG's `_delta_at_time(t)` method.

### Interface

Inherit from the `VolatilityModel` ABC in `core/volatility.py`:

```python
class VolatilityModel(ABC):

    @abstractmethod
    def at_time(self, sigma_0: float, t: float) -> float:
        """Compute volatility at time t given initial volatility sigma_0."""
        ...

    @abstractmethod
    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        """Vectorized version of at_time."""
        ...
```

### Pattern

All built-in volatility models are frozen dataclasses that also inherit the ABC:

```python
@dataclass(frozen=True)
class MyVolModel(VolatilityModel):
    ...
```

### Attachment

Set on the `volatility_model` field of `NIGParameters`:

```python
params = NIGParameters(
    mu=0, delta=0.02, alpha=15, beta=-2,
    volatility_model=MyVolModel(...)
)
```

Currently only NIG supports the `volatility_model` field. Other distributions use the simpler `beta` field for linear scale growth.

### Minimal Example: Regime-Switching Volatility

```python
from dataclasses import dataclass
import numpy as np
from temporalpdf.core.volatility import VolatilityModel


@dataclass(frozen=True)
class RegimeSwitching(VolatilityModel):
    """Two-regime volatility: low vol until switch_time, high vol after."""

    high_vol_multiplier: float = 2.0
    switch_time: float = 10.0

    def at_time(self, sigma_0: float, t: float) -> float:
        if t < self.switch_time:
            return sigma_0
        return sigma_0 * self.high_vol_multiplier

    def at_times(self, sigma_0: float, t: np.ndarray) -> np.ndarray:
        result = np.full_like(t, sigma_0, dtype=float)
        result[t >= self.switch_time] = sigma_0 * self.high_vol_multiplier
        return result
```

Usage:

```python
from temporalpdf.distributions.nig import NIGParameters

params = NIGParameters(
    mu=0, delta=0.02, alpha=15, beta=-2,
    volatility_model=RegimeSwitching(high_vol_multiplier=1.5, switch_time=20),
)
# At t=0: delta = 0.02
# At t=25: delta = 0.02 * 1.5 = 0.03
```

---

## Adding a New Dynamics Model

Dynamics models control how a fitted parameter evolves across sequential re-fits (inter-forecast time evolution). They are used by `TemporalModel.project()` to simulate future parameter paths.

### Interface

Satisfy the `DynamicsModel` Protocol in `temporal/dynamics/base.py`:

```python
class DynamicsModel(Protocol):

    def fit(self, param_series: NDArray[np.float64]) -> "DynamicsModel":
        """Fit to historical parameter values. Return self."""
        ...

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """Return (n_paths, horizon) array of simulated future values."""
        ...

    def summary(self) -> dict[str, float]:
        """Return fitted parameters as a dict."""
        ...
```

### Pattern

Built-in dynamics models are dataclasses with optional pre-set parameters (set via `__init__`) and fitted state (set in `fit()`). The `fit()` method returns `self` for method chaining.

### Integration

Pass dynamics models in the `dynamics` dict when constructing `TemporalModel`:

```python
model = TemporalModel(
    distribution="nig",
    tracking=ParameterTracker("nig", window=60, step=1),
    dynamics={
        "mu": Constant(),
        "delta": MyDynamicsModel(...),
        "alpha": MeanReverting(),
        "beta": Constant(),
    },
)
```

Each key is a parameter name; each value is a dynamics model that will be fitted to the historical series of that parameter.

### Minimal Example: Exponential Smoothing

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class ExponentialSmoothing:
    """Simple exponential smoothing dynamics."""

    alpha: float = 0.3         # Smoothing factor
    _level: float | None = None
    _sigma: float | None = None

    def fit(self, param_series: NDArray[np.float64]) -> "ExponentialSmoothing":
        # Compute smoothed level
        level = param_series[0]
        for val in param_series[1:]:
            level = self.alpha * val + (1 - self.alpha) * level
        self._level = float(level)
        # Estimate residual volatility
        diffs = np.diff(param_series)
        self._sigma = float(np.std(diffs)) if len(diffs) > 0 else 0.01
        return self

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        paths = np.zeros((n_paths, horizon))
        paths[:, 0] = current_value
        noise = np.random.randn(n_paths, horizon) * self._sigma
        for t in range(1, horizon):
            paths[:, t] = (
                self.alpha * paths[:, t - 1]
                + (1 - self.alpha) * self._level
                + noise[:, t]
            )
        return paths

    def summary(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "level": self._level or 0.0,
            "sigma": self._sigma or 0.0,
        }
```

---

## Adding a New Weighting Scheme

Weighting schemes control how observations are weighted during parameter estimation. They are passed to `TemporalModel` via the `weighting` argument.

### Interface

Satisfy the `WeightScheme` Protocol in `temporal/weights/base.py`:

```python
class WeightScheme(Protocol):

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return array of n weights summing to 1. Index 0 = most recent."""
        ...

    def effective_sample_size(self, n: int) -> float:
        """Return ESS = 1 / sum(w_i^2)."""
        ...
```

### Conventions

- **Index 0** is the most recent observation. Index `n-1` is the oldest.
- **Weights sum to 1.** Normalise before returning.
- **ESS formula:** `1.0 / np.sum(weights ** 2)`. This is standard across all built-in schemes.

### Minimal Example: Triangular Weights

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class Triangular:
    """Triangular weighting: peaks at most recent, linearly decays to zero."""

    def get_weights(self, n: int) -> NDArray[np.float64]:
        raw = np.arange(n, 0, -1, dtype=np.float64)  # [n, n-1, ..., 1]
        return raw / raw.sum()

    def effective_sample_size(self, n: int) -> float:
        w = self.get_weights(n)
        return 1.0 / np.sum(w ** 2)
```

Usage:

```python
model = TemporalModel(
    distribution="nig",
    tracking=ParameterTracker("nig", window=60, step=1),
    weighting=Triangular(),
    dynamics={"mu": Constant(), "delta": Constant()},
)
```

**Note:** In the current implementation, weighting is computed but `fit()` does not yet incorporate the weights into MLE — this is a documented TODO. The weighting infrastructure is in place for when this is implemented.

---

## Adding a New Scoring Rule

Scoring rules evaluate how well a predicted distribution matches observed data. They are used by the discovery pipeline for distribution selection and can be used independently for model monitoring.

### Interface

Satisfy the `ScoringRule` Protocol in `scoring/rules.py`:

```python
class ScoringRule(Protocol):

    @property
    def name(self) -> str: ...

    @property
    def is_proper(self) -> bool: ...

    def __call__(
        self,
        dist: Distribution,
        params: object,
        y: float | NDArray[np.float64],
        t: float,
    ) -> float | NDArray[np.float64]: ...
```

A scoring rule takes a distribution, parameters, observed value(s), and a time point, and returns a score. Lower scores indicate better predictions (by convention in this library).

### What Makes a Proper Scoring Rule

A scoring rule `S(P, y)` is **strictly proper** if the expected score `E_Q[S(P, Y)]` is uniquely minimised when `P = Q` — that is, the forecaster's best strategy is to report their true belief. CRPS and Log Score are both strictly proper. If your rule is not proper, set `is_proper = False`.

### Integration with Discovery

The discovery pipeline (`discovery/selection.py`) uses hard-coded scoring rules (`"crps"` and `"log_score"`). Adding a new scoring rule to the discovery pipeline requires modifying `_score_distribution()` in `selection.py` to add an `if scoring_name == "my_score"` branch. The `ScoringRule` protocol is for standalone use and custom evaluation loops.

### Minimal Example: Dawid-Sebastiani Score

The Dawid-Sebastiani score uses only the first two moments: `DSS = log(sigma^2) + (y - mu)^2 / sigma^2`.

```python
import numpy as np
from numpy.typing import NDArray


class DawidSebastiani:
    """
    Dawid-Sebastiani Score.

    Uses only mean and variance. Proper but not strictly proper
    (indifferent between distributions with same first two moments).
    """

    @property
    def name(self) -> str:
        return "Dawid-Sebastiani"

    @property
    def is_proper(self) -> bool:
        return True  # proper but not strictly proper

    def __call__(
        self,
        dist,
        params: object,
        y: float | NDArray[np.float64],
        t: float = 0.0,
    ) -> float | NDArray[np.float64]:
        # Requires dist to have expected_value(x, t, params) and variance(x, t, params)
        # These are provided by the V1 ABC (TimeEvolvingDistribution)
        x_grid = np.linspace(-1, 1, 1000)  # integration grid
        mu = dist.expected_value(x_grid, t, params)
        var = dist.variance(x_grid, t, params)
        if var <= 0:
            return float("inf")
        return np.log(var) + (y - mu) ** 2 / var
```

Usage:

```python
dss = DawidSebastiani()
score = dss(nig_dist, params, y_observed, t=0)
```

### PyTorch Scoring (ML Training)

The CRPS loss used for training `DistributionalRegressor` in `ml.py` is a separate PyTorch implementation, not the same as the numpy-based scoring rules. If you want to train an ML model with a custom loss, you must implement it in PyTorch with differentiable operations in `ml.py:_crps_loss_via_sampling()` or add a parallel loss function. The scoring module's numpy implementations are for evaluation only.
