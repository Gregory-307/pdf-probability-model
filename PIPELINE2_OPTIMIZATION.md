# Pipeline 2 Optimization Plan

## CRITICAL CONTEXT - READ FIRST

### What is Pipeline 2?
```
Features → Model → Distribution Parameters (μ, σ, ν) → Distribution → Simulate → P(event)
```

Pipeline 2 predicts **distribution parameters**, not the target directly. This gives you:
- Point prediction (mean/median of distribution)
- Uncertainty (standard deviation, quantiles)
- Risk metrics (VaR, CVaR)
- Full distributional shape

### Current Implementation Problem
```
Current: MSE loss on (μ, σ, ν) separately via MultiOutputRegressor
Problem: MSE on parameters ≠ good probabilistic predictions
```

The model minimizes parameter error, but we evaluate on barrier probability accuracy. Train-test mismatch.

### The Solution: DistributionalRegressor
Build a PyTorch wrapper that:
- Takes ANY temporalpdf distribution
- Uses CRPS loss via sampling (proper scoring rule)
- Learns parameters jointly via neural network

---

## IMPLEMENTATION STATUS

| Priority | Description | Status | Files |
|----------|-------------|--------|-------|
| 1 | DistributionalRegressor (CRPS Training) | **DONE** | `ml.py` |
| 2 | Analytical Barrier Approximations | **DONE** | `utilities.py` |
| 3 | Importance Sampling | **DONE** | `utilities.py` |
| 4 | Quasi-Monte Carlo | **DONE** | `utilities.py` |
| 5 | Calibration Features | **DONE** | `features.py` |
| 6 | Conformal Prediction | **DONE** | `utilities.py` |
| 7 | Temporal Dynamics | **DONE** | `utilities.py` |
| 8 | End-to-End Training | **DONE** | `ml.py` |

**Last Updated**: 2026-01-21

---

## PRIORITY 1: DistributionalRegressor (CRPS Training)

### Goal
Replace `MultiOutputRegressor(GBR)` with proper probabilistic training.

### API Design
```python
from temporalpdf.ml import DistributionalRegressor

model = DistributionalRegressor(
    distribution="student_t",  # or "nig", "normal", etc.
    loss="crps",               # or "log_score"
    base_learner="mlp",        # or "gbm" (future)
)
model.fit(X_train, y_train_returns)  # Raw target values, not pre-fitted params
params = model.predict(X_test)       # Returns StudentTParameters objects
```

### Implementation Details

**File: `src/temporalpdf/ml.py`**

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Union

class DistributionalMLP(nn.Module):
    """Neural network that outputs distribution parameters."""

    def __init__(self, n_features: int, distribution: str, hidden_dims: list = [64, 32]):
        super().__init__()

        self.distribution = distribution

        # Shared feature extraction
        layers = []
        prev_dim = n_features
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        # Output heads for each parameter
        if distribution == "student_t":
            self.n_params = 3  # μ, σ, ν
        elif distribution == "normal":
            self.n_params = 2  # μ, σ
        elif distribution == "nig":
            self.n_params = 4  # μ, δ, α, β

        self.output = nn.Linear(hidden_dims[-1], self.n_params)

    def forward(self, x):
        """Returns raw parameters (before constraints)."""
        shared = self.shared(x)
        raw_params = self.output(shared)
        return self._apply_constraints(raw_params)

    def _apply_constraints(self, raw):
        """Apply parameter constraints via smooth transforms."""
        if self.distribution == "student_t":
            mu = raw[:, 0]
            sigma = nn.functional.softplus(raw[:, 1]) + 1e-6  # σ > 0
            nu = nn.functional.softplus(raw[:, 2]) + 2.0      # ν > 2
            return torch.stack([mu, sigma, nu], dim=1)

        elif self.distribution == "normal":
            mu = raw[:, 0]
            sigma = nn.functional.softplus(raw[:, 1]) + 1e-6
            return torch.stack([mu, sigma], dim=1)

        elif self.distribution == "nig":
            mu = raw[:, 0]
            delta = nn.functional.softplus(raw[:, 1]) + 1e-6   # δ > 0
            alpha = nn.functional.softplus(raw[:, 2]) + 1e-6   # α > 0
            beta = raw[:, 3]  # β can be any real (but |β| < α)
            # Clamp beta to valid range
            beta = torch.clamp(beta, -alpha + 1e-6, alpha - 1e-6)
            return torch.stack([mu, delta, alpha, beta], dim=1)


def crps_loss_via_sampling(params, y_true, distribution: str, n_samples: int = 100):
    """
    CRPS via sampling - distribution agnostic.

    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    First term: expected distance from prediction to truth
    Second term: expected spread (rewards sharpness)
    """
    samples = sample_distribution(params, distribution, n_samples)  # (n_samples, batch)

    # Term 1: E|X - y|
    term1 = torch.abs(samples - y_true.unsqueeze(0)).mean(dim=0)

    # Term 2: E|X - X'| (use two independent sample sets)
    half = n_samples // 2
    term2 = torch.abs(samples[:half] - samples[half:2*half]).mean(dim=0)

    return (term1 - 0.5 * term2).mean()


def sample_distribution(params, distribution: str, n_samples: int):
    """
    Reparameterized sampling for gradients to flow through.
    """
    batch_size = params.shape[0]

    if distribution == "normal":
        mu, sigma = params[:, 0], params[:, 1]
        z = torch.randn(n_samples, batch_size, device=params.device)
        return mu + sigma * z

    elif distribution == "student_t":
        mu, sigma, nu = params[:, 0], params[:, 1], params[:, 2]
        # Reparameterization: X = μ + σ * Z * sqrt(ν/V) where Z~N(0,1), V~χ²(ν)
        z = torch.randn(n_samples, batch_size, device=params.device)
        # Chi-squared via gamma: χ²(ν) = Gamma(ν/2, 1/2)
        chi2 = torch.distributions.Chi2(nu)
        v = chi2.rsample((n_samples,))  # (n_samples, batch)
        return mu + sigma * z * torch.sqrt(nu / v)

    elif distribution == "nig":
        mu, delta, alpha, beta = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        # NIG reparameterization via inverse Gaussian mixture
        # X = μ + β*Z + sqrt(Z)*N where Z ~ IG(δ, sqrt(α²-β²))
        gamma = torch.sqrt(alpha**2 - beta**2)

        # Sample inverse Gaussian (approximate via transformation)
        u = torch.randn(n_samples, batch_size, device=params.device)
        y = u ** 2
        x = delta/gamma + y/(2*gamma**2) - torch.sqrt(y*(4*delta*gamma + y)) / (2*gamma**2)

        # Mixture
        z_ig = torch.where(torch.rand_like(x) < delta/(delta + x*gamma), x, delta**2/x)
        z_normal = torch.randn(n_samples, batch_size, device=params.device)

        return mu + beta * z_ig + torch.sqrt(z_ig) * z_normal


class DistributionalRegressor:
    """
    Distribution-agnostic regressor with proper scoring rule training.

    Instead of predicting parameters with MSE, trains with CRPS or log score
    to directly optimize probabilistic accuracy.
    """

    def __init__(
        self,
        distribution: Literal["normal", "student_t", "nig"] = "student_t",
        loss: Literal["crps", "log_score"] = "crps",
        base_learner: Literal["mlp"] = "mlp",
        hidden_dims: list = [64, 32],
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 32,
        n_samples: int = 100,  # For CRPS estimation
        device: str = "cpu",
    ):
        self.distribution = distribution
        self.loss = loss
        self.base_learner = base_learner
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.device = device

        self.model = None
        self.optimizer = None

    def fit(self, X, y):
        """
        Train the model on features X and target values y.

        Note: y should be raw target values (e.g., returns), NOT pre-fitted parameters.
        The model learns to predict distribution parameters that best explain y.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        n_features = X.shape[1]
        self.model = DistributionalMLP(n_features, self.distribution, self.hidden_dims)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()

                params = self.model(X_batch)

                if self.loss == "crps":
                    loss = crps_loss_via_sampling(params, y_batch, self.distribution, self.n_samples)
                elif self.loss == "log_score":
                    loss = negative_log_likelihood(params, y_batch, self.distribution)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss/len(loader):.4f}")

        return self

    def predict(self, X):
        """
        Predict distribution parameters for new data.

        Returns: Array of shape (n_samples, n_params) with constrained parameters.
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            params = self.model(X)

        return params.cpu().numpy()

    def predict_distribution(self, X):
        """
        Predict and return temporalpdf parameter objects.
        """
        import temporalpdf as tpdf

        params_array = self.predict(X)

        if self.distribution == "student_t":
            return [tpdf.StudentTParameters(mu_0=p[0], sigma_0=p[1], nu=p[2])
                    for p in params_array]
        elif self.distribution == "normal":
            return [tpdf.NormalParameters(mu=p[0], sigma=p[1])
                    for p in params_array]
        elif self.distribution == "nig":
            return [tpdf.NIGParameters(mu=p[0], delta=p[1], alpha=p[2], beta=p[3])
                    for p in params_array]


def negative_log_likelihood(params, y_true, distribution: str):
    """Log score loss."""
    if distribution == "normal":
        mu, sigma = params[:, 0], params[:, 1]
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(y_true).mean()

    elif distribution == "student_t":
        mu, sigma, nu = params[:, 0], params[:, 1], params[:, 2]
        dist = torch.distributions.StudentT(nu, mu, sigma)
        return -dist.log_prob(y_true).mean()

    # NIG requires custom implementation
    else:
        raise NotImplementedError(f"Log score not implemented for {distribution}")
```

### Files to Create/Modify

| File | Action |
|------|--------|
| `src/temporalpdf/ml.py` | Create - core module |
| `src/temporalpdf/__init__.py` | Add: `from .ml import DistributionalRegressor` |
| `experiments/run_experiments_v2.py` | Test new approach vs old |

---

## PRIORITY 2: Analytical Barrier Approximations

### Goal
100x speedup for barrier probability calculation.

### Implementation

**Add to `src/temporalpdf/utilities.py`:**

```python
from scipy.stats import norm
import numpy as np

def barrier_prob_normal_analytical(mu: float, sigma: float, horizon: int, barrier: float) -> float:
    """
    Analytical barrier probability for random walk with Normal increments.

    Uses reflection principle - exact for Brownian motion.

    Args:
        mu: Mean of daily returns
        sigma: Std dev of daily returns
        horizon: Number of days
        barrier: Cumulative return threshold (e.g., 0.05 for 5%)

    Returns:
        P(max cumulative sum >= barrier)
    """
    drift = mu * horizon
    vol = sigma * np.sqrt(horizon)

    if vol < 1e-10:
        return 1.0 if drift >= barrier else 0.0

    d1 = (barrier - drift) / vol
    d2 = (barrier + drift) / vol

    p = norm.cdf(-d1)

    # Correction for drift
    if mu > 1e-10:
        p += np.exp(2 * mu * barrier / (sigma**2)) * norm.cdf(-d2)

    return np.clip(p, 0, 1)


def barrier_prob_fat_tail_approx(
    mu: float,
    sigma: float,
    nu: float,
    horizon: int,
    barrier: float
) -> float:
    """
    Approximate barrier probability for fat-tailed distributions.

    Uses Normal analytical formula with inflated volatility to account
    for fat tails (Student-t, NIG).

    Args:
        mu: Location parameter
        sigma: Scale parameter
        nu: Degrees of freedom (tail index). Lower = fatter tails.
        horizon: Number of days
        barrier: Cumulative return threshold

    Returns:
        Approximate P(max cumulative sum >= barrier)
    """
    # Inflate sigma to account for fat tails
    if nu > 2:
        # Student-t variance = sigma^2 * nu/(nu-2)
        tail_factor = np.sqrt(nu / (nu - 2))
    else:
        # Infinite variance case - use conservative estimate
        tail_factor = 3.0

    effective_sigma = sigma * tail_factor
    return barrier_prob_normal_analytical(mu, effective_sigma, horizon, barrier)


def barrier_prob_nig_approx(
    mu: float,
    delta: float,
    alpha: float,
    beta: float,
    horizon: int,
    barrier: float
) -> float:
    """
    Approximate barrier probability for NIG distribution.

    NIG has semi-heavy tails, so we use adjusted Normal approximation.
    """
    # NIG variance = delta / gamma^3 where gamma = sqrt(alpha^2 - beta^2)
    gamma = np.sqrt(alpha**2 - beta**2)
    nig_variance = delta / (gamma**3)
    effective_sigma = np.sqrt(nig_variance)

    # NIG kurtosis indicates tail heaviness
    # Use kurtosis to estimate effective degrees of freedom
    nig_kurtosis = 3 * (1 + 4 * beta**2 / gamma**2) / (delta * gamma)
    effective_nu = max(4, 6 / nig_kurtosis + 4)  # Map kurtosis to approx nu

    return barrier_prob_fat_tail_approx(mu, effective_sigma, effective_nu, horizon, barrier)
```

### Usage

```python
from temporalpdf.utilities import barrier_prob_fat_tail_approx

# Fast approximation (microseconds vs milliseconds)
p_fast = barrier_prob_fat_tail_approx(mu=0.001, sigma=0.02, nu=5, horizon=10, barrier=0.05)

# Compare with MC simulation for accuracy check
p_mc = simulate_barrier_prob(params, horizon=10, barrier=0.05, n_sims=10000)
```

---

## PRIORITY 3: Importance Sampling for Rare Events

### Goal
10-100x variance reduction for low-probability events (5% barrier).

### Implementation

**Add to `src/temporalpdf/utilities.py`:**

```python
def importance_sampling_barrier(
    params,
    distribution: str,
    horizon: int,
    barrier: float,
    n_sims: int = 1000
) -> float:
    """
    Barrier probability with importance sampling for rare events.

    Tilts the distribution toward barrier-hitting paths, then corrects
    with likelihood ratios. Much lower variance for rare events.

    Args:
        params: Distribution parameters (StudentTParameters, etc.)
        distribution: "student_t", "normal", or "nig"
        horizon: Number of time steps
        barrier: Cumulative threshold
        n_sims: Number of simulation paths

    Returns:
        Importance-weighted barrier probability estimate
    """
    # Extract parameters
    if distribution == "student_t":
        mu = params.mu_0
        sigma = params.sigma_0
        nu = params.nu
    elif distribution == "normal":
        mu = params.mu
        sigma = params.sigma
        nu = None
    elif distribution == "nig":
        mu = params.mu
        sigma = np.sqrt(params.delta / np.sqrt(params.alpha**2 - params.beta**2)**3)
        nu = 4.0  # Approximate

    # Optimal drift shift toward barrier (Girsanov-style)
    optimal_shift = barrier / (horizon * sigma)
    tilted_mu = mu + optimal_shift * sigma

    # Sample from tilted distribution
    if distribution == "student_t":
        from scipy.stats import t
        samples = tilted_mu + sigma * t.rvs(nu, size=(n_sims, horizon))
    elif distribution == "normal":
        samples = tilted_mu + sigma * np.random.randn(n_sims, horizon)
    elif distribution == "nig":
        # Use shifted NIG
        import temporalpdf as tpdf
        shifted_params = tpdf.NIGParameters(
            mu=tilted_mu, delta=params.delta, alpha=params.alpha, beta=params.beta
        )
        samples = tpdf.NIG().sample(n_sims * horizon, t=0, params=shifted_params)
        samples = samples.reshape(n_sims, horizon)

    # Compute paths
    paths = np.cumsum(samples, axis=1)
    max_paths = np.max(paths, axis=1)
    hits = max_paths >= barrier

    # Likelihood ratio correction
    # log(p_original / p_tilted) = -shift * sum(X) + 0.5 * shift^2 * sigma^2 * T
    log_ratio = (-optimal_shift * np.sum(samples - tilted_mu + mu, axis=1) +
                 0.5 * optimal_shift**2 * sigma**2 * horizon)
    weights = np.exp(log_ratio)

    # Normalize weights (self-normalized importance sampling)
    weights = weights / weights.sum()

    # Importance-weighted estimator
    return np.sum(hits * weights)
```

---

## PRIORITY 4: Quasi-Monte Carlo

### Goal
2-10x variance reduction with minimal code change.

### Implementation

**Add to `src/temporalpdf/utilities.py`:**

```python
from scipy.stats import qmc

def quasi_monte_carlo_barrier(
    params,
    distribution: str,
    horizon: int,
    barrier: float,
    n_sims: int = 1000
) -> float:
    """
    Barrier probability using Quasi-Monte Carlo (Sobol sequences).

    QMC fills the sample space more uniformly than pseudo-random,
    giving lower variance with same number of samples.
    """
    # Generate Sobol sequence
    sampler = qmc.Sobol(d=horizon, scramble=True)
    uniform_samples = sampler.random(n=n_sims)  # (n_sims, horizon) in [0,1]

    # Transform to standard normal
    from scipy.stats import norm, t

    if distribution == "normal":
        mu, sigma = params.mu, params.sigma
        z = norm.ppf(uniform_samples)
        samples = mu + sigma * z

    elif distribution == "student_t":
        mu, sigma, nu = params.mu_0, params.sigma_0, params.nu
        z = t.ppf(uniform_samples, df=nu)
        samples = mu + sigma * z

    elif distribution == "nig":
        # NIG inverse CDF is complex - fall back to normal approx for QMC
        gamma = np.sqrt(params.alpha**2 - params.beta**2)
        sigma = np.sqrt(params.delta / gamma**3)
        z = norm.ppf(uniform_samples)
        samples = params.mu + sigma * z

    # Compute paths
    paths = np.cumsum(samples, axis=1)
    max_paths = np.max(paths, axis=1)

    return np.mean(max_paths >= barrier)
```

---

## PRIORITY 5: Calibration Features

### Goal
Add features that directly predict distribution parameters.

### Implementation

**Add to `experiments/prepare_data.py` or create `src/temporalpdf/features.py`:**

```python
import numpy as np
from scipy import stats

def hill_estimator(data: np.ndarray, k: int = 5) -> float:
    """
    Hill estimator for tail index.

    Estimates the tail exponent (related to degrees of freedom).
    Higher = thinner tails, Lower = fatter tails.
    """
    sorted_data = np.sort(np.abs(data))[::-1]
    if len(sorted_data) < k + 1:
        return 4.0  # Default
    log_ratios = np.log(sorted_data[:k] / sorted_data[k])
    if np.sum(log_ratios) == 0:
        return 10.0
    return k / np.sum(log_ratios)


def calibration_features(window: np.ndarray) -> dict:
    """
    Features that directly inform distribution parameters.

    These are specifically designed to predict μ, σ, ν.
    """
    return {
        # Predicts ν (tail index)
        "hill_estimator": hill_estimator(window, k=5),

        # Tests normality - high = fat tails
        "jarque_bera": stats.jarque_bera(window)[0],

        # Realized kurtosis (predicts ν)
        "kurtosis": stats.kurtosis(window),

        # Realized skewness (predicts β for NIG)
        "skewness": stats.skew(window),

        # Volatility clustering (predicts time-varying σ)
        "vol_clustering": np.corrcoef(np.abs(window[:-1]), np.abs(window[1:]))[0, 1],

        # GARCH-style vol forecast
        "garch_proxy": np.mean(window[-5:]**2) / np.mean(window**2),

        # Regime indicator (recent vs overall vol)
        "vol_regime": np.std(window[-5:]) / (np.std(window) + 1e-8),

        # Extreme event frequency
        "extreme_freq": np.mean(np.abs(window) > 2 * np.std(window)),

        # Tail ratio (upper vs lower)
        "tail_asymmetry": (np.sum(window > 2*np.std(window)) /
                          (np.sum(window < -2*np.std(window)) + 1)),
    }
```

---

## PRIORITY 6: Conformal Prediction

### Goal
Distribution-free coverage guarantees on predictions.

### Implementation

```python
from mapie.regression import MapieRegressor

def add_conformal_uncertainty(base_model, X_train, y_train, X_cal, y_cal, alpha=0.1):
    """
    Wrap any model with conformal prediction for calibrated intervals.

    Args:
        base_model: Trained sklearn-compatible model
        X_train, y_train: Training data (already used to fit base_model)
        X_cal, y_cal: Calibration data (held out)
        alpha: Miscoverage rate (0.1 = 90% intervals)

    Returns:
        MAPIE model that gives prediction intervals
    """
    # Wrap with conformal prediction
    mapie = MapieRegressor(base_model, method="plus", cv="prefit")
    mapie.fit(X_cal, y_cal)

    return mapie


def predict_with_intervals(mapie_model, X_test, alpha=0.1):
    """
    Get predictions with calibrated uncertainty intervals.
    """
    y_pred, y_pis = mapie_model.predict(X_test, alpha=alpha)
    # y_pred: point predictions
    # y_pis: array of shape (n_samples, 2, 1) with [lower, upper] bounds
    return y_pred, y_pis[:, 0, 0], y_pis[:, 1, 0]  # pred, lower, upper
```

---

## PRIORITY 7: Use Temporal Dynamics

### Goal
Use temporalpdf's time-varying parameter projection.

### Implementation

```python
import temporalpdf as tpdf

def simulate_with_temporal_dynamics(
    dist_type: str,
    historical_window: np.ndarray,
    horizon: int,
    barrier: float,
    n_sims: int = 1000
) -> float:
    """
    Simulate barrier probability with time-varying parameters.

    Instead of fixed σ over all days, σ evolves via GARCH dynamics.
    """
    # Set up temporal model
    temporal = tpdf.TemporalModel(
        distribution=dist_type,
        tracking=tpdf.ParameterTracker(dist_type, window=60),
        dynamics={"sigma_0": tpdf.GARCH(1, 1)},  # or sigma for normal
    )

    # Fit to historical data
    temporal.fit(historical_window)

    # Project parameters forward
    projection = temporal.project(horizon=horizon, n_paths=n_sims)

    # Simulate with time-varying parameters
    hits = 0
    for path_idx in range(n_sims):
        cumsum = 0
        max_cumsum = 0

        for t in range(horizon):
            # Get parameters at time t for this path
            sigma_t = projection.param_paths["sigma_0"][path_idx, t]
            mu_t = projection.param_paths.get("mu_0", [0]*horizon)[path_idx, t] \
                   if "mu_0" in projection.param_paths else 0

            # Sample single return
            if dist_type == "student_t":
                nu = projection.param_paths.get("nu", [5]*horizon)[path_idx, t]
                day_return = mu_t + sigma_t * np.random.standard_t(nu)
            else:
                day_return = mu_t + sigma_t * np.random.randn()

            cumsum += day_return
            max_cumsum = max(max_cumsum, cumsum)

        if max_cumsum >= barrier:
            hits += 1

    return hits / n_sims
```

---

## PRIORITY 8: End-to-End Differentiable Training

### Goal
Train directly on barrier Brier score, not parameter MSE.

### Implementation

```python
import torch
import torch.nn as nn

def soft_barrier_prob(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
    barrier: float,
    horizon: int,
    n_sims: int = 500,
    temperature: float = 10.0
) -> torch.Tensor:
    """
    Differentiable barrier probability via soft threshold.

    Uses reparameterization trick for Student-t to enable gradients.
    Soft sigmoid threshold makes the barrier crossing differentiable.
    """
    batch_size = mu.shape[0]

    # Reparameterized Student-t sampling
    z = torch.randn(n_sims, batch_size, horizon, device=mu.device)
    chi2 = torch.distributions.Chi2(nu.unsqueeze(0).unsqueeze(-1).expand(n_sims, batch_size, horizon))
    v = chi2.rsample()

    # Student-t samples: mu + sigma * z * sqrt(nu/v)
    samples = mu.unsqueeze(0).unsqueeze(-1) + \
              sigma.unsqueeze(0).unsqueeze(-1) * z * torch.sqrt(nu.unsqueeze(0).unsqueeze(-1) / v)

    # Cumulative sum along horizon
    cumsum = torch.cumsum(samples, dim=-1)  # (n_sims, batch, horizon)
    max_cumsum = torch.max(cumsum, dim=-1).values  # (n_sims, batch)

    # Soft threshold for differentiability
    # sigmoid(temp * (x - barrier)) ≈ 1 if x > barrier, 0 otherwise
    soft_hits = torch.sigmoid(temperature * (max_cumsum - barrier))

    # Average over simulations
    return soft_hits.mean(dim=0)  # (batch,)


class EndToEndBarrierModel(nn.Module):
    """
    Neural network trained end-to-end on barrier probability.

    Loss = Brier(predicted_barrier_prob, actual_barrier_hit)
    """

    def __init__(self, n_features: int, hidden_dims: list = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = n_features
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        # Output: μ, log(σ), log(ν-2)
        self.output = nn.Linear(hidden_dims[-1], 3)

    def forward(self, x, barrier: float, horizon: int, n_sims: int = 200):
        """
        Forward pass returns barrier probability.
        """
        shared = self.shared(x)
        raw = self.output(shared)

        # Apply constraints
        mu = raw[:, 0]
        sigma = torch.nn.functional.softplus(raw[:, 1]) + 1e-6
        nu = torch.nn.functional.softplus(raw[:, 2]) + 2.0

        # Differentiable barrier probability
        return soft_barrier_prob(mu, sigma, nu, barrier, horizon, n_sims)


def train_end_to_end(model, X, y_barrier, barrier, horizon, n_epochs=100, lr=1e-3):
    """
    Train model directly on barrier prediction accuracy.

    Args:
        model: EndToEndBarrierModel
        X: Features (n_samples, n_features)
        y_barrier: Binary labels - did barrier get hit? (n_samples,)
        barrier: Barrier level
        horizon: Forecast horizon
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_barrier, dtype=torch.float32)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        p_barrier = model(X, barrier, horizon, n_sims=200)

        # Brier score loss
        loss = ((p_barrier - y) ** 2).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Brier: {loss.item():.4f}")

    return model
```

---

## Summary Table

| Priority | Optimization | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| 1 | DistributionalRegressor (CRPS) | Medium-High | High | To implement |
| 2 | Analytical barrier approximation | Medium | High | To implement |
| 3 | Importance sampling | Medium | High | To implement |
| 4 | Quasi-Monte Carlo | Low | Medium | To implement |
| 5 | Calibration features | Low | Medium | To implement |
| 6 | Conformal prediction | Low | Medium | To implement |
| 7 | Temporal dynamics | Low | Medium-High | To implement |
| 8 | End-to-end differentiable | High | Highest | Research |

---

## What CRPS and MLP Mean

### MLP (Multi-Layer Perceptron)
A neural network. Layers of neurons with nonlinear activations.
```
Input features → [Linear → ReLU → Linear → ReLU → Linear] → Output parameters
     (32)              (64)         (32)         (3: μ, σ, ν)
```
Why: Can learn complex mappings. Trains with gradient descent, enabling any differentiable loss.

### CRPS (Continuous Ranked Probability Score)
Measures how good a predicted distribution is at capturing the true outcome.

**Formula:**
```
CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
```
- First term: expected distance from samples to truth
- Second term: expected spread of distribution (rewards sharpness)

**Why CRPS over MSE on parameters?**
- MSE treats all parameter errors equally
- CRPS directly measures probabilistic accuracy
- Aligns training with evaluation

### "CRPS via sampling"
Instead of closed-form CRPS (different formula per distribution), draw samples and compute numerically:
```python
samples = distribution.sample(1000)
term1 = |samples - y_true|.mean()
term2 = |samples[:500] - samples[500:]|.mean()
crps = term1 - 0.5 * term2
```
Distribution-agnostic - works with any distribution that can sample.
