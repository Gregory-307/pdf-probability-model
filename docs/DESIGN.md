# temporalpdf Design Document

**Version**: 2.0 (Library Pivot)
**Author**: Greg Butcher
**Status**: Draft
**Last Updated**: January 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Literature Review](#2-literature-review)
3. [Architecture](#3-architecture)
4. [Distribution Choices](#4-distribution-choices)
5. [Decision Utilities](#5-decision-utilities)
6. [ML Integration](#6-ml-integration)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [API Design](#8-api-design)

---

## 1. Problem Statement

### 1.1 The Fundamental Limitation of Point Predictions

Traditional machine learning models for regression produce **point estimates**—single predicted values that minimize some loss function (typically MSE or MAE). In financial contexts, this manifests as:

```
Model: E[r_{t+h} | X_t] = f(X_t; θ)
```

Where `r_{t+h}` is the future return, `X_t` are features at time `t`, and `θ` are learned parameters.

**The critical problem**: Point predictions discard all information about uncertainty. A model predicting +2% return with high confidence is fundamentally different from one predicting +2% with massive uncertainty, yet both produce identical outputs.

### 1.2 Why This Matters in Finance

Consider two scenarios:

| Metric | Scenario A | Scenario B |
|--------|------------|------------|
| Predicted Return | +2% | +2% |
| True Distribution | N(2%, 1%) | N(2%, 20%) |
| P(loss > 5%) | 0.0001% | 36.3% |
| Optimal Kelly Fraction | 200% | 0.5% |

Same point prediction. Radically different investment decisions.

### 1.3 What is Distributional Regression?

**Distributional regression** (also called "probabilistic prediction" or "distribution forecasting") models the entire conditional distribution of the target variable:

```
Model: P(r_{t+h} | X_t) = D(μ(X_t), σ(X_t), ν(X_t), ...)
```

Instead of predicting a single number, we predict the **parameters of a probability distribution**. This provides:

1. **Central tendency** (mean/median)
2. **Uncertainty quantification** (variance/scale)
3. **Tail behavior** (degrees of freedom, tail indices)
4. **Asymmetry** (skewness parameters)

### 1.4 The temporalpdf Solution

**temporalpdf** is a Python library for:

1. **Defining** flexible probability distributions with time-evolving parameters
2. **Training** ML models to predict distribution parameters from features
3. **Evaluating** distributional forecasts with proper scoring rules
4. **Deciding** optimal actions using risk measures (VaR, CVaR) and utility functions (Kelly)

The library bridges the gap between:
- Statistical distribution theory
- Modern gradient boosting / ML pipelines
- Quantitative finance decision-making

---

## 2. Literature Review

### 2.1 Probabilistic Prediction with Gradient Boosting

**NGBoost: Natural Gradient Boosting for Probabilistic Prediction**
Duan, T., Avati, A., Ding, D.Y., Thai, K.K., Basu, S., Ng, A., & Schuler, A. (2020).
*Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119:2690-2700.
[Paper](https://proceedings.mlr.press/v119/duan20a.html) | [arXiv](https://arxiv.org/abs/1910.03225)

NGBoost introduces a principled approach to probabilistic regression via gradient boosting:

- Uses **natural gradients** to handle multi-parameter distributions
- Modular: works with any base learner, distribution family, and scoring rule
- The key insight: ordinary gradients are problematic for learning distribution parameters because the parameter space has non-Euclidean geometry

The natural gradient correction:

$$\tilde{\nabla}_\theta \mathcal{L} = I(\theta)^{-1} \nabla_\theta \mathcal{L}$$

Where $I(\theta)$ is the Fisher information matrix.

### 2.2 Quantile Regression

**Regression Quantiles**
Koenker, R. & Bassett, G. (1978).
*Econometrica*, 46(1), 33-50.
[Paper](https://www.econometricsociety.org/publications/econometrica/1978/01/01/regression-quantiles)

Quantile regression estimates conditional quantiles rather than the conditional mean:

$$\hat{Q}_\tau(Y|X) = \arg\min_{\beta} \sum_{i} \rho_\tau(y_i - x_i'\beta)$$

Where the **pinball loss** is:

$$\rho_\tau(u) = u(\tau - \mathbf{1}_{u<0})$$

Quantile regression is **distribution-free** but requires fitting separate models for each quantile. temporalpdf takes the parametric approach instead, which is more efficient when the distributional form is approximately correct.

### 2.3 Volatility Modeling (GARCH)

**Generalized Autoregressive Conditional Heteroskedasticity**
Bollerslev, T. (1986).
*Journal of Econometrics*, 31(3), 307-327.
[Paper](https://www.sciencedirect.com/science/article/abs/pii/0304407686900631)

The GARCH(1,1) model captures **volatility clustering**:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Key insight: volatility is **persistent** and **predictable**. temporalpdf's time-evolving parameters (the `beta` parameter for volatility growth) are inspired by this observation.

### 2.4 Financial Distribution Choices

#### 2.4.1 Normal Inverse Gaussian (NIG)

**Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling**
Barndorff-Nielsen, O.E. (1997).
*Scandinavian Journal of Statistics*, 24(1), 1-13.
[Paper](https://onlinelibrary.wiley.com/doi/10.1111/1467-9469.00045)

**Processes of Normal Inverse Gaussian Type**
Barndorff-Nielsen, O.E. (1998).
*Finance and Stochastics*, 2(1), 41-68.
[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=57888)

The NIG distribution is defined as a **normal variance-mean mixture** with inverse Gaussian mixing:

$$X = \mu + \beta V + \sqrt{V} Z$$

Where $V \sim IG(\delta, \sqrt{\alpha^2 - \beta^2})$ and $Z \sim N(0,1)$.

The PDF is:

$$f(x; \alpha, \beta, \mu, \delta) = \frac{\alpha \delta}{\pi} \exp(\delta\gamma + \beta(x-\mu)) \frac{K_1(\alpha\sqrt{\delta^2 + (x-\mu)^2})}{\sqrt{\delta^2 + (x-\mu)^2}}$$

Where $\gamma = \sqrt{\alpha^2 - \beta^2}$ and $K_1$ is the modified Bessel function of the second kind.

#### 2.4.2 Generalized Hyperbolic Family

**A Comparison of Generalized Hyperbolic Distribution Models for Equity Returns**
Konlack Socgnia, V. & Wilcox, D. (2014).
*Journal of Applied Mathematics*, 2014, Article ID 263465.
[Paper](https://onlinelibrary.wiley.com/doi/10.1155/2014/263465)

The **Generalized Hyperbolic (GH)** distribution is the parent family containing:

| Distribution | GH Parameter $\lambda$ | Properties |
|--------------|------------------------|------------|
| **NIG** | $\lambda = -1/2$ | Semi-heavy tails, closed under convolution |
| **Variance Gamma** | $\lambda > 0$, $\chi = 0$ | Finite moments, popular for options |
| **Hyperbolic** | $\lambda = 1$ | Exponential tails |
| **Student's t** | $\lambda = -\nu/2$, $\gamma = 0$ | Heavy tails, no skew |

#### 2.4.3 Variance Gamma vs NIG

**Variance-Gamma and Normal-Inverse Gaussian Models: Goodness-of-fit to Chinese High-Frequency Index Returns**
Hu, W. (2016).
*North American Journal of Economics and Finance*, 36, 279-292.
[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1062940816300079)

Key finding: **NIG outperforms VG at higher frequencies**. As the time scale decreases, the goodness-of-fit advantage of NIG increases.

### 2.5 Proper Scoring Rules

**Strictly Proper Scoring Rules, Prediction, and Estimation**
Gneiting, T. & Raftery, A.E. (2007).
*Journal of the American Statistical Association*, 102(477), 359-378.
[Paper](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) | [PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

A scoring rule $S(P, y)$ is **proper** if the expected score is optimized when the forecaster reports their true belief:

$$\mathbb{E}_{Y \sim P}[S(P, Y)] \leq \mathbb{E}_{Y \sim P}[S(Q, Y)] \quad \forall Q \neq P$$

Key proper scoring rules:

**Log Score (Negative Log Likelihood)**:
$$S_{log}(P, y) = -\log p(y)$$

**Continuous Ranked Probability Score (CRPS)**:
$$CRPS(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbf{1}_{x \geq y})^2 dx$$

CRPS generalizes MAE to probabilistic forecasts and has the closed form for location-scale families:

$$CRPS(F, y) = \mathbb{E}|X - y| - \frac{1}{2}\mathbb{E}|X - X'|$$

Where $X, X' \sim F$ are independent.

### 2.6 Kelly Criterion and Position Sizing

**A New Interpretation of Information Rate**
Kelly, J.L. (1956).
*Bell System Technical Journal*, 35(4), 917-926.

**The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market**
Thorp, E.O. (2006).
*Handbook of Asset and Liability Management*, 1, 385-428.
[Paper](https://web.williams.edu/Mathematics/sjmiller/public_html/341/handouts/Thorpe_KellyCriterion2007.pdf)

The Kelly criterion maximizes the **expected logarithmic growth rate**:

$$f^* = \arg\max_f \mathbb{E}[\log(1 + f \cdot r)]$$

For a distribution with mean $\mu$ and variance $\sigma^2$, the continuous Kelly fraction approximates to:

$$f^* \approx \frac{\mu}{\sigma^2}$$

Or equivalently using the Sharpe ratio:

$$f^* = \frac{SR}{\sigma} = \frac{\mu - r_f}{\sigma^2}$$

**Fractional Kelly**: In practice, using $\kappa \cdot f^*$ where $\kappa \in (0, 1)$ reduces variance dramatically:
- 50% Kelly → 75% of optimal growth with 25% of variance
- 25% Kelly → 44% of optimal growth with 6% of variance

### 2.7 Risk Measures

**Optimization of Conditional Value-at-Risk**
Rockafellar, R.T. & Uryasev, S. (2000).
*Journal of Risk*, 2, 21-41.

**Conditional Value-at-Risk for General Loss Distributions**
Rockafellar, R.T. & Uryasev, S. (2002).
*Journal of Banking & Finance*, 26(7), 1443-1471.
[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378426602002716)

CVaR (also called Expected Shortfall) is a **coherent risk measure** satisfying:
1. Monotonicity
2. Sub-additivity (diversification reduces risk)
3. Positive homogeneity
4. Translation invariance

Unlike VaR, CVaR accounts for tail severity, not just tail probability.

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           temporalpdf                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Distributions │    │   Scoring    │    │   Decision   │              │
│  │              │    │    Rules     │    │   Utilities  │              │
│  │  - NIG       │    │              │    │              │              │
│  │  - Normal    │    │  - LogScore  │    │  - VaR       │              │
│  │  - Student-t │    │  - CRPS      │    │  - CVaR      │              │
│  │  - VG        │    │  - Energy    │    │  - Kelly     │              │
│  │  - Custom    │    │              │    │  - Quantiles │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │  Core Engine    │                                  │
│                    │                 │                                  │
│                    │ - PDF/CDF/PPF   │                                  │
│                    │ - Sampling      │                                  │
│                    │ - Moments       │                                  │
│                    │ - Time Evolution│                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│         ┌───────────────────┼───────────────────┐                       │
│         │                   │                   │                       │
│  ┌──────▼───────┐    ┌──────▼───────┐    ┌──────▼───────┐              │
│  │  ML Models   │    │  Evaluation  │    │Visualization │              │
│  │              │    │              │    │              │              │
│  │ - NGBoost    │    │ - Backtest   │    │ - 3D Surface │              │
│  │ - Sklearn    │    │ - Calibration│    │ - Heatmaps   │              │
│  │ - Custom     │    │ - PIT Histog │    │ - Bands      │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### 3.2.1 Core Layer

| Component | Responsibility |
|-----------|----------------|
| `Distribution` | Abstract base defining PDF, CDF, PPF, sampling, moments |
| `TimeEvolvingDistribution` | Extends Distribution with parameter dynamics |
| `DistributionParameters` | Immutable parameter containers |
| `EvaluationGrid` | Defines (value, time) grids for computation |
| `PDFResult` | Computed PDF matrices with metadata |

#### 3.2.2 Distribution Layer

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| `Normal` | μ, σ | Baseline, efficient computation |
| `StudentT` | μ, σ, ν | Heavy tails, outlier robustness |
| `NIG` | μ, δ, α, β | Semi-heavy tails, skew, financial returns |
| `VarianceGamma` | μ, σ, θ, ν | Options pricing, finite moments |
| `SkewNormal` | μ, σ, α | Light tails with asymmetry |

#### 3.2.3 Scoring Layer

| Score | Formula | Properties |
|-------|---------|------------|
| `LogScore` | $-\log p(y)$ | Strictly proper, sensitive to density |
| `CRPS` | $\int (F(x) - \mathbf{1}_{x \geq y})^2 dx$ | Strictly proper, robust, generalizes MAE |
| `EnergyScore` | Multivariate CRPS generalization | For vector-valued predictions |

#### 3.2.4 Decision Layer

| Utility | Purpose | Formula |
|---------|---------|---------|
| `VaR` | Threshold loss | $VaR_\alpha = F^{-1}(\alpha)$ |
| `CVaR` | Expected tail loss | $CVaR_\alpha = \mathbb{E}[X | X \leq VaR_\alpha]$ |
| `Kelly` | Optimal fraction | $f^* = \mu/\sigma^2$ |
| `ProbQuery` | Probability of events | $P(X > k)$, $P(a < X < b)$ |

### 3.3 Data Flow

```
                    Training Pipeline
                    ─────────────────

Features (X)  ──────┐
                    │
                    ▼
              ┌───────────┐
              │ ML Model  │  (e.g., NGBoost, GradientBoosting)
              │           │
              │ Predicts: │
              │  θ = f(X) │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────────────────────────────────────┐
              │  Distribution Parameters (per sample)     │
              │                                           │
              │  θ = (μ_hat, σ_hat, ν_hat, β_hat, ...)   │
              └─────┬─────────────────────────────────────┘
                    │
                    ▼
              ┌───────────────────────────────────────────┐
              │  Full Predictive Distribution             │
              │                                           │
              │  P(y | X) = D(y; θ)                       │
              │                                           │
              │  - PDF: probability density               │
              │  - CDF: cumulative probability            │
              │  - Quantiles: VaR at any level            │
              │  - Moments: mean, variance, skewness      │
              └─────┬─────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    ┌───────────┐       ┌───────────┐
    │  Scoring  │       │ Decisions │
    │           │       │           │
    │ - LogScore│       │ - VaR     │
    │ - CRPS    │       │ - CVaR    │
    │           │       │ - Kelly   │
    └───────────┘       └───────────┘
```

### 3.4 Interface Contracts

#### 3.4.1 Distribution Protocol

```python
from typing import Protocol, TypeVar
import numpy as np
from numpy.typing import NDArray

P = TypeVar("P", bound="DistributionParameters")

class Distribution(Protocol[P]):
    """Core distribution interface."""

    @property
    def name(self) -> str: ...

    @property
    def n_parameters(self) -> int: ...

    def pdf(self, x: NDArray, params: P) -> NDArray:
        """Probability density function."""
        ...

    def cdf(self, x: NDArray, params: P) -> NDArray:
        """Cumulative distribution function."""
        ...

    def ppf(self, q: NDArray, params: P) -> NDArray:
        """Percent point function (quantile function / inverse CDF)."""
        ...

    def sample(self, n: int, params: P, rng: np.random.Generator) -> NDArray:
        """Draw n samples from the distribution."""
        ...

    def mean(self, params: P) -> float:
        """Expected value."""
        ...

    def variance(self, params: P) -> float:
        """Variance."""
        ...

    def log_likelihood(self, x: NDArray, params: P) -> float:
        """Sum of log PDF values."""
        ...
```

#### 3.4.2 Scoring Rule Protocol

```python
class ScoringRule(Protocol):
    """Proper scoring rule interface."""

    @property
    def name(self) -> str: ...

    @property
    def is_proper(self) -> bool: ...

    def score(
        self,
        dist: Distribution,
        params: DistributionParameters,
        y: NDArray,
    ) -> float:
        """Compute score for observations y under distribution."""
        ...

    def gradient(
        self,
        dist: Distribution,
        params: DistributionParameters,
        y: NDArray,
    ) -> NDArray:
        """Gradient of score w.r.t. distribution parameters."""
        ...
```

#### 3.4.3 Decision Utility Protocol

```python
class RiskMeasure(Protocol):
    """Risk measure interface."""

    def __call__(
        self,
        dist: Distribution,
        params: DistributionParameters,
        alpha: float,
    ) -> float:
        """Compute risk measure at confidence level alpha."""
        ...
```

---

## 4. Distribution Choices

### 4.1 Why NIG Over Others?

The **Normal Inverse Gaussian** distribution is the default choice for financial returns because:

#### 4.1.1 Empirical Fit

Financial returns exhibit:
- **Semi-heavy tails**: Heavier than Normal, lighter than Pareto
- **Negative skewness**: Large drops more common than large gains
- **Excess kurtosis**: More probability in tails and center than Normal

NIG captures all three with interpretable parameters.

#### 4.1.2 Mathematical Properties

| Property | NIG | Variance Gamma | Student's t | Normal |
|----------|-----|----------------|-------------|--------|
| Closed under convolution | **Yes** | No | No | Yes |
| Closed under linear transform | **Yes** | No | No | Yes |
| Finite moments (all orders) | **Yes** | Yes | Only < ν | Yes |
| Can model skewness | **Yes** | Yes | No | No |
| Semi-heavy tails | **Yes** | Yes | Heavy | Light |
| Levy process compatible | **Yes** | Yes | No | Yes |

**Closure under convolution** is critical: if $X_1 \sim NIG$ and $X_2 \sim NIG$ (independent, same shape parameters), then $X_1 + X_2 \sim NIG$. This makes aggregation (e.g., daily → weekly returns) analytically tractable.

#### 4.1.3 Parameter Interpretation

| Parameter | Symbol | Interpretation | Financial Meaning |
|-----------|--------|----------------|-------------------|
| Location | μ | Shifts distribution | Expected return component |
| Scale | δ | Controls spread | Base volatility |
| Steepness | α | Tail heaviness | Fat-tail intensity |
| Asymmetry | β | Skewness direction | Downside vs upside risk |

The **reparameterization** often used in finance:

$$\bar{\alpha} = \alpha\delta, \quad \bar{\beta} = \beta\delta$$

$$\mu_{NIG} = \mu + \frac{\delta\beta}{\sqrt{\alpha^2 - \beta^2}}$$

$$\sigma_{NIG}^2 = \frac{\delta\alpha^2}{(\alpha^2 - \beta^2)^{3/2}}$$

### 4.2 When to Use Each Distribution

```
                    Decision Tree for Distribution Selection
                    ────────────────────────────────────────

                              ┌─────────────────┐
                              │  What type of   │
                              │     data?       │
                              └────────┬────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
               ▼                       ▼                       ▼
        ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
        │  Financial  │         │   Sensor/   │         │   General   │
        │   Returns   │         │  Physical   │         │   ML Task   │
        └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
               │                       │                       │
               ▼                       ▼                       ▼
        ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
        │  Skewness?  │         │  Outliers?  │         │  Baseline:  │
        └──────┬──────┘         └──────┬──────┘         │   Normal    │
               │                       │                └─────────────┘
       ┌───────┴───────┐       ┌───────┴───────┐
       │               │       │               │
       ▼               ▼       ▼               ▼
┌────────────┐  ┌────────────┐ ┌────────────┐  ┌────────────┐
│ Yes: NIG   │  │ No: use    │ │ Yes:       │  │ No:        │
│            │  │ Student-t  │ │ Student-t  │  │ Normal     │
└────────────┘  └────────────┘ └────────────┘  └────────────┘
```

**Detailed Recommendations**:

| Scenario | Distribution | Why |
|----------|--------------|-----|
| Stock/ETF returns | NIG | Skew + semi-heavy tails |
| Volatility forecasting | Log-Normal / Gamma | Positive support |
| Credit spreads | NIG or VG | Heavy tails, skew |
| Option-implied | Variance Gamma | Industry standard, finite moments |
| Measurement errors | Student's t | Outlier robustness |
| Baseline / fast | Normal | Closed form everything |
| Highly asymmetric | Skew-Normal or NIG | Controls skew direction |

### 4.3 Parameter Estimation

For NIG, maximum likelihood estimation requires numerical optimization:

$$\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{n} \log f_{NIG}(x_i; \theta)$$

**Method of Moments** provides starting values:

$$\hat{\mu} = \bar{x} - \frac{\hat{\delta}\hat{\beta}}{\hat{\gamma}}$$

$$\hat{\delta} = \frac{3}{\hat{\kappa} - 3} \cdot s^2 / \sqrt{3/(\hat{\kappa}-3) + 1}$$

Where $\bar{x}$ is sample mean, $s^2$ is sample variance, and $\hat{\kappa}$ is sample kurtosis.

---

## 5. Decision Utilities

### 5.1 Value at Risk (VaR)

**Definition**: The $\alpha$-quantile of the loss distribution.

$$VaR_\alpha(X) = F_X^{-1}(\alpha) = \inf\{x : F_X(x) \geq \alpha\}$$

For returns (not losses), we typically want the left tail:

$$VaR_\alpha = -F_X^{-1}(\alpha)$$

**Example**: 95% VaR of -3% means there's a 5% chance of losing more than 3%.

**Computation**: For distributions with closed-form quantile functions:

```python
var_95 = -dist.ppf(0.05, params)
```

For others, use numerical inversion or sampling:

```python
samples = dist.sample(100_000, params)
var_95 = -np.percentile(samples, 5)
```

### 5.2 Conditional Value at Risk (CVaR)

**Definition**: The expected loss given that loss exceeds VaR.

$$CVaR_\alpha(X) = \mathbb{E}[X | X \leq VaR_\alpha(X)]$$

Equivalently (Rockafellar-Uryasev formulation):

$$CVaR_\alpha(X) = \min_{\gamma} \left\{ \gamma + \frac{1}{\alpha} \mathbb{E}[(X - \gamma)^-] \right\}$$

Where $(x)^- = \max(-x, 0)$.

**Why CVaR over VaR?**

| Property | VaR | CVaR |
|----------|-----|------|
| Coherent risk measure | No | **Yes** |
| Sub-additive | No | **Yes** |
| Accounts for tail severity | No | **Yes** |
| Convex optimization | No | **Yes** |

**Computation**:

For continuous distributions:

$$CVaR_\alpha = \frac{1}{\alpha} \int_{-\infty}^{VaR_\alpha} x \cdot f(x) dx$$

```python
def cvar(dist, params, alpha):
    var = dist.ppf(alpha, params)

    # Numerical integration of tail
    x = np.linspace(dist.ppf(0.0001, params), var, 10000)
    pdf_values = dist.pdf(x, params)

    return np.trapezoid(x * pdf_values, x) / alpha
```

### 5.3 Kelly Criterion

**Objective**: Maximize expected log-wealth growth.

$$f^* = \arg\max_f \mathbb{E}[\log(1 + f \cdot r)]$$

**Continuous Approximation** (Taylor expansion around f=0):

$$\mathbb{E}[\log(1 + fr)] \approx f\mu - \frac{f^2}{2}\sigma^2$$

Taking derivative and setting to zero:

$$\frac{d}{df}[f\mu - \frac{f^2}{2}\sigma^2] = \mu - f\sigma^2 = 0$$

$$\boxed{f^* = \frac{\mu}{\sigma^2}}$$

**With Risk-Free Rate**:

$$f^* = \frac{\mu - r_f}{\sigma^2} = \frac{SR \cdot \sigma}{\sigma^2} = \frac{SR}{\sigma}$$

**Full Distribution Kelly** (numerical):

```python
def kelly_fraction(dist, params, f_range=(-2, 2), n_points=1000):
    """Find Kelly-optimal fraction using full distribution."""
    fractions = np.linspace(*f_range, n_points)

    # Sample from distribution
    samples = dist.sample(100_000, params)

    # Compute expected log growth for each fraction
    growth = np.array([
        np.mean(np.log(1 + f * samples))
        for f in fractions
    ])

    return fractions[np.argmax(growth)]
```

**Fractional Kelly**:

In practice, use a fraction $\kappa \in (0, 1)$ of full Kelly to reduce variance:

| $\kappa$ | Growth vs Full Kelly | Variance vs Full Kelly |
|----------|---------------------|------------------------|
| 1.0 | 100% | 100% |
| 0.5 | 75% | 25% |
| 0.25 | 44% | 6% |

```python
fractional_kelly = 0.5 * kelly_fraction(dist, params)
```

### 5.4 Probability Queries

Distributional predictions enable rich probability queries:

```python
class ProbabilityQueries:
    """Compute probabilities of events from distribution."""

    def prob_greater_than(self, dist, params, threshold):
        """P(X > threshold)"""
        return 1 - dist.cdf(threshold, params)

    def prob_less_than(self, dist, params, threshold):
        """P(X < threshold)"""
        return dist.cdf(threshold, params)

    def prob_between(self, dist, params, lower, upper):
        """P(lower < X < upper)"""
        return dist.cdf(upper, params) - dist.cdf(lower, params)

    def prob_loss_exceeds(self, dist, params, loss_threshold):
        """P(X < -loss_threshold) for returns"""
        return dist.cdf(-loss_threshold, params)

    def expected_gain_if_positive(self, dist, params):
        """E[X | X > 0]"""
        # Numerical integration
        x = np.linspace(0, dist.ppf(0.9999, params), 10000)
        pdf_vals = dist.pdf(x, params)
        prob_positive = 1 - dist.cdf(0, params)
        return np.trapezoid(x * pdf_vals, x) / prob_positive
```

---

## 6. ML Integration

### 6.1 How Models Produce Distribution Parameters

The key insight: instead of predicting $y$ directly, we predict $\theta = (\theta_1, ..., \theta_k)$—the parameters of a distribution $D(y; \theta)$.

**Training objective**: Minimize a proper scoring rule (typically negative log-likelihood):

$$\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \log p(y_i | X_i; \theta(X_i))$$

Where $\theta(X_i) = f(X_i; W)$ is the model's parameter prediction.

### 6.2 Architecture Options

#### Option A: Multi-Output Regression

```
Input Features ──► Shared Layers ──► Output Heads ──► [μ, log(σ), ν, β]
                        │
                        ├──► Head 1: μ (unbounded)
                        ├──► Head 2: log(σ) → softplus → σ > 0
                        ├──► Head 3: ν → softplus + 2 → ν > 2
                        └──► Head 4: β (unbounded for skew)
```

#### Option B: NGBoost-Style (Natural Gradients)

```python
# Pseudocode for NGBoost training
for iteration in range(n_iterations):
    # Current distribution parameters
    θ = model.predict(X)

    # Compute natural gradient of scoring rule
    grad = scoring_rule.natural_gradient(dist, θ, y)

    # Fit base learner to gradient
    base_learner.fit(X, grad)

    # Update model
    model.add_learner(base_learner, learning_rate)
```

### 6.3 sklearn-Compatible Interface

```python
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class DistributionalRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible wrapper for distributional regression.

    Parameters
    ----------
    distribution : Distribution
        The distribution family to use (e.g., NIG, Normal)
    base_estimator : estimator, default=None
        Base estimator for each parameter. If None, uses GradientBoostingRegressor
    scoring_rule : str, default='log_score'
        Proper scoring rule for training ('log_score', 'crps')

    Attributes
    ----------
    estimators_ : dict
        Fitted estimators for each distribution parameter
    """

    def __init__(
        self,
        distribution,
        base_estimator=None,
        scoring_rule='log_score',
    ):
        self.distribution = distribution
        self.base_estimator = base_estimator
        self.scoring_rule = scoring_rule

    def fit(self, X, y):
        """
        Fit distributional regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
        """
        X, y = self._validate_data(X, y)

        # Initialize estimators for each parameter
        self.estimators_ = {}
        for param_name in self.distribution.parameter_names:
            est = clone(self.base_estimator) if self.base_estimator else \
                  GradientBoostingRegressor(n_estimators=100)
            self.estimators_[param_name] = est

        # Iterative fitting with proper scoring rule
        self._fit_with_scoring_rule(X, y)

        return self

    def predict(self, X):
        """
        Predict distribution parameters.

        Returns
        -------
        params : DistributionParameters
            Predicted parameters for each sample
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        param_dict = {}
        for name, est in self.estimators_.items():
            param_dict[name] = est.predict(X)

        return self._make_params(param_dict)

    def predict_distribution(self, X):
        """
        Return full predictive distribution for each sample.

        Returns
        -------
        distributions : list of (Distribution, Parameters) tuples
        """
        params = self.predict(X)
        return [(self.distribution, p) for p in params]

    def predict_mean(self, X):
        """Predict expected value (for sklearn compatibility)."""
        params = self.predict(X)
        return np.array([self.distribution.mean(p) for p in params])

    def predict_var(self, X):
        """Predict variance."""
        params = self.predict(X)
        return np.array([self.distribution.variance(p) for p in params])

    def predict_quantile(self, X, q):
        """Predict quantile."""
        params = self.predict(X)
        return np.array([self.distribution.ppf(q, p) for p in params])

    def score(self, X, y):
        """
        Compute mean proper score (higher is better).

        Uses negative of the scoring rule so higher = better (sklearn convention).
        """
        params = self.predict(X)
        scores = [
            -self._score_fn(self.distribution, p, yi)
            for p, yi in zip(params, y)
        ]
        return np.mean(scores)
```

### 6.4 Training Workflow

```python
import temporalpdf as tpdf
from sklearn.model_selection import train_test_split

# 1. Load data
df = load_financial_data()
X = df[feature_columns].values
y = df['return'].values

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Initialize model
model = tpdf.DistributionalRegressor(
    distribution=tpdf.NIG(),
    scoring_rule='log_score',
)

# 4. Fit
model.fit(X_train, y_train)

# 5. Predict distributions
params_test = model.predict(X_test)

# 6. Evaluate with proper scores
log_scores = [
    tpdf.log_score(tpdf.NIG(), p, y)
    for p, y in zip(params_test, y_test)
]
crps_scores = [
    tpdf.crps(tpdf.NIG(), p, y)
    for p, y in zip(params_test, y_test)
]

print(f"Mean Log Score: {np.mean(log_scores):.4f}")
print(f"Mean CRPS: {np.mean(crps_scores):.4f}")

# 7. Decision utilities
for params in params_test[:5]:
    print(f"VaR 95%: {tpdf.var(tpdf.NIG(), params, 0.05):.2%}")
    print(f"CVaR 95%: {tpdf.cvar(tpdf.NIG(), params, 0.05):.2%}")
    print(f"Kelly: {tpdf.kelly(tpdf.NIG(), params):.1%}")
```

---

## 7. Evaluation Strategy

### 7.1 Benchmarking Against Point Predictions

To demonstrate value of distributional regression over point predictions:

| Metric | Point Model | Distributional Model | Interpretation |
|--------|-------------|---------------------|----------------|
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | $\frac{1}{n}\sum|y_i - \hat{\mu}_i|$ | Location accuracy |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | $\sqrt{\frac{1}{n}\sum(y_i - \hat{\mu}_i)^2}$ | Location + scale |
| CRPS | N/A | $\frac{1}{n}\sum CRPS(F_i, y_i)$ | Full distribution |
| Log Score | N/A | $\frac{1}{n}\sum \log p_i(y_i)$ | Likelihood |
| Coverage | N/A | $\frac{1}{n}\sum \mathbf{1}[y_i \in CI_i]$ | Calibration |

**Key insight**: Distributional models should match point models on MAE/RMSE (same location accuracy) while providing additional calibrated uncertainty.

### 7.2 Proper Scoring Rules

#### 7.2.1 Log Score (Negative Log-Likelihood)

$$S_{log}(F, y) = -\log f(y)$$

**Properties**:
- Strictly proper
- Sensitive to density at observation
- Can be infinite if $f(y) = 0$ (problematic for bounded distributions)
- Local score (only cares about density at $y$)

#### 7.2.2 Continuous Ranked Probability Score (CRPS)

$$CRPS(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbf{1}_{x \geq y})^2 dx$$

**Properties**:
- Strictly proper
- Always finite
- Equivalent to MAE when $F$ is point mass
- Global score (considers entire distribution)

**Closed forms exist** for many distributions:

For Normal $N(\mu, \sigma^2)$:

$$CRPS = \sigma \left[ \frac{y - \mu}{\sigma}(2\Phi(\frac{y-\mu}{\sigma}) - 1) + 2\phi(\frac{y-\mu}{\sigma}) - \frac{1}{\sqrt{\pi}} \right]$$

#### 7.2.3 Energy Score (Multivariate)

For $d$-dimensional forecasts:

$$ES(F, y) = \mathbb{E}\|X - y\| - \frac{1}{2}\mathbb{E}\|X - X'\|$$

Where $X, X' \sim F$ and $\|\cdot\|$ is typically Euclidean norm.

### 7.3 Calibration Assessment

A well-calibrated model produces PIT values that are uniformly distributed.

**Probability Integral Transform (PIT)**:

$$u_i = F_i(y_i) = P(Y \leq y_i | X_i)$$

If the model is well-calibrated, $u_i \sim Uniform(0, 1)$.

**Diagnostics**:

1. **PIT Histogram**: Should be flat (uniform)
2. **Kolmogorov-Smirnov Test**: $H_0$: PITs are uniform
3. **Reliability Diagram**: Plot observed frequency vs. predicted probability

```python
def calibration_plot(pit_values, n_bins=10):
    """Create calibration/reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    observed_freq = []
    expected_freq = []

    for i in range(n_bins):
        mask = (pit_values >= bins[i]) & (pit_values < bins[i+1])
        observed_freq.append(mask.mean())
        expected_freq.append(1 / n_bins)

    plt.bar(bins[:-1], observed_freq, width=1/n_bins, alpha=0.7)
    plt.axhline(1/n_bins, color='red', linestyle='--', label='Perfect calibration')
    plt.xlabel('PIT bin')
    plt.ylabel('Frequency')
    plt.legend()
```

### 7.4 Recommended Datasets

| Dataset | Description | Why Use It |
|---------|-------------|------------|
| **S&P 500 Returns** | Daily/weekly equity returns | Standard benchmark, known stylized facts |
| **VIX** | Volatility index | Test volatility forecasting |
| **FX Rates** | EUR/USD, GBP/USD | Different dynamics than equities |
| **Crypto** | BTC/ETH returns | Extreme tails, regime changes |
| **Treasury Yields** | 10Y rates | Mean-reverting processes |
| **UCI ML Repo** | Various regression datasets | Non-financial validation |

**Specific Recommendations**:

1. **Yahoo Finance via yfinance**: Free, easy access to equity data
2. **FRED**: Federal Reserve data for macro/rates
3. **Kaggle Financial Datasets**: Curated, documented
4. **OpenML**: Standardized benchmark tasks

---

## 8. API Design

### 8.1 Key Classes

```python
# Core distribution classes
class Distribution(Protocol):
    """Base distribution protocol."""

class NIG(Distribution):
    """Normal Inverse Gaussian distribution."""

class Normal(Distribution):
    """Normal/Gaussian distribution."""

class StudentT(Distribution):
    """Student's t distribution."""

class VarianceGamma(Distribution):
    """Variance Gamma distribution."""

# Parameter containers (immutable dataclasses)
@dataclass(frozen=True)
class NIGParameters:
    mu: float      # Location
    delta: float   # Scale (>0)
    alpha: float   # Steepness (>|beta|)
    beta: float    # Asymmetry

@dataclass(frozen=True)
class NormalParameters:
    mu: float      # Mean
    sigma: float   # Std dev (>0)

# Scoring rules
class LogScore:
    """Negative log-likelihood scoring rule."""

class CRPS:
    """Continuous Ranked Probability Score."""

# Risk measures
class VaR:
    """Value at Risk."""

class CVaR:
    """Conditional Value at Risk (Expected Shortfall)."""

class KellyCriterion:
    """Kelly optimal fraction calculator."""

# ML integration
class DistributionalRegressor(sklearn.BaseEstimator):
    """sklearn-compatible distributional regression."""

# Visualization
class DistributionPlotter:
    """Visualization utilities for distributions."""
```

### 8.2 Example Usage

#### Basic Distribution Operations

```python
import temporalpdf as tpdf
import numpy as np

# Create a NIG distribution
nig = tpdf.NIG()

# Define parameters
params = tpdf.NIGParameters(
    mu=0.001,      # 0.1% location
    delta=0.02,    # 2% scale
    alpha=15.0,    # Moderate tails
    beta=-2.0,     # Negative skew
)

# Evaluate PDF
x = np.linspace(-0.1, 0.1, 1000)
pdf_values = nig.pdf(x, params)

# Compute moments
print(f"Mean: {nig.mean(params):.4f}")
print(f"Std Dev: {np.sqrt(nig.variance(params)):.4f}")
print(f"Skewness: {nig.skewness(params):.4f}")
print(f"Kurtosis: {nig.kurtosis(params):.4f}")

# Quantiles
print(f"5th percentile: {nig.ppf(0.05, params):.4f}")
print(f"Median: {nig.ppf(0.50, params):.4f}")
print(f"95th percentile: {nig.ppf(0.95, params):.4f}")

# Sample
samples = nig.sample(10000, params, rng=np.random.default_rng(42))
```

#### Risk Measures

```python
# VaR at 95% confidence (5% tail)
var_95 = tpdf.var(nig, params, alpha=0.05)
print(f"95% VaR: {var_95:.2%}")

# CVaR (Expected Shortfall)
cvar_95 = tpdf.cvar(nig, params, alpha=0.05)
print(f"95% CVaR: {cvar_95:.2%}")

# Kelly fraction
kelly_f = tpdf.kelly(nig, params)
print(f"Kelly fraction: {kelly_f:.1%}")

# Probability queries
prob_loss_5pct = tpdf.prob_less_than(nig, params, -0.05)
print(f"P(loss > 5%): {prob_loss_5pct:.2%}")
```

#### ML Training

```python
import temporalpdf as tpdf
from sklearn.model_selection import cross_val_score

# Prepare data
X, y = load_features_and_returns()

# Initialize distributional regressor
model = tpdf.DistributionalRegressor(
    distribution=tpdf.NIG(),
    scoring_rule='crps',
    n_estimators=200,
    learning_rate=0.05,
)

# Cross-validation with CRPS as metric
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring=tpdf.make_scorer('crps'),
)
print(f"CV CRPS: {-scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit final model
model.fit(X, y)

# Predict on new data
X_new = get_new_features()
params_pred = model.predict(X_new)

# Get distributional predictions
for i, params in enumerate(params_pred[:3]):
    print(f"\nSample {i}:")
    print(f"  Predicted mean: {tpdf.NIG().mean(params):.2%}")
    print(f"  Predicted vol: {np.sqrt(tpdf.NIG().variance(params)):.2%}")
    print(f"  95% VaR: {tpdf.var(tpdf.NIG(), params, 0.05):.2%}")
    print(f"  Kelly: {tpdf.kelly(tpdf.NIG(), params):.1%}")
```

#### Time-Evolving Distributions

```python
import temporalpdf as tpdf

# Parameters that evolve over time
time_params = tpdf.TimeEvolvingNIGParameters(
    mu_0=0.001,        # Initial location
    delta_0=0.02,      # Initial scale
    alpha=15.0,        # Fixed steepness
    beta=-2.0,         # Fixed asymmetry
    mu_drift=0.0001,   # Location drifts up
    delta_growth=0.1,  # Scale grows 10% per unit time
)

# Evaluate over 30-day horizon
result = tpdf.evaluate_time_evolving(
    distribution=tpdf.NIG(),
    params=time_params,
    time_range=(0, 30),
    value_range=(-0.2, 0.2),
)

# Visualize
plotter = tpdf.DistributionPlotter()
fig = plotter.surface_3d(result, title="30-Day Forecast Distribution")
plotter.save(fig, "forecast.png")

# Time-dependent risk measures
for t in [1, 7, 30]:
    params_t = time_params.at_time(t)
    print(f"\nDay {t}:")
    print(f"  VaR 95%: {tpdf.var(tpdf.NIG(), params_t, 0.05):.2%}")
    print(f"  CVaR 95%: {tpdf.cvar(tpdf.NIG(), params_t, 0.05):.2%}")
```

#### Model Evaluation

```python
import temporalpdf as tpdf

# After training and predicting
y_test = actual_returns
params_pred = model.predict(X_test)

# Compute proper scores
log_scores = tpdf.log_score_batch(tpdf.NIG(), params_pred, y_test)
crps_scores = tpdf.crps_batch(tpdf.NIG(), params_pred, y_test)

print(f"Mean Log Score: {np.mean(log_scores):.4f}")
print(f"Mean CRPS: {np.mean(crps_scores):.4f}")

# Calibration check
pit_values = tpdf.pit_batch(tpdf.NIG(), params_pred, y_test)
tpdf.plot_pit_histogram(pit_values)

# Compare to baseline point prediction
point_model = GradientBoostingRegressor()
point_model.fit(X_train, y_train)
y_pred_point = point_model.predict(X_test)

# Point model metrics
point_mae = np.mean(np.abs(y_test - y_pred_point))
point_rmse = np.sqrt(np.mean((y_test - y_pred_point)**2))

# Distributional model point metrics (using mean)
dist_means = np.array([tpdf.NIG().mean(p) for p in params_pred])
dist_mae = np.mean(np.abs(y_test - dist_means))
dist_rmse = np.sqrt(np.mean((y_test - dist_means)**2))

print(f"\nPoint Model - MAE: {point_mae:.4f}, RMSE: {point_rmse:.4f}")
print(f"Dist Model  - MAE: {dist_mae:.4f}, RMSE: {dist_rmse:.4f}")
print(f"Dist Model  - CRPS: {np.mean(crps_scores):.4f} (no point equivalent)")
```

---

## Appendix A: Mathematical Reference

### A.1 NIG Distribution Formulas

**PDF**:
$$f(x; \alpha, \beta, \mu, \delta) = \frac{\alpha \delta}{\pi} \exp(\delta\gamma + \beta(x-\mu)) \frac{K_1(\alpha\sqrt{\delta^2 + (x-\mu)^2})}{\sqrt{\delta^2 + (x-\mu)^2}}$$

Where $\gamma = \sqrt{\alpha^2 - \beta^2}$ and $K_1$ is the modified Bessel function.

**Moments**:
$$\mathbb{E}[X] = \mu + \frac{\delta\beta}{\gamma}$$

$$\text{Var}(X) = \frac{\delta\alpha^2}{\gamma^3}$$

$$\text{Skew}(X) = \frac{3\beta}{\alpha\sqrt{\delta\gamma}}$$

$$\text{Kurt}(X) = 3 + \frac{3(1 + 4\beta^2/\alpha^2)}{\delta\gamma}$$

### A.2 CRPS Formulas

**General**:
$$CRPS(F, y) = \mathbb{E}|X - y| - \frac{1}{2}\mathbb{E}|X - X'|$$

**Normal** $N(\mu, \sigma^2)$:
$$CRPS = \sigma \left[ z(2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]$$

Where $z = (y - \mu)/\sigma$.

### A.3 Kelly Derivation

Starting from:
$$\max_f \mathbb{E}[\log(1 + fr)]$$

Taylor expand $\log(1 + fr)$ around $fr = 0$:
$$\log(1 + fr) \approx fr - \frac{(fr)^2}{2} + O(f^3r^3)$$

Take expectation:
$$\mathbb{E}[\log(1 + fr)] \approx f\mathbb{E}[r] - \frac{f^2}{2}\mathbb{E}[r^2]$$
$$= f\mu - \frac{f^2}{2}(\sigma^2 + \mu^2)$$

For small $\mu$:
$$\approx f\mu - \frac{f^2\sigma^2}{2}$$

Differentiate and set to zero:
$$\frac{d}{df} = \mu - f\sigma^2 = 0$$
$$\boxed{f^* = \frac{\mu}{\sigma^2}}$$

---

## Appendix B: References

1. Duan, T., et al. (2020). NGBoost: Natural Gradient Boosting for Probabilistic Prediction. *ICML*.
2. Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
3. Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
4. Barndorff-Nielsen, O.E. (1997). Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling. *Scandinavian Journal of Statistics*, 24(1), 1-13.
5. Barndorff-Nielsen, O.E. (1998). Processes of Normal Inverse Gaussian Type. *Finance and Stochastics*, 2(1), 41-68.
6. Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477), 359-378.
7. Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2, 21-41.
8. Rockafellar, R.T. & Uryasev, S. (2002). Conditional Value-at-Risk for General Loss Distributions. *Journal of Banking & Finance*, 26(7), 1443-1471.
9. Kelly, J.L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal*, 35(4), 917-926.
10. Thorp, E.O. (2006). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market. *Handbook of Asset and Liability Management*, 1, 385-428.
11. Konlack Socgnia, V. & Wilcox, D. (2014). A Comparison of Generalized Hyperbolic Distribution Models for Equity Returns. *Journal of Applied Mathematics*.
12. Hu, W. (2016). Variance-Gamma and Normal-Inverse Gaussian Models. *North American Journal of Economics and Finance*, 36, 279-292.
