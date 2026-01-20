"""
Predictive distribution that integrates over parameter uncertainty.

The predictive distribution is:
    P(r_t) = integral P(r | theta) * P(theta | data, t) d_theta

Approximated via Monte Carlo by:
1. Sample parameter paths from dynamics models
2. For each path, sample from the conditional distribution
3. Aggregate samples
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from ..core.result import RiskMetric, DecisionSummary

if TYPE_CHECKING:
    from ..distributions.nig import NIGDistribution, NIGParameters
    from ..distributions.normal import NormalDistribution
    from ..distributions.student_t import StudentTDistribution


def get_distribution(name: str):
    """Get distribution instance by name."""
    from ..distributions.nig import NIGDistribution
    from ..distributions.normal import NormalDistribution
    from ..distributions.student_t import StudentTDistribution
    from ..distributions.skew_normal import SkewNormalDistribution

    distributions = {
        "nig": NIGDistribution,
        "normal": NormalDistribution,
        "student_t": StudentTDistribution,
        "skew_normal": SkewNormalDistribution,
    }
    if name not in distributions:
        raise ValueError(f"Unknown distribution: {name}")
    return distributions[name]()


def create_params(distribution: str, **kwargs):
    """Create parameter object for distribution."""
    from ..distributions.nig import NIGParameters
    from ..core.parameters import NormalParameters, StudentTParameters, SkewNormalParameters

    if distribution == "nig":
        return NIGParameters(
            mu=kwargs.get("mu", 0.0),
            delta=kwargs.get("delta", 0.02),
            alpha=kwargs.get("alpha", 15.0),
            beta=kwargs.get("beta", 0.0),
        )
    elif distribution == "normal":
        return NormalParameters(
            mu_0=kwargs.get("mu_0", kwargs.get("mu", 0.0)),
            sigma_0=kwargs.get("sigma_0", kwargs.get("sigma", 0.02)),
        )
    elif distribution == "student_t":
        return StudentTParameters(
            mu_0=kwargs.get("mu_0", kwargs.get("mu", 0.0)),
            sigma_0=kwargs.get("sigma_0", kwargs.get("sigma", 0.02)),
            nu=kwargs.get("nu", 5.0),
        )
    elif distribution == "skew_normal":
        return SkewNormalParameters(
            mu_0=kwargs.get("mu_0", kwargs.get("mu", 0.0)),
            sigma_0=kwargs.get("sigma_0", kwargs.get("sigma", 0.02)),
            alpha=kwargs.get("alpha", 0.0),
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


@dataclass
class PredictiveDistribution:
    """
    Predictive distribution integrating over parameter uncertainty.

    This is the key class for decision-making under uncertainty.
    It provides:
    - Point estimates (mean, median)
    - Risk measures (VaR, CVaR)
    - Position sizing (Kelly)
    - Full quantile function

    All with uncertainty quantification via confidence intervals.

    Attributes:
        distribution: Name of distribution ('nig', 'normal', 'student_t')
        param_paths: Parameter samples at time t ({param: array of samples})
        t: Forecast horizon
        n_samples: Number of samples to draw from predictive distribution
    """
    distribution: str
    param_paths: dict[str, NDArray[np.float64]]  # At specific t
    t: int
    n_samples: int = 10000
    _samples: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def _ensure_samples(self) -> None:
        """Generate samples from predictive distribution (lazy evaluation)."""
        if self._samples is not None:
            return

        dist = get_distribution(self.distribution)
        samples = []

        n_paths = len(next(iter(self.param_paths.values())))
        samples_per_path = max(1, self.n_samples // n_paths)

        for i in range(n_paths):
            # Get params for this path
            param_dict = {k: float(v[i]) for k, v in self.param_paths.items()}

            try:
                params = create_params(self.distribution, **param_dict)
                # Sample from distribution with these params
                path_samples = dist.sample(samples_per_path, t=0, params=params)
                samples.extend(path_samples)
            except Exception:
                # Skip invalid parameter combinations
                continue

        self._samples = np.array(samples) if samples else np.zeros(100)

    @property
    def samples(self) -> NDArray[np.float64]:
        """Get samples from predictive distribution."""
        self._ensure_samples()
        return self._samples

    def mean(self) -> float:
        """Expected value of predictive distribution."""
        return float(np.mean(self.samples))

    def std(self) -> float:
        """Standard deviation of predictive distribution."""
        return float(np.std(self.samples))

    def median(self) -> float:
        """Median of predictive distribution."""
        return float(np.median(self.samples))

    def quantile(self, q: float) -> float:
        """Quantile q of predictive distribution."""
        return float(np.quantile(self.samples, q))

    def var(self, alpha: float = 0.05) -> float:
        """
        Value at Risk.

        The loss threshold exceeded with probability alpha.
        Positive VaR means potential loss.

        Args:
            alpha: Exceedance probability (0.05 = 95% VaR)

        Returns:
            VaR value (positive = loss)
        """
        return -self.quantile(alpha)

    def cvar(self, alpha: float = 0.05) -> float:
        """
        Conditional VaR (Expected Shortfall).

        Expected loss given we're in the tail.

        Args:
            alpha: Tail probability

        Returns:
            CVaR value (positive = loss)
        """
        threshold = self.quantile(alpha)
        tail_samples = self.samples[self.samples <= threshold]
        if len(tail_samples) > 0:
            return -float(np.mean(tail_samples))
        return -threshold

    def prob_profit(self) -> float:
        """Probability of positive return."""
        return float(np.mean(self.samples > 0))

    def _compute_kelly(self) -> float:
        """Kelly fraction = mean / variance."""
        var = self.std() ** 2
        if var < 1e-10:
            return 0.0
        return self.mean() / var

    def decision_summary(
        self,
        alpha: float = 0.05,
        confidence_level: float = 0.90,
    ) -> DecisionSummary:
        """
        Compute full decision summary with confidence intervals.

        CIs are computed by bootstrapping over the samples.

        Args:
            alpha: VaR/CVaR confidence level
            confidence_level: Confidence level for CIs

        Returns:
            DecisionSummary with all metrics and CIs
        """
        self._ensure_samples()

        # Point estimates
        var_val = self.var(alpha)
        cvar_val = self.cvar(alpha)
        kelly_val = self._compute_kelly()
        prob_val = self.prob_profit()

        # Bootstrap CIs
        n_bootstrap = 1000
        var_boots, cvar_boots, kelly_boots, prob_boots = [], [], [], []

        samples = self.samples
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(samples), len(samples), replace=True)
            boot_samples = samples[boot_idx]

            # VaR
            var_boots.append(-np.quantile(boot_samples, alpha))

            # CVaR
            threshold = np.quantile(boot_samples, alpha)
            tail = boot_samples[boot_samples <= threshold]
            cvar_boots.append(-np.mean(tail) if len(tail) > 0 else -threshold)

            # Kelly
            boot_var = np.var(boot_samples)
            kelly_boots.append(np.mean(boot_samples) / (boot_var + 1e-10))

            # Prob profit
            prob_boots.append(np.mean(boot_samples > 0))

        ci_low = (1 - confidence_level) / 2
        ci_high = 1 - ci_low

        return DecisionSummary(
            var=RiskMetric(
                value=var_val,
                confidence_interval=(
                    float(np.quantile(var_boots, ci_low)),
                    float(np.quantile(var_boots, ci_high))
                ),
            ),
            cvar=RiskMetric(
                value=cvar_val,
                confidence_interval=(
                    float(np.quantile(cvar_boots, ci_low)),
                    float(np.quantile(cvar_boots, ci_high))
                ),
            ),
            kelly=RiskMetric(
                value=kelly_val,
                confidence_interval=(
                    float(np.quantile(kelly_boots, ci_low)),
                    float(np.quantile(kelly_boots, ci_high))
                ),
            ),
            prob_profit=RiskMetric(
                value=prob_val,
                confidence_interval=(
                    float(np.quantile(prob_boots, ci_low)),
                    float(np.quantile(prob_boots, ci_high))
                ),
            ),
            expected_return=self.mean(),
            volatility=self.std(),
            t=self.t,
            alpha=alpha,
        )
