"""
Rolling backtest framework for VaR model evaluation.
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from .tests import kupiec_test, christoffersen_test, conditional_coverage_test
from ..utilities import fit
from ..decision import var


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes:
        var_forecasts: Array of VaR forecasts
        actual_returns: Array of actual returns
        exceedances: Boolean array of VaR exceedances
        exceedance_rate: Fraction of exceedances
        n_exceedances: Number of exceedances
        n_total: Total number of forecasts
        kupiec_stat: Kupiec test statistic
        kupiec_pvalue: Kupiec test p-value
        kupiec_pass: Whether Kupiec test passed
        christoffersen_stat: Christoffersen test statistic
        christoffersen_pvalue: Christoffersen test p-value
        christoffersen_pass: Whether Christoffersen test passed
        status: Overall pass/fail status
    """
    var_forecasts: NDArray[np.float64]
    actual_returns: NDArray[np.float64]
    exceedances: NDArray[np.bool_]
    exceedance_rate: float
    n_exceedances: int
    n_total: int
    kupiec_stat: float
    kupiec_pvalue: float
    kupiec_pass: bool
    christoffersen_stat: float
    christoffersen_pvalue: float
    christoffersen_pass: bool
    status: str

    def summary(self) -> str:
        """Generate summary string."""
        return "\n".join([
            "Backtest Results",
            "=" * 50,
            f"Forecasts:           {self.n_total}",
            f"Exceedances:         {self.n_exceedances} ({self.exceedance_rate:.1%})",
            "",
            f"Kupiec p-value:      {self.kupiec_pvalue:.4f} ({'PASS' if self.kupiec_pass else 'FAIL'})",
            f"Christoffersen p:    {self.christoffersen_pvalue:.4f} ({'PASS' if self.christoffersen_pass else 'FAIL'})",
            "",
            f"Overall Status:      {self.status}",
        ])


@dataclass
class Backtest:
    """
    Rolling backtest framework.

    At each time step:
    1. Fit distribution to lookback window
    2. Compute VaR forecast
    3. Record whether actual return exceeded VaR
    4. Evaluate exceedance rate and run statistical tests

    Example:
        >>> bt = Backtest(distribution='nig', lookback=252, alpha=0.05)
        >>> result = bt.run(returns)
        >>> print(result.summary())
    """
    distribution: Literal["normal", "student_t", "nig"]
    lookback: int
    forecast_horizon: int = 1
    alpha: float = 0.05
    step: int = 1
    significance: float = 0.05

    # Internal state
    _var_forecasts: list = field(default_factory=list, init=False, repr=False)
    _actual_returns: list = field(default_factory=list, init=False, repr=False)
    _exceedances: list = field(default_factory=list, init=False, repr=False)

    def run(self, data: NDArray[np.float64]) -> BacktestResult:
        """
        Execute the rolling backtest.

        Args:
            data: Array of returns

        Returns:
            BacktestResult with statistics and test results
        """
        data = np.asarray(data)
        n = len(data)

        self._var_forecasts = []
        self._actual_returns = []
        self._exceedances = []

        for i in range(self.lookback, n - self.forecast_horizon + 1, self.step):
            window = data[i - self.lookback:i]
            actual = data[i + self.forecast_horizon - 1]

            # Fit and compute VaR
            var_forecast = self._compute_var(window)

            self._var_forecasts.append(var_forecast)
            self._actual_returns.append(actual)
            self._exceedances.append(-actual > var_forecast)

        var_forecasts = np.array(self._var_forecasts)
        actual_returns = np.array(self._actual_returns)
        exceedances = np.array(self._exceedances)

        # Compute statistics
        exceedance_rate = float(np.mean(exceedances))
        n_exceedances = int(np.sum(exceedances))
        n_total = len(exceedances)

        # Run tests
        kup_stat, kup_p, kup_reject = kupiec_test(exceedances, self.alpha, self.significance)
        chr_stat, chr_p, chr_reject = christoffersen_test(exceedances, self.significance)

        # Determine status
        if not kup_reject and not chr_reject:
            status = "PASS"
        elif kup_reject and chr_reject:
            status = "FAIL_BOTH"
        elif kup_reject:
            status = "FAIL_COVERAGE"
        else:
            status = "FAIL_INDEPENDENCE"

        return BacktestResult(
            var_forecasts=var_forecasts,
            actual_returns=actual_returns,
            exceedances=exceedances,
            exceedance_rate=exceedance_rate,
            n_exceedances=n_exceedances,
            n_total=n_total,
            kupiec_stat=kup_stat,
            kupiec_pvalue=kup_p,
            kupiec_pass=not kup_reject,
            christoffersen_stat=chr_stat,
            christoffersen_pvalue=chr_p,
            christoffersen_pass=not chr_reject,
            status=status,
        )

    def _compute_var(self, window: NDArray[np.float64]) -> float:
        """Compute VaR from window data."""
        if self.distribution == "historical":
            return -float(np.percentile(window, self.alpha * 100))

        params = fit(window, distribution=self.distribution)

        if self.distribution == "normal":
            from scipy import stats as sp_stats
            return -sp_stats.norm.ppf(self.alpha, params.mu_0, params.sigma_0)

        elif self.distribution == "student_t":
            from scipy import stats as sp_stats
            return -sp_stats.t.ppf(self.alpha, params.nu, params.mu_0, params.sigma_0)

        elif self.distribution == "nig":
            from ..distributions.nig import NIGDistribution
            nig = NIGDistribution()
            return var(nig, params, alpha=self.alpha, t=0.0)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


def compare_backtests(
    data: NDArray[np.float64],
    distributions: list[str],
    lookback: int = 252,
    alpha: float = 0.05,
) -> dict:
    """
    Compare multiple distributions via backtesting.

    Args:
        data: Returns data
        distributions: List of distribution names to compare
        lookback: Lookback window
        alpha: VaR confidence level

    Returns:
        Dictionary with results for each distribution
    """
    results = {}

    for dist in distributions:
        bt = Backtest(distribution=dist, lookback=lookback, alpha=alpha)
        result = bt.run(data)
        results[dist] = {
            "exceedance_rate": result.exceedance_rate,
            "kupiec_pvalue": result.kupiec_pvalue,
            "christoffersen_pvalue": result.christoffersen_pvalue,
            "status": result.status,
        }

    return results
