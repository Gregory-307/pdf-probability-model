"""
Real data example: Nordic stock returns distributional analysis.

This example demonstrates temporalpdf with real financial data from
Nordic stocks (2014-2016).

Key points:
1. Uses REAL market data (included in data/sample_returns.csv)
2. Fits NIG distribution parameters using MLE
3. Evaluates with proper scoring rules (CRPS)
4. Computes risk measures (VaR, CVaR) on actual predictions
5. Compares NIG to Normal baseline
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats as scipy_stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import temporalpdf as tpdf


def load_sample_data() -> pd.DataFrame:
    """Load included sample data (Nordic stock returns)."""
    data_path = Path(__file__).parent.parent / "data" / "sample_returns.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {data_path}. "
            "Please ensure data/sample_returns.csv exists."
        )

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path.name}")
    print(f"Companies: {df['COMPANYNAME'].nunique()}")
    print(f"Date range: {df['ASOFDATE'].min()} to {df['ASOFDATE'].max()}")
    return df


def fit_nig_mle(
    returns: np.ndarray,
    verbose: bool = True,
) -> tuple[tpdf.NIGParameters, dict]:
    """
    Fit NIG parameters using Maximum Likelihood Estimation.

    Uses scipy.optimize to maximize the log-likelihood.

    Returns:
        params: Fitted NIGParameters
        fit_info: Dictionary with fitting diagnostics
    """
    nig = tpdf.NIG()

    # Initial guess using method of moments
    mean = np.mean(returns)
    std = np.std(returns)
    skew = scipy_stats.skew(returns)

    # Starting values (in log-space for positive parameters)
    x0 = np.array([
        mean,           # mu
        np.log(std),    # log(delta)
        np.log(10.0),   # log(alpha)
        0.0,            # beta (will be constrained)
    ])

    def neg_log_likelihood(theta):
        mu, log_delta, log_alpha, beta_raw = theta
        delta = np.exp(log_delta)
        alpha = np.exp(log_alpha)

        # Constraint: |beta| < alpha
        beta = alpha * np.tanh(beta_raw)  # Maps R to (-alpha, alpha)

        try:
            params = tpdf.NIGParameters(mu=mu, delta=delta, alpha=alpha, beta=beta)
            pdf_vals = nig.pdf(returns, 0, params)
            pdf_vals = np.maximum(pdf_vals, 1e-300)
            return -np.sum(np.log(pdf_vals))
        except (ValueError, RuntimeWarning):
            return 1e10

    # Optimize
    result = optimize.minimize(
        neg_log_likelihood,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
    )

    # Extract parameters
    mu, log_delta, log_alpha, beta_raw = result.x
    delta = np.exp(log_delta)
    alpha = np.exp(log_alpha)
    beta = alpha * np.tanh(beta_raw)

    params = tpdf.NIGParameters(mu=mu, delta=delta, alpha=alpha, beta=beta)

    fit_info = {
        'converged': result.success,
        'log_likelihood': -result.fun,
        'n_iterations': result.nit,
        'aic': 2 * 4 + 2 * result.fun,  # 4 parameters
        'bic': 4 * np.log(len(returns)) + 2 * result.fun,
    }

    if verbose:
        print(f"  Converged: {result.success}")
        print(f"  Log-likelihood: {-result.fun:.2f}")
        print(f"  AIC: {fit_info['aic']:.2f}")
        print(f"  BIC: {fit_info['bic']:.2f}")

    return params, fit_info


def compute_calibration_metrics(
    returns: np.ndarray,
    nig: tpdf.NIGDistribution,
    params: tpdf.NIGParameters,
) -> dict:
    """
    Compute calibration diagnostics.

    Returns PIT (Probability Integral Transform) values and
    Kolmogorov-Smirnov test against uniform.
    """
    # PIT: F(y) should be uniform if model is well-calibrated
    pit_values = nig.cdf(returns, 0, params)

    # KS test against uniform
    ks_stat, ks_pvalue = scipy_stats.kstest(pit_values, 'uniform')

    return {
        'pit_values': pit_values,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'is_calibrated': ks_pvalue > 0.05,  # At 5% significance
    }


def main() -> None:
    """Run real data analysis with comprehensive evaluation."""
    print("=" * 70)
    print("temporalpdf: Real Data Analysis")
    print("Normal Inverse Gaussian (NIG) Distribution on Nordic Stock Returns")
    print("=" * 70)
    print()

    # Load real data
    df = load_sample_data()
    print()

    # Extract returns
    returns = df["target"].dropna().values

    print(f"Sample Statistics:")
    print(f"  N observations: {len(returns):,}")
    print(f"  Mean: {np.mean(returns):.4f}%")
    print(f"  Std Dev: {np.std(returns):.4f}%")
    print(f"  Skewness: {scipy_stats.skew(returns):.4f}")
    print(f"  Kurtosis: {scipy_stats.kurtosis(returns):.4f}")
    print(f"  Min: {np.min(returns):.2f}%")
    print(f"  Max: {np.max(returns):.2f}%")
    print()

    # Split into train/test
    n_train = int(len(returns) * 0.8)
    train_returns = returns[:n_train]
    test_returns = returns[n_train:]

    print(f"Train/Test Split: {n_train:,} / {len(test_returns):,}")
    print()

    # Fit NIG using MLE
    print("Fitting NIG distribution (MLE)...")
    nig = tpdf.NIG()
    params, fit_info = fit_nig_mle(train_returns)
    print(f"  mu (location): {params.mu:.6f}")
    print(f"  delta (scale): {params.delta:.6f}")
    print(f"  alpha (tail heaviness): {params.alpha:.4f}")
    print(f"  beta (skewness): {params.beta:.4f}")
    print()

    # Fitted distribution moments
    print("Fitted Distribution Moments:")
    fitted_mean = nig.mean(0, params)
    fitted_var = nig.variance(0, params)
    fitted_skew = nig.skewness(0, params)
    fitted_kurt = nig.kurtosis(0, params)
    print(f"  Mean: {fitted_mean:.4f}% (sample: {np.mean(train_returns):.4f}%)")
    print(f"  Std: {np.sqrt(fitted_var):.4f}% (sample: {np.std(train_returns):.4f}%)")
    print(f"  Skewness: {fitted_skew:.4f} (sample: {scipy_stats.skew(train_returns):.4f})")
    print(f"  Kurtosis: {fitted_kurt:.4f} (sample: {scipy_stats.kurtosis(train_returns):.4f})")
    print()

    # Calibration check
    print("Calibration Check (PIT Uniformity)...")
    calib = compute_calibration_metrics(train_returns, nig, params)
    print(f"  KS Statistic: {calib['ks_statistic']:.4f}")
    print(f"  KS p-value: {calib['ks_pvalue']:.4f}")
    print(f"  Calibrated (p > 0.05): {calib['is_calibrated']}")
    print()

    # Risk measures
    print("Risk Measures (from fitted distribution):")
    var_95 = tpdf.var(nig, params, alpha=0.05)
    var_99 = tpdf.var(nig, params, alpha=0.01)
    cvar_95 = tpdf.cvar(nig, params, alpha=0.05, n_samples=50000)
    kelly = tpdf.kelly_fraction(nig, params)

    print(f"  VaR 95%: {var_95:.2f}% (5% chance of losing more)")
    print(f"  VaR 99%: {var_99:.2f}% (1% chance of losing more)")
    print(f"  CVaR 95%: {cvar_95:.2f}% (expected loss in worst 5%)")
    print(f"  Kelly fraction: {kelly:.2%}")
    print()

    # Compare NIG vs Normal on test set using CRPS
    print("Out-of-Sample Evaluation (Test Set):")
    print("-" * 50)

    # NIG CRPS
    rng = np.random.default_rng(42)
    crps_nig = []
    for y in test_returns:
        score = tpdf.crps(nig, params, y, t=0, n_samples=5000, rng=rng)
        crps_nig.append(score)
    mean_crps_nig = np.mean(crps_nig)

    # Normal baseline (fit on train)
    normal_mu = np.mean(train_returns)
    normal_sigma = np.std(train_returns)
    crps_normal = []
    for y in test_returns:
        score = tpdf.crps_normal(y, normal_mu, normal_sigma)
        crps_normal.append(score)
    mean_crps_normal = np.mean(crps_normal)

    # Log scores
    log_scores_nig = []
    for y in test_returns:
        score = tpdf.log_score(nig, params, y, t=0)
        log_scores_nig.append(score)
    mean_log_nig = np.mean(log_scores_nig)

    normal_dist = scipy_stats.norm(normal_mu, normal_sigma)
    log_scores_normal = -normal_dist.logpdf(test_returns)
    mean_log_normal = np.mean(log_scores_normal)

    print(f"  CRPS (lower is better):")
    print(f"    Normal: {mean_crps_normal:.4f}")
    print(f"    NIG:    {mean_crps_nig:.4f}")
    improvement_crps = (mean_crps_normal - mean_crps_nig) / mean_crps_normal * 100
    print(f"    Improvement: {improvement_crps:+.1f}%")
    print()

    print(f"  Log Score (lower is better):")
    print(f"    Normal: {mean_log_normal:.4f}")
    print(f"    NIG:    {mean_log_nig:.4f}")
    improvement_log = (mean_log_normal - mean_log_nig) / mean_log_normal * 100
    print(f"    Improvement: {improvement_log:+.1f}%")
    print()

    # Probability queries
    print("Probability Queries:")
    print(f"  P(return > 0%): {tpdf.prob_greater_than(nig, params, 0):.1%}")
    print(f"  P(return > 5%): {tpdf.prob_greater_than(nig, params, 5):.1%}")
    print(f"  P(return < -5%): {tpdf.prob_less_than(nig, params, -5):.1%}")
    print(f"  P(-2% < return < 2%): {tpdf.prob_between(nig, params, -2, 2):.1%}")
    print()

    print("=" * 70)
    print("Analysis complete.")
    print()
    print("Key Findings:")
    print(f"  - NIG provides {improvement_crps:+.1f}% improvement in CRPS over Normal")
    print(f"  - Distribution is {'well-calibrated' if calib['is_calibrated'] else 'poorly calibrated'}")
    print(f"  - Recommended Kelly fraction: {kelly:.1%} (use {0.5*kelly:.1%} for half-Kelly)")
    print("=" * 70)


if __name__ == "__main__":
    main()
