"""
Utility functions for common tasks.

Includes:
- Distribution fitting (MLE)
- Automatic distribution selection
- Rolling VaR backtesting
- Model comparison utilities
"""

from typing import Literal, Sequence, Union
import numpy as np
from scipy import stats, optimize

from .distributions.nig import NIGDistribution, NIGParameters
from .distributions.student_t import StudentTDistribution
from .distributions.normal import NormalDistribution
from .core.parameters import StudentTParameters, NormalParameters
from .scoring.rules import crps, crps_normal
from .decision.risk import var


# =============================================================================
# FITTING FUNCTIONS - The core API
# =============================================================================

def fit_nig(data: np.ndarray) -> NIGParameters:
    """
    Fit NIG distribution to data via maximum likelihood.

    Args:
        data: Array of observations (e.g., returns)

    Returns:
        NIGParameters with fitted mu, delta, alpha, beta

    Example:
        >>> params = tpdf.fit_nig(returns)
        >>> print(f"mu={params.mu:.4f}, delta={params.delta:.4f}")
    """
    data = np.asarray(data)
    nig = NIGDistribution()

    # Initial guess
    x0 = [np.mean(data), np.log(np.std(data) + 0.01), np.log(5.0), 0.0]

    def nll(theta):
        try:
            mu = theta[0]
            delta = np.exp(theta[1])
            alpha = np.exp(theta[2])
            beta = alpha * np.tanh(theta[3])
            p = NIGParameters(mu=mu, delta=delta, alpha=alpha, beta=beta)
            pdf_vals = nig.pdf(data, 0, p)
            return -np.sum(np.log(np.maximum(pdf_vals, 1e-300)))
        except:
            return 1e10

    result = optimize.minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 1000})
    mu, log_d, log_a, br = result.x

    return NIGParameters(
        mu=mu,
        delta=np.exp(log_d),
        alpha=np.exp(log_a),
        beta=np.exp(log_a) * np.tanh(br)
    )


def fit_student_t(data: np.ndarray) -> StudentTParameters:
    """
    Fit Student-t distribution to data via maximum likelihood.

    Args:
        data: Array of observations

    Returns:
        StudentTParameters with fitted nu, mu_0, sigma_0

    Example:
        >>> params = tpdf.fit_student_t(returns)
        >>> print(f"nu={params.nu:.2f}, mu={params.mu_0:.4f}")
    """
    data = np.asarray(data)
    nu, loc, scale = stats.t.fit(data)
    return StudentTParameters(nu=nu, mu_0=loc, sigma_0=scale)


def fit_normal(data: np.ndarray) -> NormalParameters:
    """
    Fit Normal distribution to data.

    Args:
        data: Array of observations

    Returns:
        NormalParameters with fitted mu_0, sigma_0

    Example:
        >>> params = tpdf.fit_normal(returns)
        >>> print(f"mu={params.mu_0:.4f}, sigma={params.sigma_0:.4f}")
    """
    data = np.asarray(data)
    return NormalParameters(mu_0=np.mean(data), sigma_0=np.std(data))


def fit(
    data: np.ndarray,
    distribution: Literal["nig", "student_t", "normal"] = "nig"
) -> Union[NIGParameters, StudentTParameters, NormalParameters]:
    """
    Fit a distribution to data via maximum likelihood.

    Args:
        data: Array of observations (e.g., returns)
        distribution: Which distribution to fit ('nig', 'student_t', 'normal')

    Returns:
        Fitted parameters (NIGParameters, StudentTParameters, or NormalParameters)

    Example:
        >>> params = tpdf.fit(returns, distribution='nig')
        >>> var_5 = tpdf.var(tpdf.NIG(), params, alpha=0.05)
    """
    if distribution == "nig":
        return fit_nig(data)
    elif distribution == "student_t":
        return fit_student_t(data)
    elif distribution == "normal":
        return fit_normal(data)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'nig', 'student_t', or 'normal'.")


# =============================================================================
# SELECTION AND COMPARISON FUNCTIONS
# =============================================================================

def select_best_distribution(
    data: np.ndarray,
    candidates: Sequence[str] = ("normal", "student_t", "nig"),
    metric: Literal["crps", "log_score"] = "crps",
    test_fraction: float = 0.2,
    n_samples: int = 2000,
) -> dict:
    """
    Automatically select the best distribution for given data.

    Uses proper scoring rules to objectively compare distributions
    on held-out test data.

    Args:
        data: Array of observations (e.g., returns)
        candidates: Distribution names to compare
        metric: Scoring metric ('crps' or 'log_score')
        test_fraction: Fraction of data to use for testing (0 = use all for fitting)
        n_samples: MC samples for CRPS calculation

    Returns:
        dict with keys:
            'best': name of best distribution
            'scores': dict of {name: score} for all candidates
            'params': dict of {name: fitted_params} for all candidates
            'confidence': how confident we are (based on score gaps)

    Example:
        >>> result = tpdf.select_best_distribution(returns)
        >>> params = tpdf.NIGParameters(**result['params']['nig'])
    """
    data = np.asarray(data)
    n = len(data)

    if test_fraction > 0:
        split = int(n * (1 - test_fraction))
        train_data = data[:split]
        test_data = data[split:]
    else:
        train_data = data
        test_data = data  # Score on same data if no test fraction

    scores = {}
    params = {}

    nig = NIGDistribution()

    for dist_name in candidates:
        if dist_name == "normal":
            fitted = fit_normal(train_data)
            params["normal"] = {"mu": fitted.mu_0, "sigma": fitted.sigma_0}

            if metric == "crps":
                scores["normal"] = np.mean([crps_normal(y, fitted.mu_0, fitted.sigma_0) for y in test_data])
            else:
                scores["normal"] = np.mean([
                    -np.log(max(stats.norm.pdf(y, fitted.mu_0, fitted.sigma_0), 1e-300))
                    for y in test_data
                ])

        elif dist_name == "student_t":
            fitted = fit_student_t(train_data)
            params["student_t"] = {"nu": fitted.nu, "mu_0": fitted.mu_0, "sigma_0": fitted.sigma_0}

            if metric == "crps":
                crps_vals = []
                for y in test_data:
                    samples = stats.t.rvs(fitted.nu, fitted.mu_0, fitted.sigma_0, size=n_samples)
                    crps_vals.append(
                        np.mean(np.abs(samples - y)) -
                        0.5 * np.mean(np.abs(samples[:n_samples//2] - samples[n_samples//2:]))
                    )
                scores["student_t"] = np.mean(crps_vals)
            else:
                scores["student_t"] = np.mean([
                    -np.log(max(stats.t.pdf(y, fitted.nu, fitted.mu_0, fitted.sigma_0), 1e-300))
                    for y in test_data
                ])

        elif dist_name == "nig":
            fitted = fit_nig(train_data)
            params["nig"] = {"mu": fitted.mu, "delta": fitted.delta, "alpha": fitted.alpha, "beta": fitted.beta}

            if metric == "crps":
                scores["nig"] = np.mean([
                    crps(nig, fitted, y, t=0.0, n_samples=n_samples)
                    for y in test_data
                ])
            else:
                scores["nig"] = np.mean([
                    -np.log(max(nig.pdf(np.array([y]), 0, fitted)[0], 1e-300))
                    for y in test_data
                ])

    # Determine best
    best = min(scores.keys(), key=lambda k: scores[k])

    # Calculate confidence based on score gaps
    sorted_scores = sorted(scores.values())
    if len(sorted_scores) >= 2:
        gap = (sorted_scores[1] - sorted_scores[0]) / sorted_scores[0]
        if gap > 0.1:
            confidence = "high"
        elif gap > 0.03:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "n/a"

    return {
        "best": best,
        "scores": scores,
        "params": params,
        "confidence": confidence,
    }


def compare_distributions(
    data: np.ndarray,
    distributions: Sequence[str] = ("normal", "student_t", "nig"),
    n_folds: int = 5,
    metric: Literal["crps", "log_score"] = "crps",
) -> dict:
    """
    Compare distributions using cross-validation.

    Args:
        data: Array of observations
        distributions: Distribution names to compare
        n_folds: Number of CV folds
        metric: Scoring metric

    Returns:
        dict with:
            'mean_scores': {dist: mean_score}
            'std_scores': {dist: std_score}
            'winner': name of best distribution
            'significant': bool, whether winner is statistically significant

    Example:
        >>> result = tpdf.compare_distributions(returns)
        >>> print(f"Winner: {result['winner']} (significant: {result['significant']})")
    """
    data = np.asarray(data)
    n = len(data)
    fold_size = n // n_folds

    fold_scores = {dist: [] for dist in distributions}

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        test_data = data[test_start:test_end]
        train_data = np.concatenate([data[:test_start], data[test_end:]])

        result = select_best_distribution(
            np.concatenate([train_data, test_data]),
            candidates=distributions,
            metric=metric,
            test_fraction=len(test_data) / len(data),
        )

        for dist in distributions:
            fold_scores[dist].append(result["scores"].get(dist, np.nan))

    mean_scores = {dist: np.mean(fold_scores[dist]) for dist in distributions}
    std_scores = {dist: np.std(fold_scores[dist]) for dist in distributions}
    winner = min(mean_scores.keys(), key=lambda k: mean_scores[k])

    # Statistical significance via paired t-test
    if len(distributions) >= 2:
        runner_up = sorted(mean_scores.keys(), key=lambda k: mean_scores[k])[1]
        _, pvalue = stats.ttest_rel(fold_scores[winner], fold_scores[runner_up])
        significant = pvalue < 0.05
    else:
        significant = False

    return {
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "winner": winner,
        "significant": significant,
        "fold_scores": fold_scores,
    }


# =============================================================================
# BACKTESTING FUNCTIONS
# =============================================================================

def rolling_var_backtest(
    data: np.ndarray,
    distribution: Literal["normal", "student_t", "nig", "historical"] = "nig",
    lookback: int = 252,
    alpha: float = 0.05,
) -> dict:
    """
    Run a rolling VaR backtest on historical data.

    At each time step, fits the distribution on the lookback window
    and calculates VaR. Records whether actual return exceeded VaR.

    Args:
        data: Array of returns
        distribution: Which distribution to use
        lookback: Number of days to use for fitting
        alpha: VaR confidence level (0.05 = 95% VaR)

    Returns:
        dict with keys:
            'var_forecasts': array of VaR forecasts
            'actual_returns': array of actual returns
            'exceedances': boolean array of VaR exceedances
            'exceedance_rate': fraction of exceedances
            'kupiec_pvalue': p-value from Kupiec test
            'status': 'PASS' or 'FAIL' based on calibration

    Example:
        >>> result = tpdf.rolling_var_backtest(returns, distribution='nig')
        >>> print(f"Exceedance rate: {result['exceedance_rate']:.1%}")
    """
    data = np.asarray(data)
    n = len(data)
    nig = NIGDistribution()

    var_forecasts = []
    actual_returns = []
    exceedances = []

    for i in range(lookback, n):
        window = data[i-lookback:i]
        actual = data[i]

        if distribution == "historical":
            var_forecast = -np.percentile(window, alpha * 100)

        elif distribution == "normal":
            params = fit_normal(window)
            var_forecast = -(params.mu_0 - stats.norm.ppf(1 - alpha) * params.sigma_0)

        elif distribution == "student_t":
            params = fit_student_t(window)
            var_forecast = -stats.t.ppf(alpha, params.nu, params.mu_0, params.sigma_0)

        elif distribution == "nig":
            params = fit_nig(window)
            var_forecast = var(nig, params, alpha=alpha, t=0.0)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        var_forecasts.append(var_forecast)
        actual_returns.append(actual)
        exceedances.append(-actual > var_forecast)

    var_forecasts = np.array(var_forecasts)
    actual_returns = np.array(actual_returns)
    exceedances = np.array(exceedances)

    exceedance_rate = np.mean(exceedances)
    n_exc = np.sum(exceedances)
    n_total = len(exceedances)

    # Kupiec test
    if 0 < n_exc < n_total:
        p_hat = n_exc / n_total
        lr = -2 * (
            n_total * np.log(1 - alpha) -
            (n_exc * np.log(p_hat) + (n_total - n_exc) * np.log(1 - p_hat))
        )
        kupiec_pvalue = 1 - stats.chi2.cdf(abs(lr), 1)
    else:
        kupiec_pvalue = 0.0

    # Determine status
    if 0.03 <= exceedance_rate <= 0.07:
        status = "PASS"
    elif exceedance_rate > 0.07:
        status = "FAIL_HIGH"
    else:
        status = "FAIL_LOW"

    return {
        "var_forecasts": var_forecasts,
        "actual_returns": actual_returns,
        "exceedances": exceedances,
        "exceedance_rate": exceedance_rate,
        "n_exceedances": n_exc,
        "n_total": n_total,
        "kupiec_pvalue": kupiec_pvalue,
        "status": status,
    }


# =============================================================================
# BARRIER PROBABILITY UTILITIES
# =============================================================================

def barrier_prob_normal(
    mu: float,
    sigma: float,
    horizon: int,
    barrier: float,
) -> float:
    """
    Analytical barrier probability for random walk with Normal increments.

    Uses reflection principle - exact for Brownian motion with drift.
    Computes P(max_{t=1..T} S_t >= barrier) where S_t is cumulative sum.

    Args:
        mu: Mean of daily returns
        sigma: Std dev of daily returns
        horizon: Number of time steps
        barrier: Cumulative return threshold (e.g., 0.05 for 5%)

    Returns:
        P(max cumulative sum >= barrier)

    Example:
        >>> p = barrier_prob_normal(mu=0.001, sigma=0.02, horizon=10, barrier=0.05)
        >>> print(f"P(hit 5% in 10 days): {p:.1%}")
    """
    drift = mu * horizon
    vol = sigma * np.sqrt(horizon)

    if vol < 1e-10:
        return 1.0 if drift >= barrier else 0.0

    # Standard first passage approximation
    d1 = (barrier - drift) / vol

    # Base probability
    p = stats.norm.cdf(-d1)

    # Correction for positive drift (reflection principle)
    if mu > 1e-10 and sigma > 1e-10:
        d2 = (barrier + drift) / vol
        # Guard against overflow in exponential term
        exp_arg = 2 * mu * barrier / (sigma**2)
        if exp_arg < 700:  # exp(700) is close to float max
            p += np.exp(exp_arg) * stats.norm.cdf(-d2)
        # If exp_arg >= 700, the correction term is essentially 0 anyway
        # because cdf(-d2) will be tiny for such cases

    return float(np.clip(p, 0, 1))


def barrier_prob_student_t(
    mu: float,
    sigma: float,
    nu: float,
    horizon: int,
    barrier: float,
) -> float:
    """
    Approximate barrier probability for Student-t distribution.

    Uses Normal analytical formula with inflated volatility to account
    for fat tails. The inflation factor is sqrt(nu/(nu-2)) which is the
    ratio of Student-t std dev to scale parameter.

    Args:
        mu: Location parameter
        sigma: Scale parameter
        nu: Degrees of freedom (tail heaviness). Lower = fatter tails.
        horizon: Number of time steps
        barrier: Cumulative return threshold

    Returns:
        Approximate P(max cumulative sum >= barrier)

    Example:
        >>> p = barrier_prob_student_t(mu=0.001, sigma=0.02, nu=5, horizon=10, barrier=0.05)
    """
    # Inflate sigma to account for fat tails
    if nu > 2:
        # Student-t variance = sigma^2 * nu/(nu-2)
        tail_factor = np.sqrt(nu / (nu - 2))
    else:
        # Infinite variance case - use conservative estimate
        tail_factor = 3.0

    effective_sigma = sigma * tail_factor
    return barrier_prob_normal(mu, effective_sigma, horizon, barrier)


def barrier_prob_nig(
    mu: float,
    delta: float,
    alpha: float,
    beta: float,
    horizon: int,
    barrier: float,
) -> float:
    """
    Approximate barrier probability for NIG distribution.

    NIG has semi-heavy tails. Uses adjusted Normal approximation based on
    NIG variance and kurtosis.

    Args:
        mu: Location parameter
        delta: Scale parameter
        alpha: Tail heaviness (larger = lighter tails)
        beta: Skewness parameter
        horizon: Number of time steps
        barrier: Cumulative return threshold

    Returns:
        Approximate P(max cumulative sum >= barrier)
    """
    # NIG variance = delta / gamma^3 where gamma = sqrt(alpha^2 - beta^2)
    gamma = np.sqrt(alpha**2 - beta**2 + 1e-10)
    nig_variance = delta / (gamma**3)
    effective_sigma = np.sqrt(nig_variance)

    # NIG excess kurtosis = 3 * (1 + 4*beta^2/gamma^2) / (delta * gamma)
    # Use this to estimate effective degrees of freedom
    nig_kurtosis = 3 * (1 + 4 * beta**2 / gamma**2) / (delta * gamma + 1e-10)
    # Map kurtosis to approximate nu (nu=4 has kurtosis=inf, nu=5 has kurtosis=6)
    effective_nu = max(4.0, 6.0 / (nig_kurtosis + 1e-10) + 4.0)

    return barrier_prob_student_t(mu, effective_sigma, effective_nu, horizon, barrier)


def barrier_prob_mc(
    params: Union[StudentTParameters, NormalParameters, NIGParameters],
    horizon: int,
    barrier: float,
    n_sims: int = 10000,
    distribution: Literal["normal", "student_t", "nig"] = "student_t",
) -> float:
    """
    Monte Carlo barrier probability estimation.

    Standard simulation approach - sample paths and count barrier hits.

    Args:
        params: Distribution parameters
        horizon: Number of time steps
        barrier: Cumulative return threshold
        n_sims: Number of simulation paths
        distribution: Distribution type

    Returns:
        Estimated P(max cumulative sum >= barrier)
    """
    if distribution == "normal":
        mu = params.mu_0 if hasattr(params, 'mu_0') else params.mu
        sigma = params.sigma_0 if hasattr(params, 'sigma_0') else params.sigma
        samples = np.random.normal(mu, sigma, size=(n_sims, horizon))

    elif distribution == "student_t":
        mu = params.mu_0
        sigma = params.sigma_0
        nu = params.nu
        samples = mu + sigma * np.random.standard_t(nu, size=(n_sims, horizon))

    elif distribution == "nig":
        # NIG sampling via inverse Gaussian mixture
        gamma = np.sqrt(params.alpha**2 - params.beta**2)
        # Inverse Gaussian samples
        ig_samples = stats.invgauss.rvs(
            mu=params.delta / gamma,
            scale=params.delta**2,
            size=(n_sims, horizon)
        )
        normal_samples = np.random.randn(n_sims, horizon)
        samples = params.mu + params.beta * ig_samples + np.sqrt(ig_samples) * normal_samples

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Compute cumulative sums and check barrier
    cumsum = np.cumsum(samples, axis=1)
    max_cumsum = np.max(cumsum, axis=1)

    return float(np.mean(max_cumsum >= barrier))


def barrier_prob_importance_sampling(
    params: Union[StudentTParameters, NormalParameters],
    horizon: int,
    barrier: float,
    n_sims: int = 10000,
    distribution: Literal["normal", "student_t"] = "normal",
) -> float:
    """
    Barrier probability with importance sampling for rare events.

    Tilts the distribution toward barrier-hitting paths, then corrects
    with likelihood ratios. Much lower variance for rare events.

    NOTE: The exponential tilting approach is exact for Normal distribution.
    For Student-t, the likelihood ratio is approximate and results may be
    biased. Use barrier_prob_qmc() for Student-t distributions.

    Args:
        params: Distribution parameters
        horizon: Number of time steps
        barrier: Cumulative return threshold
        n_sims: Number of simulation paths
        distribution: Distribution type. "normal" recommended (exact IS).
                      "student_t" uses approximate IS (may be biased).

    Returns:
        Importance-weighted barrier probability estimate

    Example:
        >>> # For Normal distribution, IS gives lower variance
        >>> params = tpdf.NormalParameters(mu_0=0.001, sigma_0=0.02)
        >>> p_is = barrier_prob_importance_sampling(params, horizon=10, barrier=0.10)
    """
    if distribution == "normal":
        mu = params.mu_0 if hasattr(params, 'mu_0') else params.mu
        sigma = params.sigma_0 if hasattr(params, 'sigma_0') else params.sigma
        nu = None
    else:  # student_t
        mu = params.mu_0
        sigma = params.sigma_0
        nu = params.nu

    # Optimal drift shift toward barrier (exponential tilting)
    optimal_shift = barrier / (horizon * sigma**2)
    tilted_mu = mu + optimal_shift * sigma**2

    # Sample from tilted distribution
    if distribution == "normal":
        samples = np.random.normal(tilted_mu, sigma, size=(n_sims, horizon))
    else:
        samples = tilted_mu + sigma * np.random.standard_t(nu, size=(n_sims, horizon))

    # Compute paths
    cumsum = np.cumsum(samples, axis=1)
    max_cumsum = np.max(cumsum, axis=1)
    hits = max_cumsum >= barrier

    # Likelihood ratio correction
    # For normal: log(p_orig / p_tilted) = -shift * sum(X) + 0.5 * shift^2 * sigma^2 * T + shift * mu * T
    path_sums = np.sum(samples, axis=1)
    log_ratio = (
        -optimal_shift * path_sums +
        optimal_shift * mu * horizon +
        0.5 * optimal_shift**2 * sigma**2 * horizon
    )
    weights = np.exp(log_ratio)

    # Self-normalized importance sampling estimator
    return float(np.sum(hits * weights) / np.sum(weights))


def barrier_prob_qmc(
    params: Union[StudentTParameters, NormalParameters],
    horizon: int,
    barrier: float,
    n_sims: int = 1024,
    distribution: Literal["normal", "student_t"] = "student_t",
) -> float:
    """
    Barrier probability using Quasi-Monte Carlo (Sobol sequences).

    QMC fills the sample space more uniformly than pseudo-random,
    giving lower variance with the same number of samples.

    Args:
        params: Distribution parameters
        horizon: Number of time steps
        barrier: Cumulative return threshold
        n_sims: Number of paths (should be power of 2 for Sobol)
        distribution: Distribution type

    Returns:
        QMC estimate of P(max cumulative sum >= barrier)

    Note:
        n_sims should be a power of 2 for optimal Sobol sequence properties.
    """
    from scipy.stats import qmc

    # Generate Sobol sequence
    sampler = qmc.Sobol(d=horizon, scramble=True)
    uniform_samples = sampler.random(n=n_sims)  # (n_sims, horizon) in [0,1]

    # Transform to distribution
    if distribution == "normal":
        mu = params.mu_0 if hasattr(params, 'mu_0') else params.mu
        sigma = params.sigma_0 if hasattr(params, 'sigma_0') else params.sigma
        z = stats.norm.ppf(uniform_samples)
        samples = mu + sigma * z

    elif distribution == "student_t":
        mu = params.mu_0
        sigma = params.sigma_0
        nu = params.nu
        z = stats.t.ppf(uniform_samples, df=nu)
        samples = mu + sigma * z

    else:
        raise ValueError(f"QMC not implemented for {distribution}")

    # Compute paths
    cumsum = np.cumsum(samples, axis=1)
    max_cumsum = np.max(cumsum, axis=1)

    return float(np.mean(max_cumsum >= barrier))
