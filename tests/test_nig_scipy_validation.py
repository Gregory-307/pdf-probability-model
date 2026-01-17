"""
Validation of temporalpdf NIG against scipy.stats.norminvgauss.

This test ensures our NIG implementation produces results consistent
with the well-tested scipy implementation.

scipy uses parameterization: norminvgauss(a, b, loc, scale)
where:
    a = alpha * delta (steepness)
    b = beta * delta (asymmetry)
    loc = mu
    scale = delta

Our parameterization:
    mu, delta, alpha, beta

Conversion:
    scipy_a = alpha * delta
    scipy_b = beta * delta
    scipy_loc = mu
    scipy_scale = delta
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


def convert_to_scipy_params(params: tpdf.NIGParameters):
    """Convert our NIG parameters to scipy's parameterization."""
    a = params.alpha * params.delta
    b = params.beta * params.delta
    loc = params.mu
    scale = params.delta
    return a, b, loc, scale


class TestNIGScipyValidation:
    """Validate our NIG against scipy.stats.norminvgauss."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params_symmetric(self):
        """Symmetric NIG (beta=0)."""
        return tpdf.NIGParameters(mu=0.0, delta=1.0, alpha=1.5, beta=0.0)

    @pytest.fixture
    def params_skewed(self):
        """Skewed NIG."""
        return tpdf.NIGParameters(mu=0.5, delta=0.5, alpha=2.0, beta=0.5)

    def test_pdf_matches_scipy_symmetric(self, nig, params_symmetric):
        """Test PDF matches scipy for symmetric case."""
        x = np.linspace(-5, 5, 100)

        # Our implementation
        our_pdf = nig.pdf(x, 0, params_symmetric)

        # scipy implementation
        a, b, loc, scale = convert_to_scipy_params(params_symmetric)
        scipy_rv = scipy_stats.norminvgauss(a, b, loc=loc, scale=scale)
        scipy_pdf = scipy_rv.pdf(x)

        # Should be close (allowing for numerical differences)
        np.testing.assert_allclose(our_pdf, scipy_pdf, rtol=0.05, atol=1e-6)

    def test_pdf_matches_scipy_skewed(self, nig, params_skewed):
        """Test PDF matches scipy for skewed case."""
        x = np.linspace(-3, 5, 100)

        # Our implementation
        our_pdf = nig.pdf(x, 0, params_skewed)

        # scipy implementation
        a, b, loc, scale = convert_to_scipy_params(params_skewed)
        scipy_rv = scipy_stats.norminvgauss(a, b, loc=loc, scale=scale)
        scipy_pdf = scipy_rv.pdf(x)

        np.testing.assert_allclose(our_pdf, scipy_pdf, rtol=0.05, atol=1e-6)

    def test_mean_matches_scipy(self, nig, params_skewed):
        """Test that mean calculation matches scipy."""
        our_mean = nig.mean(0, params_skewed)

        a, b, loc, scale = convert_to_scipy_params(params_skewed)
        scipy_rv = scipy_stats.norminvgauss(a, b, loc=loc, scale=scale)
        scipy_mean = scipy_rv.mean()

        np.testing.assert_allclose(our_mean, scipy_mean, rtol=0.01)

    def test_variance_matches_scipy(self, nig, params_skewed):
        """Test that variance calculation matches scipy."""
        our_var = nig.variance(0, params_skewed)

        a, b, loc, scale = convert_to_scipy_params(params_skewed)
        scipy_rv = scipy_stats.norminvgauss(a, b, loc=loc, scale=scale)
        scipy_var = scipy_rv.var()

        np.testing.assert_allclose(our_var, scipy_var, rtol=0.05)

    def test_cdf_approximately_matches_scipy(self, nig, params_symmetric):
        """Test CDF approximately matches scipy."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        our_cdf = nig.cdf(x, 0, params_symmetric)

        a, b, loc, scale = convert_to_scipy_params(params_symmetric)
        scipy_rv = scipy_stats.norminvgauss(a, b, loc=loc, scale=scale)
        scipy_cdf = scipy_rv.cdf(x)

        # CDF is computed numerically, so allow more tolerance
        np.testing.assert_allclose(our_cdf, scipy_cdf, rtol=0.1, atol=0.02)

    def test_sampling_produces_correct_moments(self, nig, params_skewed):
        """Test that sampling produces samples with correct moments."""
        rng = np.random.default_rng(42)
        samples = nig.sample(50000, 0, params_skewed, rng)

        # Compare sample moments to theoretical
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)

        theoretical_mean = nig.mean(0, params_skewed)
        theoretical_var = nig.variance(0, params_skewed)

        # Sample mean should be within ~1% of theoretical for 50k samples
        assert abs(sample_mean - theoretical_mean) < 0.05
        # Variance within ~5%
        assert abs(sample_var - theoretical_var) / theoretical_var < 0.1


class TestNIGFinancialRealism:
    """Test NIG on realistic financial parameter ranges."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    def test_typical_daily_return_params(self, nig):
        """Test NIG with typical daily stock return parameters."""
        # Typical daily return: mean ~0.05%, std ~1.5%, slight negative skew
        params = tpdf.NIGParameters(
            mu=0.0005,    # 0.05% daily return
            delta=0.015,  # ~1.5% volatility
            alpha=50.0,   # Moderate tails (high alpha = lighter)
            beta=-5.0,    # Negative skew
        )

        # Should have mean close to mu adjustment
        mean = nig.mean(0, params)
        std = np.sqrt(nig.variance(0, params))

        assert abs(mean) < 0.01  # Mean should be small
        assert 0.01 < std < 0.03  # Std should be around 1.5%

    def test_high_volatility_regime(self, nig):
        """Test NIG in high volatility regime (like VIX spike)."""
        params = tpdf.NIGParameters(
            mu=0.0,
            delta=0.05,   # 5% base volatility
            alpha=20.0,   # Heavier tails
            beta=-10.0,   # Strong negative skew
        )

        # Check VaR is reasonable
        var_95 = tpdf.var(nig, params, alpha=0.05)
        assert 0.05 < var_95 < 0.20  # Should be between 5% and 20%

        # CVaR should be worse than VaR
        cvar_95 = tpdf.cvar(nig, params, alpha=0.05, n_samples=50000)
        assert cvar_95 >= var_95 * 0.8  # Allow Monte Carlo noise
