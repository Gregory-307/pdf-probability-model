"""Tests for decision utilities (VaR, CVaR, Kelly)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


class TestVaR:
    """Test Value at Risk implementation."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=0.0)

    def test_var_positive_for_risky_distribution(self, nig, params):
        """Test that VaR is positive for distributions with downside risk."""
        var_95 = tpdf.var(nig, params, alpha=0.05)
        assert var_95 > 0

    def test_var_increases_with_confidence(self, nig, params):
        """Test that VaR increases with confidence level."""
        var_90 = tpdf.var(nig, params, alpha=0.10)  # 90% confidence
        var_95 = tpdf.var(nig, params, alpha=0.05)  # 95% confidence
        var_99 = tpdf.var(nig, params, alpha=0.01)  # 99% confidence
        assert var_90 < var_95 < var_99

    def test_var_invalid_alpha_raises(self, nig, params):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError):
            tpdf.var(nig, params, alpha=0)
        with pytest.raises(ValueError):
            tpdf.var(nig, params, alpha=1)
        with pytest.raises(ValueError):
            tpdf.var(nig, params, alpha=-0.1)


class TestCVaR:
    """Test Conditional Value at Risk implementation."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=0.0)

    def test_cvar_greater_than_var(self, nig, params):
        """Test that CVaR >= VaR (expected shortfall >= threshold)."""
        rng = np.random.default_rng(42)
        var_95 = tpdf.var(nig, params, alpha=0.05)
        cvar_95 = tpdf.cvar(nig, params, alpha=0.05, rng=rng)
        assert cvar_95 >= var_95 * 0.9  # Allow some Monte Carlo error

    def test_cvar_positive(self, nig, params):
        """Test that CVaR is positive for risky distribution."""
        rng = np.random.default_rng(42)
        cvar_95 = tpdf.cvar(nig, params, alpha=0.05, rng=rng)
        assert cvar_95 > 0

    def test_cvar_increases_with_confidence(self, nig, params):
        """Test that CVaR increases with confidence level."""
        rng = np.random.default_rng(42)
        cvar_90 = tpdf.cvar(nig, params, alpha=0.10, rng=rng)
        cvar_99 = tpdf.cvar(nig, params, alpha=0.01, rng=rng)
        assert cvar_99 > cvar_90


class TestKelly:
    """Test Kelly criterion implementation."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    def test_kelly_positive_for_positive_mean(self):
        """Test Kelly is positive when expected return is positive."""
        nig = tpdf.NIG()
        # Positive expected return
        params = tpdf.NIGParameters(mu=0.01, delta=0.02, alpha=15.0, beta=0.0)
        kelly = tpdf.kelly_fraction(nig, params)
        assert kelly > 0

    def test_kelly_negative_for_negative_mean(self):
        """Test Kelly is negative when expected return is negative."""
        nig = tpdf.NIG()
        # Negative expected return
        params = tpdf.NIGParameters(mu=-0.01, delta=0.02, alpha=15.0, beta=0.0)
        kelly = tpdf.kelly_fraction(nig, params)
        assert kelly < 0

    def test_kelly_increases_with_sharpe(self):
        """Test Kelly increases with Sharpe ratio."""
        nig = tpdf.NIG()

        # Lower Sharpe (same mean, higher variance)
        params_low = tpdf.NIGParameters(mu=0.01, delta=0.04, alpha=15.0, beta=0.0)
        # Higher Sharpe (same mean, lower variance)
        params_high = tpdf.NIGParameters(mu=0.01, delta=0.01, alpha=15.0, beta=0.0)

        kelly_low = tpdf.kelly_fraction(nig, params_low)
        kelly_high = tpdf.kelly_fraction(nig, params_high)

        assert kelly_high > kelly_low

    def test_fractional_kelly_less_than_full(self):
        """Test that fractional Kelly is less than full Kelly."""
        nig = tpdf.NIG()
        params = tpdf.NIGParameters(mu=0.01, delta=0.02, alpha=15.0, beta=0.0)

        full_kelly = tpdf.kelly_fraction(nig, params)
        half_kelly = tpdf.fractional_kelly(nig, params, fraction=0.5)

        assert np.isclose(half_kelly, 0.5 * full_kelly)


class TestProbabilityQueries:
    """Test probability query functions."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        # Symmetric distribution centered at 0
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=0.0)

    def test_prob_greater_than_half_at_mean(self, nig, params):
        """Test P(X > mean) ≈ 0.5 for symmetric distribution."""
        prob = tpdf.prob_greater_than(nig, params, threshold=0.0)
        assert 0.45 < prob < 0.55

    def test_prob_less_than_half_at_mean(self, nig, params):
        """Test P(X < mean) ≈ 0.5 for symmetric distribution."""
        prob = tpdf.prob_less_than(nig, params, threshold=0.0)
        assert 0.45 < prob < 0.55

    def test_prob_between_bounds(self, nig, params):
        """Test P(a < X < b) properties."""
        prob = tpdf.prob_between(nig, params, lower=-0.1, upper=0.1)
        assert 0 < prob < 1

        # Wider interval should have higher probability
        prob_wide = tpdf.prob_between(nig, params, lower=-0.2, upper=0.2)
        prob_narrow = tpdf.prob_between(nig, params, lower=-0.05, upper=0.05)
        assert prob_wide > prob_narrow

    def test_prob_between_invalid_bounds_raises(self, nig, params):
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError):
            tpdf.prob_between(nig, params, lower=0.1, upper=-0.1)


class TestVaRWithCI:
    """Test VaR with confidence intervals (V2 API)."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=0.0)

    def test_returns_risk_metric(self, nig, params):
        """Test that var_with_ci returns RiskMetric."""
        result = tpdf.var_with_ci(nig, params, alpha=0.05)
        assert isinstance(result, tpdf.RiskMetric)

    def test_has_confidence_interval(self, nig, params):
        """Test that result has confidence interval."""
        result = tpdf.var_with_ci(nig, params, alpha=0.05)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2

    def test_ci_contains_point_estimate(self, nig, params):
        """Test that CI contains point estimate."""
        result = tpdf.var_with_ci(nig, params, alpha=0.05)
        lower, upper = result.confidence_interval
        assert lower <= result.value <= upper

    def test_float_conversion(self, nig, params):
        """Test that RiskMetric converts to float."""
        result = tpdf.var_with_ci(nig, params, alpha=0.05)
        assert float(result) == result.value

    def test_value_matches_var(self, nig, params):
        """Test that value is in same order of magnitude as regular var function."""
        result_ci = tpdf.var_with_ci(nig, params, alpha=0.05)
        result_plain = tpdf.var(nig, params, alpha=0.05)
        # Both should be positive VaR values in similar range
        # (exact match not expected due to different methods: bootstrap vs quantile)
        assert result_ci.value > 0
        assert result_plain > 0
        assert 0.5 < result_ci.value / result_plain < 2.0  # Within 2x


class TestCVaRWithCI:
    """Test CVaR with confidence intervals (V2 API)."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=0.0)

    def test_returns_risk_metric(self, nig, params):
        """Test that cvar_with_ci returns RiskMetric."""
        rng = np.random.default_rng(42)
        result = tpdf.cvar_with_ci(nig, params, alpha=0.05, rng=rng)
        assert isinstance(result, tpdf.RiskMetric)

    def test_has_confidence_interval(self, nig, params):
        """Test that result has confidence interval."""
        rng = np.random.default_rng(42)
        result = tpdf.cvar_with_ci(nig, params, alpha=0.05, rng=rng)
        assert result.confidence_interval is not None

    def test_cvar_greater_than_var(self, nig, params):
        """Test that CVaR >= VaR."""
        rng = np.random.default_rng(42)
        var_result = tpdf.var_with_ci(nig, params, alpha=0.05)
        cvar_result = tpdf.cvar_with_ci(nig, params, alpha=0.05, rng=rng)
        assert cvar_result.value >= var_result.value * 0.9  # Allow Monte Carlo error


class TestKellyWithCI:
    """Test Kelly fraction with confidence intervals (V2 API)."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    def test_returns_risk_metric(self):
        """Test that kelly_with_ci returns RiskMetric."""
        nig = tpdf.NIG()
        params = tpdf.NIGParameters(mu=0.01, delta=0.02, alpha=15.0, beta=0.0)
        rng = np.random.default_rng(42)
        result = tpdf.kelly_with_ci(nig, params, rng=rng)
        assert isinstance(result, tpdf.RiskMetric)

    def test_has_confidence_interval(self):
        """Test that result has confidence interval."""
        nig = tpdf.NIG()
        params = tpdf.NIGParameters(mu=0.01, delta=0.02, alpha=15.0, beta=0.0)
        rng = np.random.default_rng(42)
        result = tpdf.kelly_with_ci(nig, params, rng=rng)
        assert result.confidence_interval is not None

    def test_positive_for_positive_mean(self):
        """Test Kelly is positive when mean is positive."""
        nig = tpdf.NIG()
        params = tpdf.NIGParameters(mu=0.01, delta=0.02, alpha=15.0, beta=0.0)
        rng = np.random.default_rng(42)
        result = tpdf.kelly_with_ci(nig, params, rng=rng)
        assert result.value > 0
