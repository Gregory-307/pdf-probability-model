"""Tests for proper scoring rules."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


class TestLogScore:
    """Test Log Score (negative log-likelihood)."""

    @pytest.fixture
    def normal(self):
        return tpdf.Normal()

    @pytest.fixture
    def params(self):
        return tpdf.NormalParameters(mu_0=0.0, sigma_0=1.0, delta=0.0, beta=0.0)

    def test_log_score_positive_for_low_density(self, normal, params):
        """Test that log score is large when observation is unlikely."""
        # Far from mean = low density = high (bad) log score
        score_far = tpdf.log_score(normal, params, y=5.0, t=0)
        score_near = tpdf.log_score(normal, params, y=0.0, t=0)
        assert score_far > score_near

    def test_log_score_minimum_at_mode(self, normal, params):
        """Test that log score is minimized at the mode."""
        scores = []
        for y in np.linspace(-2, 2, 100):
            scores.append(tpdf.log_score(normal, params, y, t=0))

        min_idx = np.argmin(scores)
        # Should be near the middle (at mu=0)
        assert 40 < min_idx < 60

    def test_log_score_array_input(self, normal, params):
        """Test that log score handles array input."""
        y = np.array([0.0, 0.5, 1.0])
        scores = tpdf.log_score(normal, params, y, t=0)
        assert len(scores) == 3


class TestCRPS:
    """Test Continuous Ranked Probability Score."""

    @pytest.fixture
    def nig(self):
        # Use NIG which has full interface (sample, ppf)
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        # Symmetric NIG centered at 0
        return tpdf.NIGParameters(mu=0.0, delta=0.05, alpha=15.0, beta=0.0)

    def test_crps_non_negative(self, nig, params):
        """Test that CRPS is always non-negative."""
        for y in [-0.1, 0, 0.1]:
            score = tpdf.crps(nig, params, y, t=0)
            assert score >= 0

    def test_crps_small_for_tight_distribution(self, nig):
        """Test that CRPS is small for tight distribution near observation."""
        # NIG with small delta = tight distribution
        params_tight = tpdf.NIGParameters(mu=0.0, delta=0.001, alpha=15.0, beta=0.0)
        score = tpdf.crps(nig, params_tight, y=0.0, t=0)
        assert score < 0.01

    def test_crps_increases_with_distance(self, nig, params):
        """Test that CRPS increases as observation moves from mean."""
        score_at_mean = tpdf.crps(nig, params, y=0.0, t=0)
        score_away = tpdf.crps(nig, params, y=0.2, t=0)
        assert score_away > score_at_mean


class TestCRPSNormal:
    """Test closed-form CRPS for Normal distribution."""

    def test_crps_normal_non_negative(self):
        """Test that CRPS is non-negative."""
        score = tpdf.crps_normal(y=1.0, mu=0.0, sigma=1.0)
        assert score >= 0

    def test_crps_normal_zero_at_mean(self):
        """Test CRPS properties at the mean."""
        score_at_mean = tpdf.crps_normal(y=0.0, mu=0.0, sigma=1.0)
        score_away = tpdf.crps_normal(y=2.0, mu=0.0, sigma=1.0)
        assert score_at_mean < score_away

    def test_crps_normal_scales_with_sigma(self):
        """Test that CRPS scales with sigma."""
        score_small_sigma = tpdf.crps_normal(y=0.0, mu=0.0, sigma=0.5)
        score_large_sigma = tpdf.crps_normal(y=0.0, mu=0.0, sigma=2.0)
        # Larger sigma = more uncertainty = higher CRPS
        assert score_large_sigma > score_small_sigma

    def test_crps_normal_array_input(self):
        """Test array input handling."""
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        scores = tpdf.crps_normal(y, mu, sigma)
        assert len(scores) == 3
        assert np.all(scores >= 0)
