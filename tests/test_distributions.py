"""Tests for distribution implementations."""

import numpy as np
import pytest

from temporalpdf import (
    Normal,
    NormalParameters,
    StudentT,
    StudentTParameters,
    SkewNormal,
    SkewNormalParameters,
    GeneralizedLaplace,
    GeneralizedLaplaceParameters,
    DistributionRegistry,
)


class TestNormalDistribution:
    """Tests for NormalDistribution."""

    def test_pdf_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        dist = Normal()
        params = NormalParameters(mu_0=0.0, sigma_0=0.05)
        x = np.linspace(-0.5, 0.5, 1000)

        pdf = dist.pdf(x, t=0.0, params=params)
        integral = np.trapezoid(pdf, x)

        assert 0.99 < integral < 1.01

    def test_pdf_centered_at_mu(self):
        """PDF maximum should be at mu_0 when t=0."""
        dist = Normal()
        params = NormalParameters(mu_0=0.1, sigma_0=0.05)
        x = np.linspace(-0.5, 0.5, 1000)

        pdf = dist.pdf(x, t=0.0, params=params)
        max_idx = np.argmax(pdf)

        assert abs(x[max_idx] - 0.1) < 0.01

    def test_mean_drift(self):
        """Mean should drift over time according to delta."""
        dist = Normal()
        params = NormalParameters(mu_0=0.0, sigma_0=0.05, delta=0.01)
        x = np.linspace(-0.5, 1.0, 1000)

        # At t=10, mean should be at mu_0 + delta*t = 0.1
        pdf = dist.pdf(x, t=10.0, params=params)
        max_idx = np.argmax(pdf)

        assert abs(x[max_idx] - 0.1) < 0.02

    def test_volatility_growth(self):
        """Volatility should grow over time according to beta."""
        dist = Normal()
        params_no_growth = NormalParameters(mu_0=0.0, sigma_0=0.05, beta=0.0)
        params_with_growth = NormalParameters(mu_0=0.0, sigma_0=0.05, beta=0.1)
        x = np.linspace(-0.5, 0.5, 1000)

        pdf_no_growth = dist.pdf(x, t=10.0, params=params_no_growth)
        pdf_with_growth = dist.pdf(x, t=10.0, params=params_with_growth)

        # PDF with volatility growth should be flatter (lower peak)
        assert np.max(pdf_with_growth) < np.max(pdf_no_growth)

    def test_pdf_matrix_shape(self):
        """pdf_matrix should return correct shape."""
        dist = Normal()
        params = NormalParameters(mu_0=0.0, sigma_0=0.05)
        x = np.linspace(-0.5, 0.5, 100)
        t = np.linspace(0, 60, 50)

        matrix = dist.pdf_matrix(x, t, params)

        assert matrix.shape == (50, 100)


class TestStudentTDistribution:
    """Tests for StudentTDistribution."""

    def test_pdf_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        dist = StudentT()
        params = StudentTParameters(mu_0=0.0, sigma_0=0.05, nu=5.0)
        x = np.linspace(-1.0, 1.0, 2000)

        pdf = dist.pdf(x, t=0.0, params=params)
        integral = np.trapezoid(pdf, x)

        assert 0.95 < integral < 1.05

    def test_heavier_tails_than_normal(self):
        """Student-t should have heavier tails than Normal."""
        dist_t = StudentT()
        dist_n = Normal()
        params_t = StudentTParameters(mu_0=0.0, sigma_0=0.05, nu=3.0)
        params_n = NormalParameters(mu_0=0.0, sigma_0=0.05)
        x = np.linspace(-0.5, 0.5, 1000)

        pdf_t = dist_t.pdf(x, t=0.0, params=params_t)
        pdf_n = dist_n.pdf(x, t=0.0, params=params_n)

        # At tails (far from center), Student-t should be higher
        tail_idx = 50  # Far from center
        assert pdf_t[tail_idx] > pdf_n[tail_idx]

    def test_invalid_nu_raises(self):
        """nu <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            StudentTParameters(mu_0=0.0, sigma_0=0.05, nu=0.0)
        with pytest.raises(ValueError):
            StudentTParameters(mu_0=0.0, sigma_0=0.05, nu=-1.0)


class TestSkewNormalDistribution:
    """Tests for SkewNormalDistribution."""

    def test_pdf_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        dist = SkewNormal()
        params = SkewNormalParameters(mu_0=0.0, sigma_0=0.05, alpha=2.0)
        x = np.linspace(-0.5, 0.5, 1000)

        pdf = dist.pdf(x, t=0.0, params=params)
        integral = np.trapezoid(pdf, x)

        assert 0.99 < integral < 1.01

    def test_alpha_zero_symmetric(self):
        """alpha=0 should give symmetric distribution."""
        dist = SkewNormal()
        params = SkewNormalParameters(mu_0=0.0, sigma_0=0.05, alpha=0.0)
        x = np.linspace(-0.3, 0.3, 1000)

        pdf = dist.pdf(x, t=0.0, params=params)

        # Should be approximately symmetric around 0
        mid = len(x) // 2
        left_sum = np.sum(pdf[:mid])
        right_sum = np.sum(pdf[mid:])
        assert abs(left_sum - right_sum) / left_sum < 0.05

    def test_positive_alpha_right_skew(self):
        """Positive alpha should skew distribution right."""
        dist = SkewNormal()
        params = SkewNormalParameters(mu_0=0.0, sigma_0=0.05, alpha=5.0)
        x = np.linspace(-0.3, 0.3, 1000)

        pdf = dist.pdf(x, t=0.0, params=params)

        # Right side should have more mass
        mid = len(x) // 2
        left_sum = np.sum(pdf[:mid])
        right_sum = np.sum(pdf[mid:])
        assert right_sum > left_sum


class TestGeneralizedLaplaceDistribution:
    """Tests for GeneralizedLaplaceDistribution."""

    def test_pdf_non_negative(self):
        """PDF should be non-negative everywhere."""
        dist = GeneralizedLaplace()
        params = GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=0.05, alpha=0.5, k=1.0
        )
        x = np.linspace(-0.5, 0.5, 1000)

        pdf = dist.pdf(x, t=0.0, params=params)

        assert np.all(pdf >= 0)

    def test_lambda_decay_reduces_magnitude(self):
        """Lambda decay should reduce PDF magnitude over time."""
        dist = GeneralizedLaplace()
        params_decay = GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=0.05, lambda_decay=0.1
        )
        params_no_decay = GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=0.05, lambda_decay=0.0
        )
        x = np.linspace(-0.5, 0.5, 1000)

        pdf_decay = dist.pdf(x, t=30.0, params=params_decay)
        pdf_no_decay = dist.pdf(x, t=30.0, params=params_no_decay)

        # With decay, total mass should be less
        assert np.sum(pdf_decay) < np.sum(pdf_no_decay)

    def test_invalid_sigma_raises(self):
        """sigma_0 <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            GeneralizedLaplaceParameters(mu_0=0.0, sigma_0=0.0)
        with pytest.raises(ValueError):
            GeneralizedLaplaceParameters(mu_0=0.0, sigma_0=-0.1)

    def test_invalid_k_raises(self):
        """k < 0 should raise ValueError."""
        with pytest.raises(ValueError):
            GeneralizedLaplaceParameters(mu_0=0.0, sigma_0=0.05, k=-1.0)


class TestDistributionRegistry:
    """Tests for DistributionRegistry."""

    def test_create_normal(self):
        """Registry should create Normal distribution."""
        dist = DistributionRegistry.create("normal")
        assert "Normal" in dist.name

    def test_create_student_t(self):
        """Registry should create Student-t distribution."""
        dist = DistributionRegistry.create("student_t")
        assert dist.name == "Student's t"

    def test_create_skew_normal(self):
        """Registry should create Skew-Normal distribution."""
        dist = DistributionRegistry.create("skew_normal")
        assert dist.name == "Skew-Normal"

    def test_create_generalized_laplace(self):
        """Registry should create Generalized Laplace distribution."""
        dist = DistributionRegistry.create("generalized_laplace")
        assert "Laplace" in dist.name

    def test_list_distributions(self):
        """list_available() should return all registered distributions."""
        available = DistributionRegistry.list_available()
        assert "normal" in available
        assert "student_t" in available
        assert "skew_normal" in available
        assert "generalized_laplace" in available

    def test_create_unknown_raises(self):
        """Creating unknown distribution should raise ValueError."""
        with pytest.raises(ValueError):
            DistributionRegistry.create("unknown_distribution")
