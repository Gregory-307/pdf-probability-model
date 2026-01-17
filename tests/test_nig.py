"""Tests for the Normal Inverse Gaussian distribution."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


class TestNIGParameters:
    """Test NIGParameters validation."""

    def test_valid_parameters(self):
        """Test that valid parameters are accepted."""
        params = tpdf.NIGParameters(mu=0.0, delta=1.0, alpha=2.0, beta=0.5)
        assert params.mu == 0.0
        assert params.delta == 1.0
        assert params.alpha == 2.0
        assert params.beta == 0.5

    def test_invalid_delta_raises(self):
        """Test that delta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="delta must be positive"):
            tpdf.NIGParameters(mu=0.0, delta=0.0, alpha=2.0, beta=0.5)

        with pytest.raises(ValueError, match="delta must be positive"):
            tpdf.NIGParameters(mu=0.0, delta=-1.0, alpha=2.0, beta=0.5)

    def test_invalid_alpha_raises(self):
        """Test that alpha <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            tpdf.NIGParameters(mu=0.0, delta=1.0, alpha=0.0, beta=0.5)

    def test_invalid_beta_raises(self):
        """Test that |beta| >= alpha raises ValueError."""
        with pytest.raises(ValueError, match="beta"):
            tpdf.NIGParameters(mu=0.0, delta=1.0, alpha=2.0, beta=2.0)

        with pytest.raises(ValueError, match="beta"):
            tpdf.NIGParameters(mu=0.0, delta=1.0, alpha=2.0, beta=-2.5)


class TestNIGDistribution:
    """Test NIG distribution implementation."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=-2.0)

    def test_pdf_non_negative(self, nig, params):
        """Test that PDF is always non-negative."""
        x = np.linspace(-0.2, 0.2, 1000)
        pdf = nig.pdf(x, 0, params)
        assert np.all(pdf >= 0)

    def test_pdf_integrates_approximately_to_one(self, nig, params):
        """Test that PDF integrates to approximately 1."""
        x = np.linspace(-1, 1, 10000)
        pdf = nig.pdf(x, 0, params)
        integral = np.trapezoid(pdf, x)
        assert 0.95 < integral < 1.05

    def test_mean_calculation(self, nig, params):
        """Test mean calculation formula."""
        mean = nig.mean(0, params)
        # For NIG: E[X] = mu + delta * beta / gamma
        gamma = np.sqrt(params.alpha**2 - params.beta**2)
        expected = params.mu + params.delta * params.beta / gamma
        assert np.isclose(mean, expected)

    def test_variance_positive(self, nig, params):
        """Test that variance is positive."""
        var = nig.variance(0, params)
        assert var > 0

    def test_pdf_matrix_shape(self, nig, params):
        """Test that pdf_matrix returns correct shape."""
        x = np.linspace(-0.1, 0.1, 100)
        t = np.linspace(0, 10, 50)
        matrix = nig.pdf_matrix(x, t, params)
        assert matrix.shape == (50, 100)

    def test_time_evolution_increases_variance(self, nig):
        """Test that delta_growth increases spread over time."""
        params = tpdf.NIGParameters(
            mu=0.0, delta=0.02, alpha=15.0, beta=0.0, delta_growth=0.1
        )
        var_t0 = nig.variance(0, params)
        var_t10 = nig.variance(10, params)
        assert var_t10 > var_t0

    def test_sampling(self, nig, params):
        """Test that sampling produces reasonable values."""
        rng = np.random.default_rng(42)
        samples = nig.sample(10000, 0, params, rng)

        assert len(samples) == 10000
        # Samples should be centered around the mean
        sample_mean = np.mean(samples)
        theoretical_mean = nig.mean(0, params)
        assert abs(sample_mean - theoretical_mean) < 0.01


class TestNIGCDF:
    """Test NIG CDF and quantile functions."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    @pytest.fixture
    def params(self):
        return tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=0.0)

    def test_cdf_monotonic(self, nig, params):
        """Test that CDF is monotonically increasing."""
        x = np.linspace(-0.1, 0.1, 100)
        cdf = nig.cdf(x, 0, params)
        assert np.all(np.diff(cdf) >= -1e-10)  # Allow small numerical errors

    def test_cdf_bounds(self, nig, params):
        """Test that CDF is between 0 and 1."""
        x = np.linspace(-0.5, 0.5, 100)
        cdf = nig.cdf(x, 0, params)
        assert np.all(cdf >= 0)
        assert np.all(cdf <= 1)

    def test_ppf_inverts_cdf(self, nig, params):
        """Test that PPF inverts CDF."""
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        x = nig.ppf(q, 0, params)
        q_back = nig.cdf(x, 0, params)
        np.testing.assert_allclose(q, q_back, rtol=0.1)
