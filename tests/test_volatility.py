"""Tests for volatility evolution models."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf
from temporalpdf.core.volatility import (
    LinearGrowth,
    ExponentialDecay,
    SquareRootDiffusion,
    GARCHForecast,
    TermStructure,
)


class TestLinearGrowth:
    """Tests for LinearGrowth volatility model."""

    def test_constant_volatility(self):
        """Test constant volatility (growth_rate=0)."""
        model = LinearGrowth(growth_rate=0.0)
        sigma_0 = 0.02

        assert model.at_time(sigma_0, 0) == sigma_0
        assert model.at_time(sigma_0, 10) == sigma_0
        assert model.at_time(sigma_0, 100) == sigma_0

    def test_linear_growth(self):
        """Test linear volatility growth."""
        model = LinearGrowth(growth_rate=0.05)  # 5% per time unit
        sigma_0 = 0.02

        assert model.at_time(sigma_0, 0) == sigma_0
        assert model.at_time(sigma_0, 1) == pytest.approx(sigma_0 * 1.05)
        assert model.at_time(sigma_0, 10) == pytest.approx(sigma_0 * 1.5)

    def test_vectorized(self):
        """Test vectorized computation."""
        model = LinearGrowth(growth_rate=0.05)
        sigma_0 = 0.02
        times = np.array([0, 5, 10])

        expected = sigma_0 * (1 + 0.05 * times)
        result = model.at_times(sigma_0, times)

        np.testing.assert_allclose(result, expected)


class TestExponentialDecay:
    """Tests for ExponentialDecay volatility model."""

    def test_mean_reversion(self):
        """Test mean-reversion behavior."""
        sigma_long = 0.02
        kappa = 0.1
        model = ExponentialDecay(sigma_long=sigma_long, kappa=kappa)

        sigma_0 = 0.04  # Elevated volatility

        # At t=0, should be sigma_0
        assert model.at_time(sigma_0, 0) == pytest.approx(sigma_0)

        # As t -> infinity, should approach sigma_long
        assert model.at_time(sigma_0, 100) == pytest.approx(sigma_long, rel=0.01)

        # Should decay monotonically
        t1 = model.at_time(sigma_0, 1)
        t5 = model.at_time(sigma_0, 5)
        t10 = model.at_time(sigma_0, 10)

        assert sigma_0 > t1 > t5 > t10 > sigma_long

    def test_below_long_run(self):
        """Test behavior when starting below long-run."""
        sigma_long = 0.02
        model = ExponentialDecay(sigma_long=sigma_long, kappa=0.1)

        sigma_0 = 0.01  # Below long-run

        # Should increase toward sigma_long
        assert model.at_time(sigma_0, 0) == pytest.approx(sigma_0)
        assert model.at_time(sigma_0, 50) > sigma_0
        assert model.at_time(sigma_0, 100) == pytest.approx(sigma_long, rel=0.01)

    def test_invalid_params(self):
        """Test validation of parameters."""
        with pytest.raises(ValueError, match="sigma_long must be positive"):
            ExponentialDecay(sigma_long=0, kappa=0.1)

        with pytest.raises(ValueError, match="kappa must be positive"):
            ExponentialDecay(sigma_long=0.02, kappa=0)


class TestGARCHForecast:
    """Tests for GARCHForecast volatility model."""

    def test_long_run_convergence(self):
        """Test convergence to long-run volatility."""
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        model = GARCHForecast(omega=omega, alpha=alpha, beta=beta)

        sigma_0 = 0.04  # Elevated volatility

        # Should converge to long-run vol
        long_run_vol = model.long_run_vol

        assert model.at_time(sigma_0, 0) == pytest.approx(sigma_0)
        assert model.at_time(sigma_0, 100) == pytest.approx(long_run_vol, rel=0.05)

    def test_persistence(self):
        """Test persistence property."""
        model = GARCHForecast(omega=0.00001, alpha=0.1, beta=0.85)

        assert model.persistence == pytest.approx(0.95)

    def test_half_life(self):
        """Test half-life computation."""
        model = GARCHForecast(omega=0.00001, alpha=0.1, beta=0.85)

        # Half-life for persistence 0.95
        expected_half_life = np.log(2) / (-np.log(0.95))
        assert model.half_life == pytest.approx(expected_half_life)

    def test_stationarity_constraint(self):
        """Test that alpha + beta >= 1 raises error."""
        with pytest.raises(ValueError, match="alpha \\+ beta must be < 1"):
            GARCHForecast(omega=0.00001, alpha=0.5, beta=0.6)

    def test_invalid_params(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="omega must be positive"):
            GARCHForecast(omega=0, alpha=0.1, beta=0.8)

        with pytest.raises(ValueError, match="alpha must be in"):
            GARCHForecast(omega=0.00001, alpha=-0.1, beta=0.8)


class TestTermStructure:
    """Tests for TermStructure volatility model."""

    def test_interpolation(self):
        """Test linear interpolation."""
        times = (0.0, 10.0, 30.0)
        vols = (0.03, 0.025, 0.02)  # Declining term structure
        model = TermStructure(times=times, vols=vols)

        # Exact points
        assert model.at_time(0.03, 0) == pytest.approx(0.03)
        assert model.at_time(0.03, 10) == pytest.approx(0.025)
        assert model.at_time(0.03, 30) == pytest.approx(0.02)

        # Interpolated point (t=5 is halfway between 0 and 10)
        assert model.at_time(0.03, 5) == pytest.approx(0.0275)

    def test_extrapolation(self):
        """Test extrapolation beyond term structure."""
        times = (0.0, 10.0)
        vols = (0.03, 0.02)
        model = TermStructure(times=times, vols=vols)

        # Beyond end should use last value (np.interp behavior)
        assert model.at_time(0.03, 20) == pytest.approx(0.02)

    def test_invalid_params(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="times must start with 0"):
            TermStructure(times=(1.0, 10.0), vols=(0.03, 0.02))

        with pytest.raises(ValueError, match="strictly increasing"):
            TermStructure(times=(0.0, 10.0, 5.0), vols=(0.03, 0.02, 0.025))

        with pytest.raises(ValueError, match="all vols must be positive"):
            TermStructure(times=(0.0, 10.0), vols=(0.03, -0.02))


class TestNIGWithVolatilityModels:
    """Test NIG distribution with different volatility models."""

    @pytest.fixture
    def nig(self):
        return tpdf.NIG()

    def test_with_linear_growth(self, nig):
        """Test NIG with explicit LinearGrowth model."""
        params = tpdf.NIGParameters(
            mu=0.0,
            delta=0.02,
            alpha=15.0,
            beta=-2.0,
            volatility_model=LinearGrowth(growth_rate=0.05),
        )

        # At t=0, variance should use delta=0.02
        var_t0 = nig.variance(0, params)

        # At t=10, delta should be 0.02 * 1.5 = 0.03
        var_t10 = nig.variance(10, params)

        assert var_t10 > var_t0

    def test_with_mean_reverting(self, nig):
        """Test NIG with mean-reverting volatility."""
        params = tpdf.NIGParameters(
            mu=0.0,
            delta=0.04,  # Elevated current vol
            alpha=15.0,
            beta=-2.0,
            volatility_model=tpdf.mean_reverting(sigma_long=0.02, kappa=0.1),
        )

        # Variance should decrease over time
        var_t0 = nig.variance(0, params)
        var_t20 = nig.variance(20, params)
        var_t50 = nig.variance(50, params)

        assert var_t0 > var_t20 > var_t50

    def test_with_garch(self, nig):
        """Test NIG with GARCH volatility forecast."""
        garch_model = tpdf.garch_forecast(omega=0.0001, alpha=0.1, beta=0.85)

        params = tpdf.NIGParameters(
            mu=0.0,
            delta=0.03,  # Current elevated vol
            alpha=15.0,
            beta=-2.0,
            volatility_model=garch_model,
        )

        # Should converge to long-run
        var_t0 = nig.variance(0, params)
        var_t100 = nig.variance(100, params)

        # Long-run vol from GARCH
        long_run_std = garch_model.long_run_vol

        # Variance at t=100 should be close to long-run
        gamma = np.sqrt(params.alpha**2 - params.beta**2)
        expected_var_t100 = long_run_std * params.alpha**2 / gamma**3

        assert var_t100 == pytest.approx(expected_var_t100, rel=0.1)

    def test_pdf_matrix_with_volatility_model(self, nig):
        """Test vectorized pdf_matrix with volatility model."""
        params = tpdf.NIGParameters(
            mu=0.0,
            delta=0.04,
            alpha=15.0,
            beta=-2.0,
            volatility_model=tpdf.mean_reverting(sigma_long=0.02, kappa=0.1),
        )

        x = np.linspace(-0.2, 0.2, 100)
        time_grid = np.array([0, 10, 20, 30])

        pdf_matrix = nig.pdf_matrix(x, time_grid, params)

        # Should be (T, N) shape
        assert pdf_matrix.shape == (4, 100)

        # All values should be positive
        assert np.all(pdf_matrix > 0)

        # Should integrate to ~1 for each time slice
        dx = x[1] - x[0]
        for t_idx in range(4):
            integral = np.trapezoid(pdf_matrix[t_idx], x)
            assert integral == pytest.approx(1.0, rel=0.05)

    def test_backward_compatibility(self, nig):
        """Test that old delta_growth parameter still works."""
        params_old = tpdf.NIGParameters(
            mu=0.0,
            delta=0.02,
            alpha=15.0,
            beta=-2.0,
            delta_growth=0.05,  # Old way
        )

        params_new = tpdf.NIGParameters(
            mu=0.0,
            delta=0.02,
            alpha=15.0,
            beta=-2.0,
            volatility_model=LinearGrowth(growth_rate=0.05),  # New way
        )

        # Should produce identical results
        var_old = nig.variance(10, params_old)
        var_new = nig.variance(10, params_new)

        assert var_old == pytest.approx(var_new)


class TestConvenienceFunctions:
    """Test convenience factory functions."""

    def test_constant_volatility(self):
        """Test constant_volatility factory."""
        model = tpdf.constant_volatility()
        assert model.at_time(0.02, 100) == 0.02

    def test_linear_growth_factory(self):
        """Test linear_growth factory."""
        model = tpdf.linear_growth(0.05)
        assert model.at_time(0.02, 10) == pytest.approx(0.03)

    def test_mean_reverting_factory(self):
        """Test mean_reverting factory."""
        model = tpdf.mean_reverting(sigma_long=0.02, kappa=0.1)
        assert model.at_time(0.04, 100) == pytest.approx(0.02, rel=0.01)

    def test_garch_forecast_factory(self):
        """Test garch_forecast factory."""
        model = tpdf.garch_forecast(omega=0.00001, alpha=0.1, beta=0.85)
        assert model.persistence == pytest.approx(0.95)
