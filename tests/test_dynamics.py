"""Tests for temporal dynamics models (V2 API)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


class TestConstant:
    """Test Constant dynamics model."""

    def test_projection_returns_constant(self):
        """Test that Constant projects same value at all horizons."""
        const = tpdf.Constant()
        const.fit(np.array([10.0, 10.1, 9.9, 10.0]))

        proj = const.project(current_value=10.0, horizon=10, n_paths=100)
        # All paths at all horizons should equal the long-run value
        assert np.allclose(proj, const.long_run_value)

    def test_fit_uses_mean(self):
        """Test that Constant uses mean of observed values."""
        const = tpdf.Constant()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        const.fit(data)

        assert np.isclose(const.long_run_value, 3.0)

    def test_summary_returns_dict(self):
        """Test summary returns dictionary."""
        const = tpdf.Constant()
        const.fit(np.array([10.0]))
        summary = const.summary()
        assert isinstance(summary, dict)
        assert "long_run_value" in summary


class TestRandomWalk:
    """Test Random Walk dynamics model."""

    def test_mean_equals_current_value(self):
        """Test that mean projection equals current value (no drift)."""
        rw = tpdf.RandomWalk(estimate_drift=False)
        rw.fit(np.array([10.0, 10.1, 9.9, 10.05, 9.95]))

        np.random.seed(42)
        proj = rw.project(current_value=10.0, horizon=100, n_paths=10000)
        # Mean at each horizon should be approximately 10
        mean_path = proj.mean(axis=0)
        assert np.allclose(mean_path, 10.0, atol=0.2)

    def test_variance_grows_with_horizon(self):
        """Test that variance grows with horizon."""
        rw = tpdf.RandomWalk()
        rw.sigma = 0.1  # Set sigma directly for predictable test
        rw.drift = 0.0

        np.random.seed(42)
        proj = rw.project(current_value=10.0, horizon=100, n_paths=10000)
        var_path = proj.var(axis=0)

        # Variance should generally increase
        assert var_path[-1] > var_path[0]

    def test_summary_returns_dict(self):
        """Test summary returns dictionary."""
        rw = tpdf.RandomWalk()
        rw.fit(np.array([10.0, 10.1, 9.9]))
        summary = rw.summary()
        assert isinstance(summary, dict)
        assert "drift" in summary
        assert "sigma" in summary


class TestMeanReverting:
    """Test Mean Reverting (Ornstein-Uhlenbeck) dynamics model."""

    def test_converges_to_long_run_mean(self):
        """Test that projection converges to long-run mean."""
        mr = tpdf.MeanReverting()
        mr.kappa = 0.1
        mr.long_run = 5.0
        mr.sigma = 0.1

        np.random.seed(42)
        proj = mr.project(current_value=10.0, horizon=200, n_paths=5000)
        mean_path = proj.mean(axis=0)

        # Should converge toward 5.0
        assert abs(mean_path[-1] - 5.0) < abs(mean_path[0] - 5.0)

    def test_higher_kappa_faster_reversion(self):
        """Test that higher kappa gives faster mean reversion."""
        mr_slow = tpdf.MeanReverting()
        mr_slow.kappa = 0.05
        mr_slow.long_run = 5.0
        mr_slow.sigma = 0.05

        mr_fast = tpdf.MeanReverting()
        mr_fast.kappa = 0.5
        mr_fast.long_run = 5.0
        mr_fast.sigma = 0.05

        np.random.seed(42)
        proj_slow = mr_slow.project(current_value=10.0, horizon=50, n_paths=5000)
        np.random.seed(42)
        proj_fast = mr_fast.project(current_value=10.0, horizon=50, n_paths=5000)

        # Fast should be closer to long_run at horizon 50
        assert abs(proj_fast.mean(axis=0)[-1] - 5.0) < abs(proj_slow.mean(axis=0)[-1] - 5.0)

    def test_half_life_calculation(self):
        """Test half-life calculation."""
        mr = tpdf.MeanReverting()
        mr.kappa = 0.1
        mr.long_run = 5.0
        mr.sigma = 0.1
        # Half-life = ln(2) / kappa
        expected_hl = np.log(2) / 0.1
        assert np.isclose(mr.half_life(), expected_hl)


class TestAR:
    """Test Autoregressive dynamics model."""

    def test_ar1_fit_and_project(self):
        """Test AR(1) model fitting and projection."""
        ar = tpdf.AR(order=1)
        # Generate AR(1)-like data
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100) * 0.1)
        ar.fit(data)

        proj = ar.project(current_value=data[-1], horizon=50, n_paths=1000)
        assert proj.shape == (1000, 50)

    def test_ar2_fit_and_project(self):
        """Test AR(2) model."""
        ar = tpdf.AR(order=2)
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100) * 0.1)
        ar.fit(data)

        proj = ar.project(current_value=data[-1], horizon=50, n_paths=1000)
        assert proj.shape == (1000, 50)

    def test_summary_returns_dict(self):
        """Test summary returns dictionary."""
        ar = tpdf.AR(order=1)
        ar.fit(np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1.05]))
        summary = ar.summary()
        assert isinstance(summary, dict)
        assert "order" in summary


class TestGARCH:
    """Test GARCH dynamics model."""

    def test_garch_fit_and_project(self):
        """Test GARCH model fitting and projection."""
        garch = tpdf.GARCH(p=1, q=1)

        np.random.seed(42)
        data = np.random.normal(0, 0.02, 100)
        garch.fit(data)

        proj = garch.project(current_value=data[-1], horizon=50, n_paths=1000)
        assert proj.shape == (1000, 50)

    def test_garch_persistence(self):
        """Test GARCH persistence calculation."""
        garch = tpdf.GARCH(p=1, q=1)
        garch.alpha = np.array([0.1])
        garch.beta = np.array([0.85])

        assert np.isclose(garch.persistence(), 0.95)

    def test_summary_returns_dict(self):
        """Test summary returns dictionary."""
        garch = tpdf.GARCH(p=1, q=1)
        garch.omega = 0.0001
        garch.alpha = np.array([0.1])
        garch.beta = np.array([0.85])
        summary = garch.summary()
        assert isinstance(summary, dict)
        assert "omega" in summary
        assert "persistence" in summary


class TestDynamicsEdgeCases:
    """Test edge cases for dynamics models."""

    def test_projection_shape(self):
        """Test projection output shape."""
        mr = tpdf.MeanReverting()
        mr.kappa = 0.1
        mr.long_run = 5.0
        mr.sigma = 0.1

        proj = mr.project(current_value=10.0, horizon=50, n_paths=1000)
        assert proj.shape == (1000, 50)

    def test_constant_with_different_values(self):
        """Test Constant with various input values."""
        const = tpdf.Constant()
        const.fit(np.array([1.0, 2.0, 3.0]))

        proj = const.project(current_value=5.0, horizon=10, n_paths=10)
        # Should return long_run_value (mean of fit data = 2.0)
        assert np.allclose(proj, 2.0)

    def test_random_walk_with_preset_params(self):
        """Test RandomWalk with manually set parameters."""
        rw = tpdf.RandomWalk()
        rw.drift = 0.01
        rw.sigma = 0.02

        np.random.seed(42)
        proj = rw.project(current_value=0.0, horizon=100, n_paths=5000)

        # Mean should drift upward
        mean_final = proj[:, -1].mean()
        assert mean_final > 0.5  # Expected: 100 * 0.01 = 1.0
