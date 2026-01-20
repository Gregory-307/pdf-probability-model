"""Tests for temporal weighting schemes (V2 API)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


class TestSMA:
    """Test Simple Moving Average weighting."""

    def test_weights_sum_to_one(self):
        """Test that SMA weights sum to 1."""
        sma = tpdf.SMA(window=20)
        weights = sma.get_weights(50)
        assert np.isclose(weights.sum(), 1.0)

    def test_weights_equal_within_window(self):
        """Test that all weights within window are equal."""
        sma = tpdf.SMA(window=20)
        weights = sma.get_weights(50)
        # Last 20 should be equal
        window_weights = weights[:20]
        assert np.allclose(window_weights, window_weights[0])

    def test_weights_zero_outside_window(self):
        """Test that weights outside window are zero."""
        sma = tpdf.SMA(window=20)
        weights = sma.get_weights(50)
        assert np.allclose(weights[20:], 0)

    def test_effective_sample_size(self):
        """Test ESS equals window size for SMA."""
        sma = tpdf.SMA(window=20)
        ess = sma.effective_sample_size(50)
        assert np.isclose(ess, 20)

    def test_repr(self):
        """Test string representation."""
        sma = tpdf.SMA(window=20)
        assert "SMA" in repr(sma)
        assert "20" in repr(sma)


class TestEMA:
    """Test Exponential Moving Average weighting."""

    def test_weights_sum_to_one(self):
        """Test that EMA weights sum to 1."""
        ema = tpdf.EMA(halflife=20)
        weights = ema.get_weights(100)
        assert np.isclose(weights.sum(), 1.0)

    def test_most_recent_has_highest_weight(self):
        """Test that most recent observation has highest weight."""
        ema = tpdf.EMA(halflife=20)
        weights = ema.get_weights(100)
        # weights[0] is most recent
        assert weights[0] > weights[1] > weights[2]

    def test_weights_decay_exponentially(self):
        """Test that weights decay exponentially."""
        ema = tpdf.EMA(halflife=20)
        weights = ema.get_weights(100)
        # Ratio between consecutive weights should be constant
        ratios = weights[1:50] / weights[:49]
        assert np.allclose(ratios, ratios[0], rtol=0.01)

    def test_halflife_interpretation(self):
        """Test that weight at halflife is approximately half of initial."""
        ema = tpdf.EMA(halflife=20)
        weights = ema.get_weights(100)
        # Weight at position 20 should be about half of weight at position 0
        ratio = weights[20] / weights[0]
        assert 0.4 < ratio < 0.6  # Allow some tolerance

    def test_effective_sample_size_decreases_with_shorter_halflife(self):
        """Test ESS is lower for shorter halflife (more concentrated)."""
        ema_short = tpdf.EMA(halflife=10)
        ema_long = tpdf.EMA(halflife=50)
        ess_short = ema_short.effective_sample_size(200)
        ess_long = ema_long.effective_sample_size(200)
        assert ess_short < ess_long


class TestLinear:
    """Test Linear decay weighting."""

    def test_weights_sum_to_one(self):
        """Test that linear weights sum to 1."""
        linear = tpdf.Linear(window=20)
        weights = linear.get_weights(50)
        assert np.isclose(weights.sum(), 1.0)

    def test_most_recent_has_highest_weight(self):
        """Test that most recent observation has highest weight."""
        linear = tpdf.Linear(window=20)
        weights = linear.get_weights(50)
        assert weights[0] > weights[1] > weights[10]

    def test_weights_decay_linearly_within_window(self):
        """Test that weights decay linearly within window."""
        linear = tpdf.Linear(window=20)
        weights = linear.get_weights(50)
        # Second differences should be approximately zero (linear)
        diffs = np.diff(weights[:20])
        second_diffs = np.diff(diffs)
        assert np.allclose(second_diffs, 0, atol=1e-10)

    def test_weights_zero_outside_window(self):
        """Test that weights outside window are zero."""
        linear = tpdf.Linear(window=20)
        weights = linear.get_weights(50)
        assert np.allclose(weights[20:], 0)


class TestPowerDecay:
    """Test Power decay weighting."""

    def test_weights_sum_to_one(self):
        """Test that power decay weights sum to 1."""
        power = tpdf.PowerDecay(power=0.5)
        weights = power.get_weights(100)
        assert np.isclose(weights.sum(), 1.0)

    def test_higher_power_faster_decay(self):
        """Test that higher power gives faster decay."""
        power_slow = tpdf.PowerDecay(power=0.5)
        power_fast = tpdf.PowerDecay(power=2.0)

        weights_slow = power_slow.get_weights(100)
        weights_fast = power_fast.get_weights(100)

        # Fast decay should have higher weight on recent obs
        assert weights_fast[0] > weights_slow[0]
        # And lower weight on old obs
        assert weights_fast[50] < weights_slow[50]


class TestGaussian:
    """Test Gaussian weighting."""

    def test_weights_sum_to_one(self):
        """Test that Gaussian weights sum to 1."""
        gauss = tpdf.Gaussian(sigma=10.0)
        weights = gauss.get_weights(100)
        assert np.isclose(weights.sum(), 1.0)

    def test_peak_at_most_recent(self):
        """Test that peak weight is at most recent observation."""
        gauss = tpdf.Gaussian(sigma=10.0)
        weights = gauss.get_weights(100)
        assert weights[0] == weights.max()

    def test_smaller_sigma_more_concentrated(self):
        """Test that smaller sigma gives more concentrated weights."""
        gauss_narrow = tpdf.Gaussian(sigma=5.0)
        gauss_wide = tpdf.Gaussian(sigma=20.0)

        weights_narrow = gauss_narrow.get_weights(100)
        weights_wide = gauss_wide.get_weights(100)

        ess_narrow = gauss_narrow.effective_sample_size(100)
        ess_wide = gauss_wide.effective_sample_size(100)

        assert ess_narrow < ess_wide


class TestCustom:
    """Test Custom weighting scheme."""

    def test_custom_weights_normalized(self):
        """Test that custom weights are normalized to sum to 1."""
        # Custom takes a func(i, n) -> weight
        custom = tpdf.Custom(func=lambda i, n: max(0, 5 - i))
        weights = custom.get_weights(5)
        assert np.isclose(weights.sum(), 1.0)

    def test_custom_weights_preserve_relative_magnitudes(self):
        """Test that relative magnitudes are preserved."""
        # Create weights: [4, 2, 1] by using powers of 2
        # i=0: 2^2=4, i=1: 2^1=2, i=2: 2^0=1
        custom = tpdf.Custom(func=lambda i, n: 2.0 ** (2 - i) if i < 3 else 0)
        weights = custom.get_weights(3)
        # First weight should be 4x third weight (4/1 = 4)
        assert np.isclose(weights[0] / weights[2], 4.0)

    def test_custom_linear_decay(self):
        """Test custom linear decay function."""
        # Linear decay from window
        custom = tpdf.Custom(func=lambda i, n: max(0, 10 - i))
        weights = custom.get_weights(20)
        assert np.isclose(weights.sum(), 1.0)
        # Weight should decay
        assert weights[0] > weights[5] > weights[9]
        # Outside window should be zero
        assert np.isclose(weights[10], 0)


class TestWeightingEdgeCases:
    """Test edge cases for weighting schemes."""

    def test_single_observation(self):
        """Test weights for single observation."""
        schemes = [
            tpdf.SMA(window=20),
            tpdf.EMA(halflife=20),
            tpdf.Linear(window=20),
        ]
        for scheme in schemes:
            weights = scheme.get_weights(1)
            assert len(weights) == 1
            assert np.isclose(weights[0], 1.0)

    def test_window_larger_than_data(self):
        """Test when window is larger than available data."""
        sma = tpdf.SMA(window=100)
        weights = sma.get_weights(20)
        # Should use all available data
        assert np.isclose(weights.sum(), 1.0)
        # All weights should be equal (uniform)
        assert np.allclose(weights, weights[0])
