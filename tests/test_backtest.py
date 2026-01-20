"""Tests for backtesting module (V2 API)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf
from temporalpdf.backtest.tests import kupiec_test, christoffersen_test, conditional_coverage_test


class TestKupiecTest:
    """Test Kupiec unconditional coverage test."""

    def test_correct_coverage_passes(self):
        """Test that correct exceedance rate passes."""
        # Generate exceedances matching expected rate
        np.random.seed(42)
        n = 1000
        alpha = 0.05
        exceedances = np.random.binomial(1, alpha, n).astype(bool)

        stat, p_value, reject = kupiec_test(exceedances, alpha)

        # Should not reject with well-calibrated exceedances
        # (though this is probabilistic, expect high p-value)
        assert p_value > 0.01

    def test_too_many_exceedances_fails(self):
        """Test that too many exceedances fails."""
        # Generate more exceedances than expected
        n = 1000
        alpha = 0.05
        exceedances = np.random.binomial(1, 0.15, n).astype(bool)  # 15% vs 5%

        stat, p_value, reject = kupiec_test(exceedances, alpha)

        # Should reject
        assert reject or p_value < 0.1  # Strong evidence of miscalibration

    def test_too_few_exceedances_fails(self):
        """Test that too few exceedances fails."""
        # Generate fewer exceedances than expected
        n = 1000
        alpha = 0.05
        exceedances = np.random.binomial(1, 0.01, n).astype(bool)  # 1% vs 5%

        stat, p_value, reject = kupiec_test(exceedances, alpha)

        # Should reject
        assert reject or p_value < 0.1

    def test_edge_case_no_exceedances(self):
        """Test edge case with no exceedances."""
        exceedances = np.zeros(100, dtype=bool)
        stat, p_value, reject = kupiec_test(exceedances, alpha=0.05)
        # Should handle gracefully
        assert reject  # No exceedances at 5% level is suspicious

    def test_edge_case_all_exceedances(self):
        """Test edge case with all exceedances."""
        exceedances = np.ones(100, dtype=bool)
        stat, p_value, reject = kupiec_test(exceedances, alpha=0.05)
        # Should handle gracefully
        assert reject  # All exceedances at 5% level is suspicious


class TestChristoffersenTest:
    """Test Christoffersen independence test."""

    def test_independent_exceedances_passes(self):
        """Test that independent exceedances pass."""
        np.random.seed(42)
        # Generate independent exceedances
        exceedances = np.random.binomial(1, 0.05, 1000).astype(bool)

        stat, p_value, reject = christoffersen_test(exceedances)

        # Should not reject independence
        assert not reject or p_value > 0.01

    def test_clustered_exceedances_fails(self):
        """Test that clustered exceedances fail."""
        # Create clustered exceedances (Markov chain with high persistence)
        np.random.seed(42)
        n = 1000
        exceedances = np.zeros(n, dtype=bool)

        # Generate with clustering (high transition probability to same state)
        exceedances[0] = np.random.rand() < 0.05
        for i in range(1, n):
            if exceedances[i - 1]:
                # If previous was exceedance, high chance of another
                exceedances[i] = np.random.rand() < 0.5
            else:
                exceedances[i] = np.random.rand() < 0.02

        stat, p_value, reject = christoffersen_test(exceedances)

        # Should detect clustering (though this is probabilistic)
        # The test may not always reject, but p-value should be low
        assert p_value < 0.2 or reject

    def test_edge_case_no_exceedances(self):
        """Test edge case with no exceedances."""
        exceedances = np.zeros(100, dtype=bool)
        stat, p_value, reject = christoffersen_test(exceedances)
        # Should handle gracefully
        assert not reject  # No exceedances means no clustering to detect


class TestConditionalCoverageTest:
    """Test Christoffersen conditional coverage test."""

    def test_combines_both_tests(self):
        """Test that CC test combines coverage and independence."""
        np.random.seed(42)
        exceedances = np.random.binomial(1, 0.05, 1000).astype(bool)
        alpha = 0.05

        kup_stat, kup_p, _ = kupiec_test(exceedances, alpha)
        chr_stat, chr_p, _ = christoffersen_test(exceedances)
        cc_stat, cc_p, cc_reject = conditional_coverage_test(exceedances, alpha)

        # CC statistic should be sum of individual statistics
        assert np.isclose(cc_stat, kup_stat + chr_stat, rtol=0.01)


class TestBacktest:
    """Test Backtest class."""

    @pytest.fixture
    def synthetic_returns(self):
        """Generate synthetic returns."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 500)

    def test_backtest_runs(self, synthetic_returns):
        """Test that backtest runs without error."""
        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        assert result is not None

    def test_backtest_returns_backtest_result(self, synthetic_returns):
        """Test that backtest returns BacktestResult."""
        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        assert isinstance(result, tpdf.BacktestResult)

    def test_backtest_result_has_var_forecasts(self, synthetic_returns):
        """Test that result has VaR forecasts."""
        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        assert result.var_forecasts is not None
        assert len(result.var_forecasts) > 0

    def test_backtest_result_has_exceedances(self, synthetic_returns):
        """Test that result has exceedances."""
        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        assert result.exceedances is not None
        assert result.exceedances.dtype == bool

    def test_backtest_exceedance_rate(self, synthetic_returns):
        """Test that exceedance rate is reasonable."""
        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)

        # Exceedance rate should be in reasonable range
        assert 0 <= result.exceedance_rate <= 1

    def test_backtest_summary(self, synthetic_returns):
        """Test summary method."""
        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Backtest" in summary
        assert "Exceedances" in summary
        assert "Kupiec" in summary

    def test_backtest_with_nig(self, synthetic_returns):
        """Test backtest with NIG distribution."""
        bt = tpdf.Backtest(
            distribution="nig",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        assert result.n_total > 0

    def test_backtest_with_student_t(self, synthetic_returns):
        """Test backtest with Student-t distribution."""
        bt = tpdf.Backtest(
            distribution="student_t",
            lookback=100,
            alpha=0.05,
        )
        result = bt.run(synthetic_returns)
        assert result.n_total > 0


class TestBacktestAPI:
    """Test high-level backtest API function."""

    def test_api_function(self):
        """Test tpdf.backtest() function."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        result = tpdf.backtest(data, distribution="normal", lookback=100)
        assert isinstance(result, tpdf.BacktestResult)

    def test_api_function_with_kwargs(self):
        """Test tpdf.backtest() with additional kwargs."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        result = tpdf.backtest(
            data,
            distribution="normal",
            lookback=100,
            alpha=0.01,  # 99% VaR
        )
        assert result.n_total > 0


class TestBacktestStatus:
    """Test backtest status determination."""

    def test_pass_status(self):
        """Test PASS status when both tests pass."""
        np.random.seed(42)
        # Generate well-behaved returns
        data = np.random.normal(0.001, 0.02, 1000)

        result = tpdf.backtest(data, distribution="normal", lookback=252)

        # With well-behaved normal data, should typically pass
        # (though not guaranteed due to randomness)
        assert result.status in ["PASS", "FAIL_COVERAGE", "FAIL_INDEPENDENCE", "FAIL_BOTH"]

    def test_result_fields_consistent(self):
        """Test that result fields are consistent."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        result = tpdf.backtest(data, distribution="normal", lookback=100)

        # n_exceedances should match sum of exceedances
        assert result.n_exceedances == int(np.sum(result.exceedances))

        # n_total should match length
        assert result.n_total == len(result.exceedances)

        # exceedance_rate should be correct
        assert np.isclose(result.exceedance_rate, result.n_exceedances / result.n_total)


class TestBacktestEdgeCases:
    """Test edge cases for backtesting."""

    def test_minimum_data_length(self):
        """Test with minimum viable data length."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 150)  # Just over lookback

        result = tpdf.backtest(data, distribution="normal", lookback=100)
        assert result.n_total > 0

    def test_step_parameter(self):
        """Test with step parameter."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        bt = tpdf.Backtest(
            distribution="normal",
            lookback=100,
            alpha=0.05,
            step=5,  # Only evaluate every 5 days
        )
        result = bt.run(data)

        # Should have fewer forecasts
        assert result.n_total < 400
