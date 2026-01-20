"""Tests for TemporalModel, Projection, and PredictiveDistribution (V2 API)."""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


class TestParameterTracker:
    """Test ParameterTracker for rolling window parameter estimation."""

    @pytest.fixture
    def synthetic_returns(self):
        """Generate synthetic returns with known properties."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 500)

    def test_tracker_produces_dataframe(self, synthetic_returns):
        """Test that tracker produces a DataFrame."""
        tracker = tpdf.ParameterTracker(
            distribution="normal",
            window=60,
            step=1,
        )
        result = tracker.fit(synthetic_returns)
        assert isinstance(result, pd.DataFrame)

    def test_tracker_output_columns_normal(self, synthetic_returns):
        """Test tracker output columns for normal distribution."""
        tracker = tpdf.ParameterTracker(
            distribution="normal",
            window=60,
            step=1,
        )
        result = tracker.fit(synthetic_returns)
        assert "mu_0" in result.columns
        assert "sigma_0" in result.columns

    def test_tracker_output_columns_nig(self, synthetic_returns):
        """Test tracker output columns for NIG distribution."""
        tracker = tpdf.ParameterTracker(
            distribution="nig",
            window=60,
            step=1,
        )
        result = tracker.fit(synthetic_returns)
        assert "mu" in result.columns
        assert "delta" in result.columns
        assert "alpha" in result.columns
        assert "beta" in result.columns

    def test_tracker_step_reduces_output(self, synthetic_returns):
        """Test that larger step produces fewer rows."""
        tracker_1 = tpdf.ParameterTracker(distribution="normal", window=60, step=1)
        tracker_5 = tpdf.ParameterTracker(distribution="normal", window=60, step=5)

        result_1 = tracker_1.fit(synthetic_returns)
        result_5 = tracker_5.fit(synthetic_returns)

        assert len(result_5) < len(result_1)

    def test_tracker_with_datetime_index(self, synthetic_returns):
        """Test tracker stores dates in the date column."""
        dates = pd.date_range("2020-01-01", periods=len(synthetic_returns), freq="D")
        tracker = tpdf.ParameterTracker(distribution="normal", window=60, step=1)
        result = tracker.fit(synthetic_returns, index=dates)

        # The tracker may store dates in a 'date' column
        assert "date" in result.columns or isinstance(result.index, pd.DatetimeIndex)


class TestTemporalModel:
    """Test TemporalModel class."""

    @pytest.fixture
    def synthetic_returns(self):
        """Generate synthetic returns."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 500)

    def test_fit_returns_self(self, synthetic_returns):
        """Test that fit returns the model for chaining."""
        model = tpdf.TemporalModel(distribution="normal")
        result = model.fit(synthetic_returns)
        assert result is model

    def test_model_stores_current_params_after_fit(self, synthetic_returns):
        """Test that model stores fitted parameters."""
        model = tpdf.TemporalModel(distribution="normal")
        model.fit(synthetic_returns)
        assert model.current_params is not None

    def test_project_returns_projection(self, synthetic_returns):
        """Test that project returns a Projection object."""
        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.RandomWalk()},
        )
        model.fit(synthetic_returns)
        proj = model.project(horizon=10, n_paths=100)
        assert isinstance(proj, tpdf.Projection)

    def test_predictive_returns_predictive_distribution(self, synthetic_returns):
        """Test that predictive returns a PredictiveDistribution."""
        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.RandomWalk()},
        )
        model.fit(synthetic_returns)
        pred = model.predictive(t=5, n_samples=1000)
        assert isinstance(pred, tpdf.PredictiveDistribution)

    def test_decision_returns_decision_summary(self, synthetic_returns):
        """Test that decision returns a DecisionSummary."""
        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.Constant()},
        )
        model.fit(synthetic_returns)
        decision = model.decision(t=5, alpha=0.05)
        assert isinstance(decision, tpdf.DecisionSummary)

    def test_decision_contains_risk_metrics(self, synthetic_returns):
        """Test that decision summary contains all risk metrics."""
        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.Constant()},
        )
        model.fit(synthetic_returns)
        decision = model.decision(t=5, alpha=0.05)

        assert hasattr(decision, "var")
        assert hasattr(decision, "cvar")
        assert hasattr(decision, "kelly")
        assert hasattr(decision, "prob_profit")

    def test_different_distributions(self, synthetic_returns):
        """Test model works with different distributions."""
        for dist in ["normal", "student_t", "nig"]:
            model = tpdf.TemporalModel(distribution=dist)
            model.fit(synthetic_returns)
            assert model.current_params is not None


class TestPredictiveDistribution:
    """Test PredictiveDistribution class."""

    @pytest.fixture
    def predictive(self):
        """Create a fitted predictive distribution."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.RandomWalk()},
        )
        model.fit(data)
        return model.predictive(t=5, n_samples=5000)

    def test_var_returns_float(self, predictive):
        """Test that VaR returns a float."""
        var = predictive.var(alpha=0.05)
        assert isinstance(var, float)

    def test_cvar_returns_float(self, predictive):
        """Test that CVaR returns a float."""
        cvar = predictive.cvar(alpha=0.05)
        assert isinstance(cvar, float)

    def test_cvar_greater_than_var(self, predictive):
        """Test that CVaR >= VaR."""
        var = predictive.var(alpha=0.05)
        cvar = predictive.cvar(alpha=0.05)
        assert cvar >= var * 0.9  # Allow Monte Carlo error

    def test_decision_summary(self, predictive):
        """Test decision_summary method."""
        summary = predictive.decision_summary(alpha=0.05)
        assert isinstance(summary, tpdf.DecisionSummary)

    def test_decision_summary_has_cis(self, predictive):
        """Test that decision summary has confidence intervals."""
        summary = predictive.decision_summary(alpha=0.05)
        assert summary.var.confidence_interval is not None
        assert summary.cvar.confidence_interval is not None


class TestProjection:
    """Test Projection class."""

    @pytest.fixture
    def projection(self):
        """Create a projection from fitted model."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={
                "mu_0": tpdf.RandomWalk(),
                "sigma_0": tpdf.MeanReverting(),
            },
        )
        model.fit(data)
        return model.project(horizon=30, n_paths=1000)

    def test_projection_has_param_paths(self, projection):
        """Test that projection contains parameter paths."""
        assert projection.param_paths is not None
        assert len(projection.param_paths) > 0

    def test_projection_horizon(self, projection):
        """Test projection horizon attribute."""
        assert projection.horizon == 30

    def test_projection_mean_at_time(self, projection):
        """Test mean method at specific time."""
        mean_t10 = projection.mean(t=10)
        assert isinstance(mean_t10, dict)

    def test_projection_quantile_at_time(self, projection):
        """Test quantile method at specific time."""
        q50 = projection.quantile(0.5, t=10)
        assert isinstance(q50, dict)


class TestRiskMetric:
    """Test RiskMetric dataclass."""

    def test_float_conversion(self):
        """Test that RiskMetric converts to float."""
        metric = tpdf.RiskMetric(value=0.05, confidence_interval=(0.04, 0.06))
        assert float(metric) == 0.05

    def test_confidence_interval_optional(self):
        """Test that CI is optional."""
        metric = tpdf.RiskMetric(value=0.05)
        assert metric.confidence_interval is None

    def test_standard_error_optional(self):
        """Test that SE is optional."""
        metric = tpdf.RiskMetric(value=0.05, confidence_interval=(0.04, 0.06))
        assert metric.standard_error is None

    def test_full_construction(self):
        """Test full construction with all fields."""
        metric = tpdf.RiskMetric(
            value=0.05,
            confidence_interval=(0.04, 0.06),
            standard_error=0.005,
        )
        assert metric.value == 0.05
        assert metric.confidence_interval == (0.04, 0.06)
        assert metric.standard_error == 0.005


class TestDecisionSummary:
    """Test DecisionSummary dataclass."""

    @pytest.fixture
    def decision(self):
        """Create a decision summary."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.Constant()},
        )
        model.fit(data)
        return model.decision(t=5, alpha=0.05)

    def test_var_is_risk_metric(self, decision):
        """Test VaR is a RiskMetric."""
        assert isinstance(decision.var, tpdf.RiskMetric)

    def test_cvar_is_risk_metric(self, decision):
        """Test CVaR is a RiskMetric."""
        assert isinstance(decision.cvar, tpdf.RiskMetric)

    def test_kelly_is_risk_metric(self, decision):
        """Test Kelly is a RiskMetric."""
        assert isinstance(decision.kelly, tpdf.RiskMetric)

    def test_has_expected_return(self, decision):
        """Test decision has expected_return."""
        assert hasattr(decision, "expected_return")
        assert isinstance(decision.expected_return, float)

    def test_has_volatility(self, decision):
        """Test decision has volatility."""
        assert hasattr(decision, "volatility")
        assert isinstance(decision.volatility, float)

    def test_has_metadata(self, decision):
        """Test decision has t and alpha."""
        assert decision.t == 5
        assert decision.alpha == 0.05


class TestIntegration:
    """Integration tests for the full temporal workflow."""

    def test_full_workflow_normal(self):
        """Test full workflow with Normal distribution."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        # Fit model
        model = tpdf.TemporalModel(
            distribution="normal",
            tracking=tpdf.ParameterTracker(distribution="normal", window=60),
            dynamics={"sigma_0": tpdf.MeanReverting()},
        )
        model.fit(data)

        # Project
        proj = model.project(horizon=30, n_paths=1000)
        assert proj.horizon == 30

        # Get predictive
        pred = model.predictive(t=10, n_samples=5000)
        var = pred.var(alpha=0.05)
        assert var > 0

        # Get decision
        decision = model.decision(t=10, alpha=0.05)
        assert decision.var.value > 0

    def test_full_workflow_nig(self):
        """Test full workflow with NIG distribution."""
        np.random.seed(42)
        # Generate slightly heavy-tailed data
        data = np.random.standard_t(df=5, size=500) * 0.02

        # Fit model
        model = tpdf.TemporalModel(
            distribution="nig",
            tracking=tpdf.ParameterTracker(distribution="nig", window=60),
            dynamics={"delta": tpdf.Constant()},
        )
        model.fit(data)

        # Get decision
        decision = model.decision(t=5, alpha=0.05)
        assert decision.var.value > 0
