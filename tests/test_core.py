"""Tests for core module."""

import numpy as np
import pytest

from temporalpdf import (
    EvaluationGrid,
    PDFResult,
    ValidationResult,
    evaluate,
    Normal,
    NormalParameters,
    GeneralizedLaplace,
    GeneralizedLaplaceParameters,
)


class TestEvaluationGrid:
    """Tests for EvaluationGrid."""

    def test_from_ranges_creates_grid(self):
        """from_ranges should create proper grid."""
        grid = EvaluationGrid.from_ranges(
            value_range=(-0.5, 0.5),
            time_range=(0, 60),
            value_points=100,
            time_points=50,
        )

        assert len(grid.value_grid) == 100
        assert len(grid.time_grid) == 50
        assert grid.value_grid[0] == pytest.approx(-0.5)
        assert grid.value_grid[-1] == pytest.approx(0.5)
        assert grid.time_grid[0] == pytest.approx(0)
        assert grid.time_grid[-1] == pytest.approx(60)

    def test_value_range_property(self):
        """value_range property should return correct bounds."""
        grid = EvaluationGrid.from_ranges(
            value_range=(-0.3, 0.3),
            time_range=(0, 30),
        )
        assert grid.value_range == (-0.3, 0.3)

    def test_time_range_property(self):
        """time_range property should return correct bounds."""
        grid = EvaluationGrid.from_ranges(
            value_range=(-0.3, 0.3),
            time_range=(0, 30),
        )
        assert grid.time_range == (0, 30)


class TestPDFResult:
    """Tests for PDFResult."""

    def test_slice_at_time(self):
        """slice_at_time should return correct slice."""
        # Create a simple result
        value_grid = np.linspace(-0.5, 0.5, 100)
        time_grid = np.linspace(0, 60, 50)
        pdf_matrix = np.random.rand(50, 100)

        result = PDFResult(
            pdf_matrix=pdf_matrix,
            value_grid=value_grid,
            time_grid=time_grid,
            distribution_name="Test",
            parameters={},
        )

        x, pdf = result.slice_at_time(30.0)

        assert len(x) == 100
        assert len(pdf) == 100

    def test_expected_values_shape(self):
        """expected_values should have correct shape."""
        value_grid = np.linspace(-0.5, 0.5, 100)
        time_grid = np.linspace(0, 60, 50)
        pdf_matrix = np.random.rand(50, 100)

        result = PDFResult(
            pdf_matrix=pdf_matrix,
            value_grid=value_grid,
            time_grid=time_grid,
            distribution_name="Test",
            parameters={},
        )

        assert len(result.expected_values) == 50

    def test_cumulative_expected_value(self):
        """cumulative_expected_value should return float."""
        value_grid = np.linspace(-0.5, 0.5, 100)
        time_grid = np.linspace(0, 60, 50)
        pdf_matrix = np.random.rand(50, 100)

        result = PDFResult(
            pdf_matrix=pdf_matrix,
            value_grid=value_grid,
            time_grid=time_grid,
            distribution_name="Test",
            parameters={},
        )

        assert isinstance(result.cumulative_expected_value, float)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_summary_returns_string(self):
        """summary() should return formatted string."""
        result = ValidationResult(
            log_likelihood=-2.5,
            mae=0.01,
            mse=0.0001,
            r_squared=0.85,
            total_samples=100,
            per_sample_metrics=np.array([]),
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Log-Likelihood" in summary
        assert "MAE" in summary
        assert "MSE" in summary


class TestEvaluateFunction:
    """Tests for the evaluate convenience function."""

    def test_evaluate_with_distribution_instance(self):
        """evaluate should work with distribution instance."""
        dist = Normal()
        params = NormalParameters(mu_0=0.0, sigma_0=0.05)

        result = evaluate(
            dist,
            params,
            value_range=(-0.3, 0.3),
            time_range=(0, 30),
            value_points=50,
            time_points=25,
        )

        assert result.pdf_matrix.shape == (25, 50)
        assert "Normal" in result.distribution_name

    def test_evaluate_with_string_name(self):
        """evaluate should work with distribution name string."""
        params = NormalParameters(mu_0=0.0, sigma_0=0.05)

        result = evaluate(
            "normal",
            params,
            value_range=(-0.3, 0.3),
            time_range=(0, 30),
        )

        assert "Normal" in result.distribution_name

    def test_evaluate_generalized_laplace(self):
        """evaluate should work with Generalized Laplace."""
        params = GeneralizedLaplaceParameters(
            mu_0=0.0,
            sigma_0=0.05,
            alpha=0.5,
            delta=0.001,
            beta=0.01,
        )

        result = evaluate(
            GeneralizedLaplace(),
            params,
            time_range=(0, 60),
        )

        assert result.pdf_matrix.shape[0] == 100  # default time_points
        assert result.pdf_matrix.shape[1] == 200  # default value_points

    def test_evaluate_default_ranges(self):
        """evaluate should use sensible defaults."""
        params = NormalParameters(mu_0=0.0, sigma_0=0.05)

        result = evaluate("normal", params)

        assert result.value_grid[0] == pytest.approx(-0.2)
        assert result.value_grid[-1] == pytest.approx(0.2)
        assert result.time_grid[0] == pytest.approx(0.0)
        assert result.time_grid[-1] == pytest.approx(60.0)
