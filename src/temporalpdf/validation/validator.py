"""Model validation class."""

from typing import Callable
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..core.distribution import TimeEvolvingDistribution, DistributionParameters
from ..core.result import ValidationResult
from .metrics import log_likelihood, mae, mse, r_squared


class Validator:
    """
    Validates distribution models against observed data.

    Provides comprehensive validation metrics including:
    - Log-likelihood: How well does the PDF predict observed values?
    - Mean Absolute Error (MAE): Average absolute prediction error
    - Mean Squared Error (MSE): Average squared prediction error
    - R-squared: Coefficient of determination

    Example:
        >>> validator = Validator(distribution, value_grid)
        >>> result = validator.validate(data, params, observed_col="pct_change")
        >>> print(result.summary())

    Attributes:
        distribution: The distribution to validate
        value_grid: Grid of values for PDF evaluation
    """

    def __init__(
        self,
        distribution: TimeEvolvingDistribution,  # type: ignore[type-arg]
        value_grid: NDArray[np.float64],
    ):
        """
        Initialize the validator.

        Args:
            distribution: Distribution instance to validate
            value_grid: Array of values for PDF evaluation
        """
        self.distribution = distribution
        self.value_grid = value_grid

    def validate(
        self,
        data: pd.DataFrame,
        params: DistributionParameters,
        observed_column: str,
        time_column: str | None = None,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> ValidationResult:
        """
        Validate the distribution against observed data.

        For each observation, computes the PDF at that time point,
        calculates the expected value, and compares to the observed value.

        Args:
            data: DataFrame containing observed values
            params: Distribution parameters
            observed_column: Column with observed values to validate against
            time_column: Column with time values (uses row index if None)
            sample_size: Optional sample size for large datasets
            random_state: Random seed for sampling

        Returns:
            ValidationResult with computed metrics
        """
        df = data.dropna(subset=[observed_column]).copy()

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)

        # Get observed values and time grid
        observed = df[observed_column].values
        if time_column and time_column in df.columns:
            time_grid = df[time_column].values
        else:
            time_grid = np.arange(len(observed))

        return self._compute_metrics(observed, time_grid, params)

    def validate_arrays(
        self,
        observed: NDArray[np.float64],
        time_points: NDArray[np.float64],
        params: DistributionParameters,
    ) -> ValidationResult:
        """
        Validate using raw arrays instead of DataFrame.

        Args:
            observed: Array of observed values
            time_points: Array of time points corresponding to observations
            params: Distribution parameters

        Returns:
            ValidationResult with computed metrics
        """
        return self._compute_metrics(observed, time_points, params)

    def _compute_metrics(
        self,
        observed: NDArray[np.float64],
        time_points: NDArray[np.float64],
        params: DistributionParameters,
    ) -> ValidationResult:
        """Compute validation metrics for observed vs predicted."""
        total_ll = 0.0
        total_mae = 0.0
        total_mse = 0.0
        predicted_values: list[float] = []
        per_sample: list[tuple[float, float, float, float]] = []

        for obs, t in zip(observed, time_points):
            # Get PDF at this time point
            pdf_values = self.distribution.pdf(self.value_grid, float(t), params)

            # Calculate expected value
            pdf_sum = np.sum(pdf_values)
            if pdf_sum > 0:
                exp_val = float(np.dot(pdf_values, self.value_grid) / pdf_sum)
            else:
                exp_val = 0.0

            predicted_values.append(exp_val)

            # Calculate metrics
            ll = log_likelihood(obs, pdf_values, self.value_grid)
            sample_mae = mae(exp_val, obs)
            sample_mse = mse(exp_val, obs)

            total_ll += ll
            total_mae += sample_mae
            total_mse += sample_mse
            per_sample.append((ll, sample_mae, sample_mse, exp_val))

        n = len(observed)
        predicted_array = np.array(predicted_values)

        # Calculate R-squared
        r2 = r_squared(predicted_array, observed)

        return ValidationResult(
            log_likelihood=total_ll / n,
            mae=total_mae / n,
            mse=total_mse / n,
            r_squared=r2,
            total_samples=n,
            per_sample_metrics=np.array(per_sample),
        )

    def validate_single(
        self,
        observed: float,
        t: float,
        params: DistributionParameters,
    ) -> dict[str, float]:
        """
        Validate a single observation.

        Useful for debugging or detailed analysis of individual predictions.

        Args:
            observed: The observed value
            t: Time point
            params: Distribution parameters

        Returns:
            Dictionary with log_likelihood, mae, mse, expected_value, pdf_at_observed
        """
        pdf_values = self.distribution.pdf(self.value_grid, t, params)

        # Calculate expected value
        pdf_sum = np.sum(pdf_values)
        if pdf_sum > 0:
            exp_val = float(np.dot(pdf_values, self.value_grid) / pdf_sum)
        else:
            exp_val = 0.0

        # PDF at observed point
        pdf_at_obs = float(np.interp(observed, self.value_grid, pdf_values))

        return {
            "log_likelihood": log_likelihood(observed, pdf_values, self.value_grid),
            "mae": mae(exp_val, observed),
            "mse": mse(exp_val, observed),
            "expected_value": exp_val,
            "observed_value": observed,
            "pdf_at_observed": pdf_at_obs,
        }


class CrossValidator:
    """
    Cross-validation for distribution models.

    Provides k-fold cross-validation to assess model generalization.
    """

    def __init__(
        self,
        distribution: TimeEvolvingDistribution,  # type: ignore[type-arg]
        value_grid: NDArray[np.float64],
        n_folds: int = 5,
    ):
        """
        Initialize cross-validator.

        Args:
            distribution: Distribution instance to validate
            value_grid: Array of values for PDF evaluation
            n_folds: Number of cross-validation folds
        """
        self.distribution = distribution
        self.value_grid = value_grid
        self.n_folds = n_folds
        self._validator = Validator(distribution, value_grid)

    def cross_validate(
        self,
        data: pd.DataFrame,
        params: DistributionParameters,
        observed_column: str,
        time_column: str | None = None,
        random_state: int = 42,
    ) -> list[ValidationResult]:
        """
        Perform k-fold cross-validation.

        Args:
            data: DataFrame containing observed values
            params: Distribution parameters
            observed_column: Column with observed values
            time_column: Column with time values
            random_state: Random seed for shuffling

        Returns:
            List of ValidationResult, one per fold
        """
        df = data.dropna(subset=[observed_column]).copy()
        n = len(df)

        # Shuffle indices
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)

        # Split into folds
        fold_size = n // self.n_folds
        results: list[ValidationResult] = []

        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n

            test_indices = indices[start:end]
            test_data = df.iloc[test_indices]

            result = self._validator.validate(
                test_data, params, observed_column, time_column
            )
            results.append(result)

        return results

    def summary(self, results: list[ValidationResult]) -> dict[str, float]:
        """
        Summarize cross-validation results.

        Args:
            results: List of ValidationResult from cross_validate

        Returns:
            Dictionary with mean and std of each metric
        """
        lls = [r.log_likelihood for r in results]
        maes = [r.mae for r in results]
        mses = [r.mse for r in results]
        r2s = [r.r_squared for r in results]

        return {
            "log_likelihood_mean": float(np.mean(lls)),
            "log_likelihood_std": float(np.std(lls)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "r_squared_mean": float(np.mean(r2s)),
            "r_squared_std": float(np.std(r2s)),
        }
