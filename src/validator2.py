import pandas as pd
import numpy as np
import random
from Commercial_PDF import commercial_pdf_with_time_decay, cumulative_expected_value_corrected
import matplotlib.pyplot as plt


def calculate_future_average_change(data, window_size=10):
    """
    Calculate the future average percentage change based on the next N closing prices.

    Parameters:
        data (DataFrame): The dataset containing the 'close' column.
        window_size (int): The number of future rows to average for percentage change.

    Returns:
        Series: A column with the future average percentage change for each row.
    """
    future_avg = data["close"].rolling(window=window_size, min_periods=1).mean().shift(-window_size)
    pct_change = ((future_avg - data["close"]) / data["close"]) * 100  # Percentage change
    return pct_change


def validate_single_point(
    observed_change, pdf_function, x_grid, time_point, mu_0, sigma_0, alpha, delta, beta, lambda_decay=0.0, k=1.0
):
    """
    Validate a single point's approximation using the PDF and expected value.

    Parameters:
        observed_change (float): The observed percentage change at a specific time point (in fractions, not %).
        pdf_function (function): The PDF function to evaluate (e.g., commercial_pdf_with_time_decay).
        x_grid (array-like): The grid of percentage changes for PDF evaluation.
        time_point (float): The specific time point for validation.
        mu_0 (float): Initial mean.
        sigma_0 (float): Initial volatility.
        alpha (float): Skewness.
        delta (float): Mean rate (linear trend in the mean over time).
        beta (float): Volatility growth rate.
        lambda_decay (float): Time decay parameter to reduce prediction confidence.
        k (float): Tail sharpness parameter.

    Returns:
        dict: Validation metrics for the single point, including log-likelihood, MAE, and E(X).
    """
    # Create a single-point time grid
    time_grid = np.array([time_point])

    # Generate the PDF for the given time point
    pdf_matrix, x_grid, _ = pdf_function(
        mu_0=mu_0,
        sigma_0=sigma_0,
        alpha=alpha,
        delta=delta,
        beta=beta,
        x=x_grid,
        time_grid=time_grid,
        lambda_decay=lambda_decay,
        k=k,
    )

    # Normalize the PDF and calculate E(X)
    pdf_sum = np.sum(pdf_matrix, axis=1)
    pdf_sum = np.where(pdf_sum == 0, 1, pdf_sum)  # Avoid division by zero
    expected_value = np.dot(pdf_matrix[0], x_grid / 100) / pdf_sum[0]  # E(X)

    # Interpolate the PDF value for the observed change
    pdf_value = np.interp(observed_change, x_grid, pdf_matrix[0, :])
    pdf_value = max(pdf_value, 1e-10)  # Avoid invalid or zero PDF values

    # Metrics
    log_likelihood = np.log(pdf_value)
    mae = abs(expected_value - observed_change)
    mse = (expected_value - observed_change) ** 2

    return {
        "log_likelihood": log_likelihood,
        "mae": mae,
        "mse": mse,
        "expected_value": expected_value,
    }


def validate_model_accuracy(data, pdf_function, x_grid, window_size=10, sample_size=10):
    """
    Validate the PDF and E(X) approximation against observed data.

    Parameters:
        data (DataFrame): A dataset with the 5 coefficients for each stock and observed percentage changes.
        pdf_function (function): The PDF function to evaluate (e.g., commercial_pdf_with_time_decay).
        x_grid (array-like): The grid of percentage changes for PDF evaluation.
        window_size (int): The number of future rows to calculate the average change.
        sample_size (int): Number of stocks to randomly sample for validation.

    Returns:
        dict: A summary of validation metrics, including log-likelihood, MAE, and R².
    """
    # Add future average percentage change to the dataset
    data["future_avg_change"] = calculate_future_average_change(data, window_size)

    # Metrics storage
    total_log_likelihood = 0
    total_mae = 0
    total_mse = 0
    total_samples = 0

    # Storage for R² calculation
    all_actual_values = []
    all_predicted_values = []

    # Randomly sample stocks or iterate through all stocks
    sampled_stocks = random.sample(data["ticker"].unique().tolist(), min(sample_size, len(data["ticker"].unique())))

    for ticker in sampled_stocks:
        # Get the stock's data
        stock_data = data[data["ticker"] == ticker]
        
        # Extract coefficients
        mu_0 = stock_data["mean"].iloc[0]
        sigma_0 = stock_data["volatility"].iloc[0]
        alpha = stock_data["skewness"].iloc[0]
        delta = stock_data["mean_rate"].iloc[0]
        beta = stock_data["volatility_growth"].iloc[0]

        # Dynamically generate time_grid to match observed changes
        observed_changes = stock_data["future_avg_change"].values / 100  # Convert to fractions
        time_grid = np.linspace(0, len(observed_changes) - 1, len(observed_changes))  # Match observed length

        # Skip rows with NaN future_avg_change
        for i, observed_change in enumerate(observed_changes):
            if np.isnan(observed_change):
                continue

            # Validate single point
            metrics = validate_single_point(
                observed_change,
                pdf_function,
                x_grid,
                time_grid[i],
                mu_0,
                sigma_0,
                alpha,
                delta,
                beta,
            )

            # Accumulate metrics
            total_log_likelihood += metrics["log_likelihood"]
            total_mae += metrics["mae"]
            total_mse += metrics["mse"]
            total_samples += 1

            # Store values for R² calculation
            all_actual_values.append(observed_change)
            all_predicted_values.append(metrics["expected_value"])

    # Aggregate metrics
    avg_log_likelihood = total_log_likelihood / total_samples if total_samples > 0 else 0
    mae = total_mae / total_samples if total_samples > 0 else 0
    mse = total_mse / total_samples if total_samples > 0 else 0

    # Calculate R² (coefficient of determination)
    if total_samples > 0:
        all_actual_values = np.array(all_actual_values)
        all_predicted_values = np.array(all_predicted_values)
        ss_res = np.sum((all_actual_values - all_predicted_values) ** 2)  # Residual sum of squares
        ss_tot = np.sum((all_actual_values - np.mean(all_actual_values)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    else:
        r_squared = 0

    return {
        "Average Log-Likelihood": avg_log_likelihood,
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "R² (R-squared)": r_squared,
        "Total Samples": total_samples,
    }


# Example Usage
file_path = "data/processed_data.csv"
data = pd.read_csv(file_path)

# Grid Definitions
x_grid = np.linspace(-0.2, 0.2, 200)  # Percentage change grid

# Run validation
validation_results = validate_model_accuracy(
    data=data, pdf_function=commercial_pdf_with_time_decay,
    x_grid=x_grid, window_size=10, sample_size=10
)

print(validation_results)
