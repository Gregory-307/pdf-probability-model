import pandas as pd
import numpy as np
import random
from Commercial_PDF import commercial_pdf_with_time_decay, cumulative_expected_value_corrected

# Validation Function
def validate_model_accuracy(data, pdf_function, x_grid, sample_size=10):
    """
    Validate the PDF and E(X) approximation against observed data.

    Parameters:
        data (DataFrame): A dataset with the 5 coefficients for each stock and observed percentage changes.
        pdf_function (function): The PDF function to evaluate (e.g., commercial_pdf_with_time_decay).
        x_grid (array-like): The grid of percentage changes for PDF evaluation.
        sample_size (int): Number of stocks to randomly sample for validation.

    Returns:
        dict: A summary of validation metrics, including log-likelihood, MAE, and R².
    """
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
        observed_changes = stock_data["pct_change"].values # / 100  # Convert to fractions
        time_grid = np.linspace(0, len(observed_changes) - 1, len(observed_changes))  # Match observed length

        # Generate the PDF
        pdf_matrix, x_grid, _ = pdf_function(
            mu_0=mu_0, sigma_0=sigma_0, alpha=alpha, delta=delta, beta=beta,
            x=x_grid, time_grid=time_grid
        )

        # Validate that pdf_matrix has the correct dimensions
        if pdf_matrix.shape[0] != len(time_grid):
            raise ValueError("Mismatch between time_grid and PDF matrix rows.")

        # Calculate E(X)
        pdf_sum = np.sum(pdf_matrix, axis=1)
        pdf_sum = np.where(pdf_sum == 0, 1, pdf_sum)  # Avoid division by zero
        expected_values = np.dot(pdf_matrix, x_grid / 100) / pdf_sum

        # Validate at each time point
        for i, actual_change in enumerate(observed_changes):
            if i >= pdf_matrix.shape[0]:  # Guard against index mismatch
                continue
            
            # Interpolate PDF value for observed change
            pdf_value = np.interp(actual_change, x_grid, pdf_matrix[i, :])
            if pdf_value <= 0:  # Guard against invalid PDF values
                pdf_value = 1e-10
            
            # Log-Likelihood
            total_log_likelihood += np.log(pdf_value)
            
            # MAE and MSE for E(X)
            total_mae += abs(expected_values[i] - actual_change)
            total_mse += (expected_values[i] - actual_change) ** 2
            total_samples += 1

            # Store values for R² calculation
            all_actual_values.append(actual_change)
            all_predicted_values.append(expected_values[i])

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
        "Total Samples": total_samples
    }


# Example Usage
file_path = "data/processed_data.csv"
data = pd.read_csv(file_path)

# Grid Definitions
x_grid = np.linspace(-0.2, 0.2, 200)  # Percentage change grid
time_grid = np.linspace(0, 60, 100)  # Time grid

# Run validation
validation_results = validate_model_accuracy(
    data=data, pdf_function=commercial_pdf_with_time_decay,
    x_grid=x_grid, sample_size=500
)

print(validation_results)
