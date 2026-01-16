import pandas as pd
import numpy as np
import random
from Commercial_PDF import commercial_pdf_with_time_decay, cumulative_expected_value_corrected
from mpl_toolkits.mplot3d import Axes3D
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

def validate_and_plot_points(data, pdf_function, x_grid, window_size=10, num_points=5):
    """
    Validate and plot a few points to visually assess PDF and E(X) predictions.

    Parameters:
        data (DataFrame): A dataset with coefficients, close prices, and other columns.
        pdf_function (function): The PDF function to evaluate (e.g., commercial_pdf_with_time_decay).
        x_grid (array-like): The grid of percentage changes for PDF evaluation.
        window_size (int): The number of future rows to calculate the average change.
        num_points (int): Number of points to validate and plot.

    Returns:
        None: Displays the plots for visual assessment.
    """
    # Add future average percentage change to the dataset
    data["future_avg_change"] = calculate_future_average_change(data, window_size)

    # Randomly sample `num_points` rows for validation
    sampled_data = data.dropna(subset=["future_avg_change"]).sample(n=num_points, random_state=42)

    for _, row in sampled_data.iterrows():
        # Extract parameters
        mu_0 = row["mean"]
        sigma_0 = row["volatility"]
        alpha = row["skewness"]
        delta = row["mean_rate"]
        beta = row["volatility_growth"]
        observed_change = row["future_avg_change"] / 100  # Convert to fractions
        time_point = row.name  # Assume row index corresponds to time

        # Validate the point
        time_grid = np.array([time_point])  # Single-point time grid
        pdf_matrix, x_grid, _ = pdf_function(
            mu_0=mu_0,
            sigma_0=sigma_0,
            alpha=alpha,
            delta=delta,
            beta=beta,
            x=x_grid,
            time_grid=time_grid,
            lambda_decay=0.0,
            k=1.0,
        )

        # Normalize the PDF and calculate E(X)
        pdf_sum = np.sum(pdf_matrix, axis=1)
        pdf_sum = np.where(pdf_sum == 0, 1, pdf_sum)  # Avoid division by zero
        expected_value = np.dot(pdf_matrix[0], x_grid / 100) / pdf_sum[0]  # E(X)

        # Plot the PDF and observed value
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid * 100, pdf_matrix[0], label="PDF", color="blue")  # Convert to percentage scale
        plt.axvline(observed_change * 100, color="red", linestyle="--", label="Observed Change")  # Observed in %
        plt.axvline(expected_value * 100, color="green", linestyle="--", label="E(X) Prediction")  # E(X) in %
        plt.title(f"Validation for Ticker: {row['ticker']} at Time: {time_point}")
        plt.xlabel("Percentage Change (%)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid()
        plt.show()

def validate_single_point_with_plot(
    observed_change, pdf_function, x_grid, time_point, mu_0, sigma_0, alpha, delta, beta, lambda_decay=0.0, k=1.0
):
    """
    Validate a single point's approximation using the PDF and expected value, and plot the PDF matrix.

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

    # Plot the 3D PDF matrix for the single time point
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_grid * 100, time_grid)  # Convert x_grid to percentage scale
    ax.plot_surface(X, Y, pdf_matrix, cmap='viridis', edgecolor='none')
    ax.set_title("3D PDF Surface for Single Time Point")
    ax.set_xlabel("Percentage Change (%)")
    ax.set_ylabel("Time")
    ax.set_zlabel("Probability Density")
    plt.show()

    # Plot the PDF as a 2D heatmap (optional for debugging)
    plt.figure(figsize=(10, 6))
    plt.imshow(pdf_matrix, aspect='auto', extent=[x_grid.min() * 100, x_grid.max() * 100, time_grid.min(), time_grid.max()],
               origin='lower', cmap='viridis')
    plt.colorbar(label="Probability Density")
    plt.title("PDF Heatmap for Single Time Point")
    plt.xlabel("Percentage Change (%)")
    plt.ylabel("Time")
    plt.show()

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


def validate_single_point_debug_and_plot(
    observed_change, pdf_function, x_grid, time_point, mu_0, sigma_0, alpha, delta, beta, lambda_decay=0.0, k=1.0
):
    """
    Debug and validate a single point's approximation, and plot the PDF matrix.

    Parameters:
        observed_change (float): The observed percentage change at a specific time point (in fractions, not %).
        pdf_function (function): The PDF function to evaluate (e.g., commercial_pdf_with_time_decay).
        x_grid (array-like): The grid of percentage changes for PDF evaluation.
        time_point (float): The specific time point for validation.
        mu_0, sigma_0, alpha, delta, beta (float): Coefficients for PDF generation.
        lambda_decay (float): Time decay parameter to reduce prediction confidence.
        k (float): Tail sharpness parameter.

    Returns:
        None: Prints debug information and plots the results.
    """
    # Print the coefficients
    print("Coefficients:")
    print(f"mu_0: {mu_0}, sigma_0: {sigma_0}, alpha: {alpha}, delta: {delta}, beta: {beta}")

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

    # Plot the PDF for the single time point
    plt.figure(figsize=(12, 8))
    plt.plot(x_grid * 100, pdf_matrix[0], label="PDF", color="blue")
    plt.axvline(observed_change * 100, color="red", linestyle="--", label="Observed Change")
    plt.title("PDF Matrix (Single Slice)")
    plt.xlabel("Percentage Change (%)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate and print the expected value using the provided function
    expected_value = cumulative_expected_value_corrected(x_grid, pdf_matrix, time_grid)
    print(f"Expected Value (E(X)): {expected_value * 100:.2f}%")

    # Print the actual observed change
    print(f"Actual Observed Change: {observed_change * 100:.2f}%")

    # Optional: Print the 2D PDF slice
    print("2D PDF Slice:")
    print(pdf_matrix[0])


def plot_pdf_matrix(pdf_matrix, x_grid, time_grid, title="PDF Matrix"):
    """
    Plot a 3D surface of the PDF matrix.

    Parameters:
        pdf_matrix (2D array): The PDF matrix to plot.
        x_grid (array-like): Grid of percentage changes.
        time_grid (array-like): Grid of time points.
        title (str): Title of the plot.

    Returns:
        None: Displays the 3D plot.
    """
    X, T = np.meshgrid(x_grid, time_grid)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, pdf_matrix, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Percentage Change")
    ax.set_zlabel("Probability Density")
    plt.show()

def validate_single_point_debug_and_plot(
    observed_change, pdf_function, x_grid, time_grid, mu_0, sigma_0, alpha, delta, beta, lambda_decay=0.0, k=1.0
):
    """
    Debug and validate a single point's approximation, and plot the PDF matrix.

    Parameters:
        observed_change (float): The observed percentage change at a specific time point (in fractions, not %).
        pdf_function (function): The PDF function to evaluate (e.g., commercial_pdf_with_time_decay).
        x_grid (array-like): The grid of percentage changes for PDF evaluation.
        time_point (float): The specific time point for validation.
        mu_0, sigma_0, alpha, delta, beta (float): Coefficients for PDF generation.
        lambda_decay (float): Time decay parameter to reduce prediction confidence.
        k (float): Tail sharpness parameter.

    Returns:
        None: Prints debug information and plots the results.
    """
    # Print the coefficients
    print("Coefficients:")
    print(f"mu_0: {mu_0}, sigma_0: {sigma_0}, alpha: {alpha}, delta: {delta}, beta: {beta}")

    # Generate the PDF for the given time point
    pdf_matrix, _, _ = pdf_function(
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
    print(pdf_matrix)
    plot_pdf_matrix(pdf_matrix, x_grid, time_grid, title="Example PDF Matrix")

    # Plot the PDF for the single time point
    plt.figure(figsize=(12, 8))
    plt.plot(x_grid * 100, pdf_matrix[0], label="PDF", color="blue")
    plt.axvline(observed_change * 100, color="red", linestyle="--", label="Observed Change")
    plt.title("PDF Matrix (Single Slice)")
    plt.xlabel("Percentage Change (%)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate and print the expected value using the provided function
    expected_value = cumulative_expected_value_corrected(x_grid, pdf_matrix, time_grid)
    print(f"Expected Value (E(X)): {expected_value * 100:.2f}%")

    # Print the actual observed change
    print(f"Actual Observed Change: {observed_change * 100:.2f}%")

    # Optional: Print the 2D PDF slice
    print("2D PDF Slice:")
    print(pdf_matrix[0])



# Example Usage with Debugging
file_path = "data/processed_data.csv"
data = pd.read_csv(file_path)

# Grid Definitions
x_grid = np.linspace(-0.2, 0.2, 200)  # Percentage change grid
time_grid = np.linspace(0, 60, 100)

# Example parameters for a specific point
example_row = data.sample(1).iloc[0]  # Randomly sample one row for testing
print(example_row)
observed_change = example_row["pct_change"] / 100  # Convert to fraction
mu_0 = example_row["mean"]
sigma_0 = example_row["volatility"]
alpha = example_row["skewness"]
delta = example_row["mean_rate"]
beta = example_row["volatility_growth"]


# Run the validation with debugging and plotting
validate_single_point_debug_and_plot(
    observed_change=observed_change,
    pdf_function=commercial_pdf_with_time_decay,
    x_grid=x_grid,
    time_grid=time_grid,
    mu_0=mu_0,
    sigma_0=sigma_0,
    alpha=alpha,
    delta=delta,
    beta=beta,
    lambda_decay=0.0,
    k=1.0,
)