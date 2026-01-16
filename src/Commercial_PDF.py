# Core Libraries
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
from scipy.stats import skew  # Statistical skewness calculation
from sklearn.linear_model import LinearRegression  # Linear regression for trends


def expected_value_from_pdf(x, pdf_matrix):
    """
    Calculate the expected value from a 3D predictive PDF matrix.

    Parameters:
        x (array-like): Grid of percentage change values.
        pdf_matrix (2D array): The predictive PDF values, where each row corresponds to a time step.

    Returns:
        array: Expected values for each time step.
    """
    # Compute the expected value for each time step by integrating x * PDF(x) over x
    expected_values = np.dot(pdf_matrix, x) / np.sum(pdf_matrix, axis=1)
    return expected_values


def total_expected_value(time_grid, expected_values):
    """
    Calculate the total expected value over the given timeframe.

    Parameters:
        time_grid (array-like): The time steps (e.g., days).
        expected_values (array-like): Expected values at each time step.

    Returns:
        float: Total expected value over the timeframe.
    """
    # Compute time intervals (assuming uniform or non-uniform spacing)
    time_intervals = np.diff(time_grid, prepend=0)
    # Compute the weighted sum of expected values
    total_ex = np.sum(expected_values * time_intervals)
    return total_ex

# # Compute total expected value for the given timeframe
# total_ex = total_expected_value(time_grid, expected_values)
# total_ex


def cumulative_expected_value_corrected(x, pdf_matrix, time_grid): #<----------------------------------------------------------------------------------
    """
    Calculate the cumulative expected value E(X) over time from the predictive PDF.

    Parameters:
        x (array-like): Grid of percentage change values (e.g., -0.2 to 0.2).
        pdf_matrix (2D array): Evaluated PDF values from commercial_pdf_with_time_decay.
        time_grid (array-like): Time grid corresponding to the PDF matrix.

    Returns:
        float: The cumulative expected value E(X) over the given timeframe.
    """
    # Ensure x is treated as fractions (divide by 100 for percentage scale consistency)

    x = np.array(x) / 100 / 60  # Convert percentage changes to fractions << THIS SHOULD ACTUALLY BE AT PDF LEVEL BUT I AM LAZY

    time_intervals = np.diff(time_grid, prepend=0)  # Time intervals between points

    # Calculate the expected value at each time step (E(X_t))
    expected_values = np.dot(pdf_matrix, x) / np.sum(pdf_matrix, axis=1)

    # Compute the cumulative expected value over time (sum of E(X_t) weighted by time intervals)
    cumulative_ex = np.sum(expected_values * time_intervals)
    return cumulative_ex

# Calculate cumulative E(X) for one of the scenarios
# cumulative_ex = calculate_cumulative_expected_value(x_grid, pdf_matrix_with_decay)

# # Display the result
# cumulative_ex


# # Compute expected values for the example 3D PDF
# expected_values = expected_value_from_pdf(x_grid, pdf_3d_values)

# # Plot the expected values over time
# plt.figure(figsize=(10, 6))
# plt.plot(time_grid, expected_values, label="Expected Value", color="blue")
# plt.title("Expected Value Over Time")
# plt.xlabel("Time (days)")
# plt.ylabel("Expected Percentage Change")
# plt.legend()
# plt.grid()
# plt.show()


# pdf_matrix, x_grid, time_grid = commercial_pdf_with_grids(
#     x=np.linspace(-0.2, 0.2, 200),  # Percentage change grid
#     time_grid=np.linspace(0, 60, 100),  # Continuous time grid (0 to 60 days)
#     mu_0=0.0,  # Initial mean
#     sigma_0=0.02,  # Initial volatility
#     alpha=0.5,  # Skewness
#     delta=0.002,  # Mean rate
#     beta=0.01,  # Volatility growth
#     k=1.0  # Tail sharpness
# )

def commercial_pdf_with_time_decay(mu_0, sigma_0, alpha, delta, beta, x=np.linspace(-0.2, 0.2, 200), time_grid=np.linspace(0, 60, 100), lambda_decay=0.0, k=1.0): #<----------------------------------------------------------------------------------
    """
    Commercial-grade PDF function with separate x-grid and time-grid handling, including time decay.

    Parameters:
        x (array-like): Grid of percentage change values.
        time_grid (array-like): Grid of time values.
        mu_0 (float): Initial mean at the start of the time series.
        sigma_0 (float): Initial volatility at the start of the time series.
        alpha (float): Skewness parameter.
        delta (float): Mean rate (linear trend in the mean over time).
        beta (float): Volatility growth rate over time.
        lambda_decay (float): Time decay parameter to reduce prediction confidence.
        k (float): Tail sharpness parameter.

    Returns:
        tuple: (pdf_matrix, x, time_grid), where:
            - pdf_matrix (2D array): Evaluated PDF values with time decay.
            - x (array): The input x-grid.
            - time_grid (array): The input time-grid.
    """
    time_grid = np.array(time_grid)  # Ensure time grid is a numpy array
    pdf_matrix = np.zeros((len(time_grid), len(x)))  # Initialize the PDF matrix
    
    for i, t in enumerate(time_grid):
        mu_t = mu_0 + delta * t  # Time-dependent mean
        sigma_t = sigma_0 * (1 + beta * t)  # Time-dependent volatility
        
        # Compute the unnormalized PDF
        base = np.exp(-np.abs(x - mu_t)**(1 + k) / (2 * sigma_t**2))
        skew = 1 + alpha * (x - mu_t)
        skew = np.maximum(skew, 0)  # Ensure non-negativity
        decay = np.exp(-lambda_decay * t)  # Time decay factor
        raw_pdf = base * skew * decay
        
        # Normalize the PDF for this time step
        normalization = np.trapz(raw_pdf, x)
        normalization = normalization if normalization > 0 else 1
        pdf_matrix[i, :] = raw_pdf / normalization  # Store the normalized PDF
    
    return pdf_matrix, x, time_grid


# # Example usage with time decay included
# pdf_matrix_with_decay, x_grid, time_grid = commercial_pdf_with_time_decay(
#     x=np.linspace(-0.2, 0.2, 200),  # Percentage change grid
#     time_grid=np.linspace(0, 60, 100),  # Continuous time grid (0 to 60 days)
#     mu_0=0.0,  # Initial mean
#     sigma_0=0.02,  # Initial volatility
#     alpha=0.5,  # Skewness
#     delta=0.002,  # Mean rate
#     beta=0.01,  # Volatility growth
#     lambda_decay=0.05,  # Time decay
#     k=1.0  # Tail sharpness
# )

# # Visualize the updated 3D PDF with time decay included
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# X, T = np.meshgrid(x_grid, time_grid)
# ax.plot_surface(T, X, pdf_matrix_with_decay, cmap='viridis', edgecolor='none')
# ax.set_title("3D Predictive PDF (With Time Decay)")
# ax.set_xlabel("Time (days)")
# ax.set_ylabel("Percentage Change")
# ax.set_zlabel("Probability Density")
# plt.show()


def commercial_pdf_with_grids(mu_0, sigma_0, alpha, delta, beta, x=np.linspace(-0.2, 0.2, 200), time_grid=np.linspace(0, 60, 100), k=1.0):
    """
    Commercial-grade PDF function with separate x-grid and time-grid handling.

    Parameters:
        x (array-like): Grid of percentage change values.
        time_grid (array-like): Grid of time values.
        mu_0 (float): Initial mean at the start of the time series.
        sigma_0 (float): Initial volatility at the start of the time series.
        alpha (float): Skewness parameter.
        delta (float): Mean rate (linear trend in the mean over time).
        beta (float): Volatility growth rate over time.
        k (float): Tail sharpness parameter.

    Returns:
        tuple: (pdf_matrix, x, time_grid), where:
            - pdf_matrix (2D array): Evaluated PDF values.
            - x (array): The input x-grid.
            - time_grid (array): The input time-grid.
    """
    # Ensure time grid is a numpy array
    time_grid = np.array(time_grid)
    
    # Initialize the PDF matrix
    pdf_matrix = np.zeros((len(time_grid), len(x)))
    
    # Evaluate the PDF for each time point in the grid
    for i, t in enumerate(time_grid):
        # Compute time-dependent mean and volatility
        mu_t = mu_0 + delta * t  # Time-dependent mean
        sigma_t = sigma_0 * (1 + beta * t)  # Time-dependent volatility
        
        # Compute the unnormalized PDF
        base = np.exp(-np.abs(x - mu_t)**(1 + k) / (2 * sigma_t**2))  # Core PDF
        skew = 1 + alpha * (x - mu_t)  # Skewness adjustment
        skew = np.maximum(skew, 0)  # Ensure non-negativity
        raw_pdf = base * skew  # Combined PDF (no time decay for now)
        
        # Normalize the PDF for this time step
        normalization = np.trapz(raw_pdf, x)  # Integrate over x
        normalization = normalization if normalization > 0 else 1  # Avoid division by zero
        pdf_matrix[i, :] = raw_pdf / normalization  # Store the normalized PDF
    
    return pdf_matrix, x, time_grid


def commercial_3d_pdf(x, time_grid, mu_0, sigma_0, alpha, delta, beta, lambda_decay, k=1.0):
    """
    Commercial-grade 3D PDF function for predictive modeling over a continuous time grid.
    
    Parameters:
        x (array-like): Grid of percentage change values to evaluate the PDF.
        time_grid (array-like): Continuous grid of time values (e.g., days since t=0).
        mu_0 (float): Initial mean at the start of the time series.
        sigma_0 (float): Initial volatility at the start of the time series.
        alpha (float): Skewness parameter.
        delta (float): Mean rate (linear trend in the mean over time).
        beta (float): Volatility growth rate over time.
        lambda_decay (float): Time decay parameter to reduce prediction confidence.
        k (float): Tail sharpness parameter.
    
    Returns:
        2D array: The evaluated 3D PDF values, with shape (len(time_grid), len(x)).
    """
    # Ensure time grid is a numpy array
    time_grid = np.array(time_grid)
    
    # Initialize the PDF matrix
    pdf_matrix = np.zeros((len(time_grid), len(x)))
    
    # Evaluate the PDF for each time point in the grid
    for i, t in enumerate(time_grid):
        # Compute time-dependent mean and volatility
        mu_t = mu_0 + delta * t  # Time-dependent mean
        sigma_t = sigma_0 * (1 + beta * t)  # Time-dependent volatility
        
        # Compute the unnormalized PDF
        base = np.exp(-np.abs(x - mu_t)**(1 + k) / (2 * sigma_t**2))  # Core PDF
        skew = 1 + alpha * (x - mu_t)  # Skewness adjustment
        skew = np.maximum(skew, 0)  # Ensure non-negativity
        decay = np.exp(-lambda_decay * t)  # Time decay factor
        raw_pdf = base * skew * decay  # Combined PDF
        
        # Normalize the PDF for this time step
        normalization = np.trapz(raw_pdf, x)  # Integrate over x
        normalization = normalization if normalization > 0 else 1  # Avoid division by zero
        pdf_matrix[i, :] = raw_pdf / normalization  # Store the normalized PDF
    
    return pdf_matrix


# if __name__ == "__main__":
#     # Example usage for 3D PDF
#     x_grid = np.linspace(-0.2, 0.2, 200)  # Percentage change grid
#     time_grid = np.linspace(0, 60, 100)  # Continuous time grid (0 to 60 days)
#     mu_0 = 0.0  # Initial mean
#     sigma_0 = 0.02  # Initial volatility
#     alpha = 0.5  # Skewness
#     delta = 0.002  # Mean rate
#     beta = 0.01  # Volatility growth
#     lambda_decay = 0.05  # Time decay
#     k = 1.0  # Tail sharpness

#     # Compute the 3D PDF
#     pdf_3d_values = commercial_3d_pdf(x_grid, time_grid, mu_0, sigma_0, alpha, delta, beta, lambda_decay, k)

#     # Visualize the 3D PDF
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     X, T = np.meshgrid(x_grid, time_grid)
#     ax.plot_surface(T, X, pdf_3d_values, cmap='viridis', edgecolor='none')
#     ax.set_title("3D Predictive PDF (Probability Cone)")
#     ax.set_xlabel("Time (days)")
#     ax.set_ylabel("Percentage Change")
#     ax.set_zlabel("Probability Density")
#     plt.show()


# Update the code to calculate cumulative E(X) for each scenario and annotate the plots

# Increase padding between subplots for better visualization

if __name__ == "__main__":
    scenarios = [
        {
            "title": "Rising Stock (Predictive Cone)",
            "params": {"mu_0": 0.0, "delta": 0.5, "sigma_0": 7, "beta": 0.06, "alpha": 0.009, "lambda_decay": 1, "k": 1.0}
        },
        {
            "title": "Falling Stock (Predictive Cone)",
            "params": {"mu_0": 0.0, "delta": -0.5, "sigma_0": 7, "beta": 0.06, "alpha": -0.009, "lambda_decay": 1, "k": 1.0}
        },
        {
            "title": "Low Volatility Stock (Stable Cone)",
            "params": {"mu_0": 0.0, "delta": 0.0, "sigma_0": 4, "beta": 0.05, "alpha": 0.0, "lambda_decay": 1, "k": 1.0}
        },
        {
            "title": "High Volatility Stock (Wide Cone)",
            "params": {"mu_0": 0.0, "delta": 0.0, "sigma_0": 10, "beta": 0.03, "alpha": 0.0, "lambda_decay": 1, "k": 0.5}
        }
    ]

    x_grid = np.linspace(-100, 100, 50)  # Percentage change grid
    #print(x_grid)
    time_grid = np.linspace(0, 60, 100)  # Continuous time grid (0 to 60 days) # ----------------------------- This is the problem

    fig, axes = plt.subplots(2, 2, figsize=(20, 15), subplot_kw={"projection": "3d"})

    for ax, scenario in zip(axes.flatten(), scenarios):
        pdf_matrix, x_grid, time_grid = commercial_pdf_with_time_decay(
            x=x_grid, time_grid=time_grid, **scenario["params"]
        )
        cumulative_ex = cumulative_expected_value_corrected(x_grid, pdf_matrix, time_grid)
        X, T = np.meshgrid(x_grid, time_grid)
        ax.plot_surface(T, X, pdf_matrix, cmap='viridis', edgecolor='none')
        ax.set_title(f"{scenario['title']}\nCumulative E(X): {cumulative_ex * 100:.2f}%")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Percentage Change")
        ax.set_zlabel("Probability Density")

    # Adjust padding for better spacing
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.show()

