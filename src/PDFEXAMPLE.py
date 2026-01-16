import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the predictive PDF
def predictive_cone_pdf(x, t, mu_0, delta, sigma_0, beta, alpha, lambda_decay, k=1.0):
    mu_t = mu_0 + delta * t
    sigma_t = sigma_0 * (1 + beta * t)
    base = np.exp(-np.abs(x - mu_t)**(1 + k) / (2 * sigma_t**2))
    skew = 1 + alpha * (x - mu_t)
    skew = np.maximum(skew, 0)
    decay = np.exp(-lambda_decay * t)
    raw_pdf = base * skew * decay
    normalization = np.trapz(raw_pdf, x, axis=0)
    normalization = np.where(normalization == 0, 1, normalization)
    return raw_pdf / normalization

# Example parameters for scenarios
cone_rising_params = {"mu_0": 0.0, "delta": 0.002, "sigma_0": 0.02, "beta": 0.01, "alpha": 0.5, "lambda_decay": 0.05, "k": 1.0}
cone_falling_params = {"mu_0": 0.0, "delta": -0.002, "sigma_0": 0.02, "beta": 0.01, "alpha": -0.5, "lambda_decay": 0.05, "k": 1.0}
low_vol_params = {"mu_0": 0.0, "delta": 0.0, "sigma_0": 0.005, "beta": 0.002, "alpha": 0.0, "lambda_decay": 0.02, "k": 1.0}
high_vol_params = {"mu_0": 0.0, "delta": 0.0, "sigma_0": 0.05, "beta": 0.02, "alpha": 0.0, "lambda_decay": 0.1, "k": 0.5}

# Visualization for 4 cases
fig, axes = plt.subplots(2, 2, figsize=(20, 15), subplot_kw={"projection": "3d"})
titles = [
    "Rising Stock (Predictive Cone)",
    "Falling Stock (Predictive Cone)",
    "Low Volatility Stock (Stable Cone)",
    "High Volatility Stock (Wide Cone)"
]
parameter_sets = [cone_rising_params, cone_falling_params, low_vol_params, high_vol_params]

for ax, title, params in zip(axes.flatten(), titles, parameter_sets):
    t_values = np.linspace(0, 60, 100)
    x_values = np.linspace(-0.2, 0.2, 200)
    x_mesh, t_mesh = np.meshgrid(x_values, t_values)
    z_scenario = np.zeros_like(t_mesh)
    for i, t in enumerate(t_values):
        z_scenario[i, :] = predictive_cone_pdf(x_values, t, **params)
    ax.plot_surface(t_mesh, x_mesh, z_scenario, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Percentage Change")
    ax.set_zlabel("Probability Density")

plt.tight_layout()
plt.show()
