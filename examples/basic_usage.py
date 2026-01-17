"""
Basic usage example for temporalpdf.

This example demonstrates the core functionality:
1. Creating distributions with different parameter configurations
2. Evaluating PDFs over time
3. Visualizing results with 3D surface plots
"""

import numpy as np

# Import the library
import sys
sys.path.insert(0, "../src")
import temporalpdf as tpdf


def main() -> None:
    """Run basic usage examples."""

    # Example 1: Simple Normal distribution with time evolution
    print("Example 1: Normal Distribution with drift")
    print("-" * 50)

    normal_dist = tpdf.Normal()
    normal_params = tpdf.NormalParameters(
        mu_0=0.0,      # Start centered at 0
        sigma_0=0.05,  # Initial standard deviation of 5%
        delta=0.001,   # Mean drifts by 0.1% per time unit
        beta=0.01,     # Volatility grows by 1% per time unit
    )

    # Evaluate over a grid
    result = tpdf.evaluate(
        normal_dist,
        normal_params,
        value_range=(-0.3, 0.3),
        time_range=(0, 60),
    )

    print(f"Distribution: {result.distribution_name}")
    print(f"Grid shape: {result.shape}")
    print(f"Cumulative E[X]: {result.cumulative_expected_value:.6f}")
    print()

    # Example 2: Generalized Laplace with skewness
    print("Example 2: Generalized Laplace with positive skew")
    print("-" * 50)

    laplace_dist = tpdf.GeneralizedLaplace()
    laplace_params = tpdf.GeneralizedLaplaceParameters(
        mu_0=0.0,
        sigma_0=5.0,
        alpha=0.01,       # Positive skew (right-tailed)
        delta=0.3,        # Rising trend
        beta=0.05,        # Moderate volatility growth
        k=1.0,            # Laplace-like tails
        lambda_decay=0.5, # Some time decay
    )

    result_laplace = tpdf.evaluate(
        laplace_dist,
        laplace_params,
        value_range=(-50, 50),
        time_range=(0, 60),
    )

    print(f"Distribution: {result_laplace.distribution_name}")
    print(f"Cumulative E[X]: {result_laplace.cumulative_expected_value:.4f}")
    print()

    # Example 3: Compare multiple scenarios
    print("Example 3: Multi-scenario comparison")
    print("-" * 50)

    scenarios = [
        ("Rising (Bullish)", tpdf.GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=7, alpha=0.009, delta=0.5, beta=0.06, lambda_decay=1.0
        )),
        ("Falling (Bearish)", tpdf.GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=7, alpha=-0.009, delta=-0.5, beta=0.06, lambda_decay=1.0
        )),
        ("Low Volatility", tpdf.GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=4, alpha=0.0, delta=0.0, beta=0.05, lambda_decay=1.0
        )),
        ("High Volatility", tpdf.GeneralizedLaplaceParameters(
            mu_0=0.0, sigma_0=10, alpha=0.0, delta=0.0, beta=0.03, k=0.5, lambda_decay=1.0
        )),
    ]

    dist = tpdf.GeneralizedLaplace()
    results = []

    for name, params in scenarios:
        result = tpdf.evaluate(
            dist,
            params,
            value_range=(-100, 100),
            time_range=(0, 60),
            value_points=50,
        )
        results.append(result)
        print(f"  {name}: Cumulative E[X] = {result.cumulative_expected_value * 100:.2f}%")

    print()

    # Example 4: Create visualizations
    print("Example 4: Creating visualizations")
    print("-" * 50)

    plotter = tpdf.PDFPlotter(style=tpdf.DEFAULT_STYLE)

    # Single 3D surface
    fig1 = plotter.surface_3d(
        result_laplace,
        title="Generalized Laplace - Rising Scenario",
        xlabel="Time (days)",
        ylabel="Percentage Change",
    )
    plotter.save(fig1, "output_3d_surface.png")
    print("  Saved: output_3d_surface.png")

    # Multi-scenario comparison
    fig2 = plotter.multi_scenario(
        results,
        titles=[name for name, _ in scenarios],
    )
    plotter.save(fig2, "output_multi_scenario.png")
    print("  Saved: output_multi_scenario.png")

    # Expected value over time
    fig3 = plotter.expected_value_over_time(result_laplace)
    plotter.save(fig3, "output_expected_value.png")
    print("  Saved: output_expected_value.png")

    # Heatmap
    fig4 = plotter.heatmap(result_laplace)
    plotter.save(fig4, "output_heatmap.png")
    print("  Saved: output_heatmap.png")

    print()
    print("Done! Check the output PNG files.")


if __name__ == "__main__":
    main()
