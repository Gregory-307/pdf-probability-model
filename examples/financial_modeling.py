"""
Financial modeling example for temporalpdf.

This example demonstrates:
1. Extracting distribution coefficients from historical data
2. Creating predictive probability distributions
3. Validating predictions against observed data
4. Visualizing forecasts with confidence bands
"""

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "../src")
import temporalpdf as tpdf


def generate_sample_data(n_days: int = 365, n_tickers: int = 3) -> pd.DataFrame:
    """
    Generate sample stock price data for demonstration.

    In real usage, you would load actual price data from your data source.
    """
    np.random.seed(42)

    data = []
    tickers = [f"STOCK_{i}" for i in range(n_tickers)]

    for ticker in tickers:
        # Random walk with drift
        drift = np.random.uniform(-0.0005, 0.001)
        volatility = np.random.uniform(0.01, 0.03)

        prices = [100.0]
        for _ in range(n_days - 1):
            return_ = drift + volatility * np.random.randn()
            prices.append(prices[-1] * (1 + return_))

        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        for date, price in zip(dates, prices):
            data.append({
                "date": date,
                "ticker": ticker,
                "close": price,
            })

    return pd.DataFrame(data)


def main() -> None:
    """Run financial modeling example."""

    print("Financial Modeling with temporalpdf")
    print("=" * 60)
    print()

    # Step 1: Load or generate data
    print("Step 1: Loading sample data")
    print("-" * 40)

    df = generate_sample_data(n_days=200, n_tickers=3)
    print(f"  Data shape: {df.shape}")
    print(f"  Tickers: {df['ticker'].unique().tolist()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Step 2: Extract coefficients
    print("Step 2: Extracting distribution coefficients")
    print("-" * 40)

    extractor = tpdf.RollingCoefficientExtractor()
    config = tpdf.ExtractionConfig(
        value_column="close",
        time_column="date",
        group_column="ticker",
        horizon=60,                # 60-day rolling window
        volatility_window=7,       # 7-day volatility window
        pct_change_multiplier=100, # Convert to percentage
    )

    df_with_coeffs = extractor.extract(df, config)
    print(f"  Coefficients added: {tpdf.RollingCoefficientExtractor.COEFFICIENT_COLUMNS}")
    print(f"  Rows after extraction: {len(df_with_coeffs)}")
    print()

    # Show sample coefficients for one ticker
    sample_ticker = "STOCK_0"
    sample = df_with_coeffs[df_with_coeffs["ticker"] == sample_ticker].iloc[-1]
    print(f"  Latest coefficients for {sample_ticker}:")
    print(f"    mean (mu_0): {sample['mean']:.4f}")
    print(f"    volatility (sigma_0): {sample['volatility']:.4f}")
    print(f"    skewness (alpha): {sample['skewness']:.4f}")
    print(f"    mean_rate (delta): {sample['mean_rate']:.4f}")
    print(f"    volatility_growth (beta): {sample['volatility_growth']:.6f}")
    print()

    # Step 3: Create predictive distribution
    print("Step 3: Creating predictive distribution")
    print("-" * 40)

    # Convert coefficients to parameters
    params = extractor.coefficients_from_row(
        sample,
        k=1.0,           # Laplace-like tails
        lambda_decay=0.5 # Moderate confidence decay
    )

    dist = tpdf.GeneralizedLaplace()

    # Evaluate over 60-day prediction horizon
    result = tpdf.evaluate(
        dist,
        params,
        value_range=(-30, 30),  # -30% to +30% change
        time_range=(0, 60),     # 60-day horizon
        value_points=100,
        time_points=60,
    )

    print(f"  Distribution: {result.distribution_name}")
    print(f"  Cumulative E[X]: {result.cumulative_expected_value:.4f}")
    print(f"  E[X] at t=30: {result.expected_value_at_time(30):.4f}")
    print(f"  E[X] at t=60: {result.expected_value_at_time(60):.4f}")
    print()

    # Step 4: Compare different distribution types
    print("Step 4: Comparing distribution types")
    print("-" * 40)

    distributions = [
        ("Generalized Laplace", tpdf.GeneralizedLaplace(), tpdf.GeneralizedLaplaceParameters(
            mu_0=sample["mean"],
            sigma_0=sample["volatility"],
            alpha=sample["skewness"],
            delta=sample["mean_rate"],
            beta=sample["volatility_growth"],
        )),
        ("Normal", tpdf.Normal(), tpdf.NormalParameters(
            mu_0=sample["mean"],
            sigma_0=sample["volatility"],
            delta=sample["mean_rate"],
            beta=sample["volatility_growth"],
        )),
        ("Student's t (nu=5)", tpdf.StudentT(), tpdf.StudentTParameters(
            mu_0=sample["mean"],
            sigma_0=sample["volatility"],
            nu=5.0,  # Heavy tails
            delta=sample["mean_rate"],
            beta=sample["volatility_growth"],
        )),
    ]

    results = []
    for name, dist, params in distributions:
        res = tpdf.evaluate(
            dist,
            params,
            value_range=(-30, 30),
            time_range=(0, 60),
        )
        results.append(res)
        print(f"  {name}: Cumulative E[X] = {res.cumulative_expected_value:.4f}")

    print()

    # Step 5: Visualizations
    print("Step 5: Creating visualizations")
    print("-" * 40)

    plotter = tpdf.PDFPlotter(style=tpdf.PUBLICATION_STYLE)

    # 3D surface for primary model
    fig1 = plotter.surface_3d(
        result,
        title=f"Predictive Distribution: {sample_ticker}",
        xlabel="Days Ahead",
        ylabel="Percentage Change (%)",
    )
    plotter.save(fig1, "output_financial_3d.png", dpi=300)
    print("  Saved: output_financial_3d.png")

    # Distribution comparison at t=30
    fig2 = plotter.compare_distributions(
        results,
        time_point=30,
        labels=[name for name, _, _ in distributions],
        title="Distribution Comparison at t=30 days",
    )
    plotter.save(fig2, "output_financial_comparison.png")
    print("  Saved: output_financial_comparison.png")

    # Confidence bands
    fig3 = plotter.confidence_bands(
        result,
        confidence_levels=(0.5, 0.8, 0.95),
        title=f"Forecast Confidence Bands: {sample_ticker}",
        xlabel="Days Ahead",
        ylabel="Percentage Change (%)",
    )
    plotter.save(fig3, "output_financial_confidence.png")
    print("  Saved: output_financial_confidence.png")

    # Expected value trajectory
    fig4 = plotter.expected_value_over_time(
        result,
        title=f"Expected Return Trajectory: {sample_ticker}",
        xlabel="Days Ahead",
        ylabel="Expected Change (%)",
    )
    plotter.save(fig4, "output_financial_trajectory.png")
    print("  Saved: output_financial_trajectory.png")

    print()
    print("Done! Check the output PNG files.")


if __name__ == "__main__":
    main()
