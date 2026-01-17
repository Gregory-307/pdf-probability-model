"""
Generate showcase outputs for temporalpdf.

Run this script to regenerate all PNG/HTML visualizations in the showcase folder.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import temporalpdf as tpdf

SHOWCASE_DIR = Path(__file__).parent


def main():
    print("Generating showcase outputs...")
    print("=" * 60)

    # Common parameters
    nig = tpdf.NIG()
    params = tpdf.NIGParameters(
        mu=0.001,
        delta=0.02,
        alpha=15.0,
        beta=-2.0,
    )

    # 1. NIG vs Normal comparison
    print("1. NIG vs Normal comparison...")
    x = np.linspace(-0.15, 0.15, 500)
    pdf = nig.pdf(x, 0, params)
    normal_pdf = scipy_stats.norm.pdf(
        x, nig.mean(0, params), np.sqrt(nig.variance(0, params))
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x * 100, pdf, "b-", linewidth=2.5, label="NIG")
    axes[0].plot(x * 100, normal_pdf, "r--", linewidth=2.5, label="Normal (same μ,σ)")
    axes[0].set_xlabel("Return (%)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("NIG vs Normal: Full Distribution", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(x * 100, pdf, "b-", linewidth=2.5, label="NIG")
    axes[1].semilogy(x * 100, normal_pdf, "r--", linewidth=2.5, label="Normal")
    axes[1].set_xlabel("Return (%)", fontsize=12)
    axes[1].set_ylabel("Density (log scale)", fontsize=12)
    axes[1].set_title("NIG vs Normal: Tail Behavior", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SHOWCASE_DIR / "nig_vs_normal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: nig_vs_normal.png")

    # 2. VaR and CVaR visualization
    print("2. VaR and CVaR visualization...")
    var_95 = tpdf.var(nig, params, alpha=0.05)
    cvar_95 = tpdf.cvar(nig, params, alpha=0.05, n_samples=50000)

    x = np.linspace(-0.15, 0.15, 1000)
    pdf = nig.pdf(x, 0, params)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x * 100, pdf, "b-", linewidth=2.5, label="NIG PDF")
    ax.axvline(
        -var_95 * 100,
        color="orange",
        linestyle="--",
        linewidth=2.5,
        label=f"VaR 95% = {var_95*100:.1f}%",
    )

    var_x = -var_95
    tail_mask = x <= var_x
    ax.fill_between(
        x[tail_mask] * 100,
        pdf[tail_mask],
        alpha=0.3,
        color="red",
        label=f"CVaR region (E={cvar_95*100:.1f}%)",
    )

    ax.set_xlabel("Return (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("VaR and CVaR Visualization", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(-15, 15)

    plt.tight_layout()
    plt.savefig(SHOWCASE_DIR / "var_cvar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: var_cvar.png")

    # 3. Time-evolving distribution (3D surface)
    print("3. Time-evolving distribution (3D surface)...")
    params_evolving = tpdf.NIGParameters(
        mu=0.001,
        delta=0.02,
        alpha=15.0,
        beta=-2.0,
        mu_drift=0.0002,
        delta_growth=0.05,
    )

    result = tpdf.evaluate(
        nig,
        params_evolving,
        value_range=(-0.15, 0.15),
        time_range=(0, 30),
        value_points=100,
        time_points=50,
    )

    plotter = tpdf.PDFPlotter()
    fig = plotter.surface_3d(result, title="NIG Distribution Over 30-Day Horizon")
    plt.savefig(SHOWCASE_DIR / "time_evolution_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: time_evolution_3d.png")

    # 4. Distribution comparison
    print("4. Distribution comparison...")
    x = np.linspace(-0.15, 0.15, 500)

    normal = tpdf.Normal()
    normal_params = tpdf.NormalParameters(mu_0=0, sigma_0=0.03, delta=0, beta=0)

    student = tpdf.StudentT()
    student_params = tpdf.StudentTParameters(mu_0=0, sigma_0=0.03, nu=5, delta=0, beta=0)

    skew = tpdf.SkewNormal()
    skew_params = tpdf.SkewNormalParameters(mu_0=0, sigma_0=0.03, alpha=3, delta=0, beta=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        x * 100, normal.pdf(x, 0, normal_params), "b-", label="Normal", linewidth=2.5
    )
    axes[0].plot(
        x * 100, student.pdf(x, 0, student_params), "r--", label="Student's t (ν=5)", linewidth=2.5
    )
    axes[0].plot(
        x * 100, skew.pdf(x, 0, skew_params), "g:", label="Skew-Normal (α=3)", linewidth=2.5
    )
    axes[0].set_xlabel("Value (%)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Distribution Comparison", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(
        x * 100, normal.pdf(x, 0, normal_params), "b-", label="Normal", linewidth=2.5
    )
    axes[1].semilogy(
        x * 100, student.pdf(x, 0, student_params), "r--", label="Student's t", linewidth=2.5
    )
    axes[1].semilogy(
        x * 100, skew.pdf(x, 0, skew_params), "g:", label="Skew-Normal", linewidth=2.5
    )
    axes[1].set_xlabel("Value (%)", fontsize=12)
    axes[1].set_ylabel("Density (log)", fontsize=12)
    axes[1].set_title("Tail Comparison", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SHOWCASE_DIR / "distribution_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: distribution_comparison.png")

    # 5. Real data histogram (if available)
    print("5. Real data histogram...")
    data_path = Path(__file__).parent.parent / "data" / "sample_returns.csv"
    if data_path.exists():
        import pandas as pd

        df = pd.read_csv(data_path)
        returns = df["target"].dropna().values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(returns, bins=50, density=True, alpha=0.7, edgecolor="black", color="steelblue")
        ax.set_xlabel("Return (%)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Nordic Stock Returns (n={len(returns):,})", fontsize=14)
        ax.grid(alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Mean: {np.mean(returns):.3f}%\n"
            f"Std: {np.std(returns):.3f}%\n"
            f"Skew: {scipy_stats.skew(returns):.3f}\n"
            f"Kurt: {scipy_stats.kurtosis(returns):.3f}"
        )
        ax.text(
            0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()
        plt.savefig(SHOWCASE_DIR / "real_data_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved: real_data_histogram.png")
    else:
        print("   Skipped: sample_returns.csv not found")

    # 6. Interactive 3D plot (HTML)
    print("6. Interactive 3D plot (HTML)...")
    try:
        interactive = tpdf.InteractivePlotter(colorscale="Viridis")
        fig = interactive.surface_3d(result, title="30-Day Forecast - Drag to Rotate!")
        fig.write_html(str(SHOWCASE_DIR / "interactive_3d.html"))
        print("   Saved: interactive_3d.html")
    except Exception as e:
        print(f"   Skipped (plotly not available): {e}")

    print()
    print("=" * 60)
    print("Showcase generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
