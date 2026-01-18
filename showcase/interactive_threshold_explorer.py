"""
Interactive Threshold Explorer

This script lets you step through different probability thresholds
to see exactly what the distribution tells you at each level.

Usage:
    python interactive_threshold_explorer.py [--asset btc|sp500|eurusd]

Example:
    python interactive_threshold_explorer.py --asset sp500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import temporalpdf as tpdf

# Setup
DATA_DIR = Path(__file__).parent.parent / "data"
nig = tpdf.NIG()


def fit_nig_mle(data):
    """Fit NIG distribution via maximum likelihood."""
    x0 = [np.mean(data), np.log(np.std(data) + 0.01), np.log(5.0), 0.0]
    def nll(theta):
        mu, log_delta, log_alpha, beta_raw = theta
        delta, alpha = np.exp(log_delta), np.exp(log_alpha)
        beta = alpha * np.tanh(beta_raw)
        try:
            params = tpdf.NIGParameters(mu=mu, delta=delta, alpha=alpha, beta=beta)
            pdf_vals = nig.pdf(data, 0, params)
            return -np.sum(np.log(np.maximum(pdf_vals, 1e-300)))
        except:
            return 1e10
    result = optimize.minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 1000})
    mu, log_delta, log_alpha, beta_raw = result.x
    return tpdf.NIGParameters(
        mu=mu,
        delta=np.exp(log_delta),
        alpha=np.exp(log_alpha),
        beta=np.exp(log_alpha) * np.tanh(beta_raw)
    )


def load_data(asset: str) -> tuple[np.ndarray, str]:
    """Load data for specified asset."""
    files = {
        'btc': ('crypto_returns.csv', 'BTC (Crypto)'),
        'sp500': ('equity_returns.csv', 'S&P 500 (Equity)'),
        'eurusd': ('forex_returns.csv', 'EUR/USD (Forex)'),
    }

    if asset.lower() not in files:
        raise ValueError(f"Unknown asset: {asset}. Choose from: {list(files.keys())}")

    filename, display_name = files[asset.lower()]
    df = pd.read_csv(DATA_DIR / filename)
    return df['return_pct'].values, display_name


def create_step_through_figure(returns: np.ndarray, asset_name: str,
                                thresholds: list[float] = None):
    """Create a step-through visualization of the distribution."""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    # Fit NIG distribution
    params = fit_nig_mle(returns)

    # Generate samples for percentiles
    rng = np.random.default_rng(42)
    samples = nig.sample(100000, 0, params, rng=rng)

    # Create figure
    n_rows = (len(thresholds) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    axes = axes.flat

    x = np.linspace(-6, 6, 500)
    pdf_vals = nig.pdf(x, 0, params)

    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(thresholds)))

    for ax, p, color in zip(axes, thresholds, colors):
        threshold = np.percentile(samples, p * 100)

        # Plot distribution
        ax.plot(x, pdf_vals, 'k-', lw=2)
        ax.fill_between(x, pdf_vals, alpha=0.2, color='gray')

        # Highlight region
        ax.fill_between(x, pdf_vals, where=(x < threshold), alpha=0.6, color=color)

        # Mark threshold
        ax.axvline(threshold, color='black', lw=2, ls='--')
        ax.text(threshold + 0.15, ax.get_ylim()[1]*0.85, f'{threshold:+.2f}%',
                fontweight='bold', fontsize=11)

        # Title with interpretation
        if p <= 0.10:
            interp = f"Only {p*100:.0f}% chance of losing more than {abs(threshold):.2f}%"
        elif p >= 0.90:
            interp = f"{p*100:.0f}% chance return is below {threshold:+.2f}%"
        else:
            interp = f"{p*100:.0f}th percentile"

        ax.set_title(f'{p*100:.0f}% Threshold\n{interp}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)

    # Hide unused axes
    for ax in axes[len(thresholds):]:
        ax.axis('off')

    fig.suptitle(f'STEP-THROUGH: {asset_name} Return Distribution\n'
                 f'NIG Parameters: μ={params.mu:.3f}, δ={params.delta:.3f}, '
                 f'α={params.alpha:.2f}, β={params.beta:.3f}',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()

    return fig


def print_threshold_table(returns: np.ndarray, asset_name: str):
    """Print a table of thresholds and their interpretations."""
    params = fit_nig_mle(returns)
    rng = np.random.default_rng(42)
    samples = nig.sample(100000, 0, params, rng=rng)

    print(f"\n{asset_name} DISTRIBUTION THRESHOLDS")
    print("=" * 70)
    print(f"{'Probability':>12} {'Return':>12} {'Interpretation'}")
    print("-" * 70)

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    for p in thresholds:
        val = np.percentile(samples, p * 100)

        if p <= 0.05:
            interp = f"VaR({p*100:.0f}%): Only {p*100:.0f}% chance of loss > {abs(val):.2f}%"
        elif p < 0.5:
            interp = f"{p*100:.0f}% of returns are below this"
        elif p == 0.5:
            interp = "Median return (50/50 above/below)"
        elif p <= 0.95:
            interp = f"{p*100:.0f}% of returns are below this"
        else:
            interp = f"Only {(1-p)*100:.0f}% chance of return above {val:+.2f}%"

        print(f"{p*100:>10.0f}% {val:>+10.2f}%   {interp}")

    print("-" * 70)
    print(f"\nSummary Statistics:")
    print(f"  Expected Value: {np.mean(samples):+.3f}%")
    print(f"  Std Dev: {np.std(samples):.3f}%")
    print(f"  Skewness: {np.mean((samples - np.mean(samples))**3) / np.std(samples)**3:.3f}")
    print(f"  Kurtosis: {np.mean((samples - np.mean(samples))**4) / np.std(samples)**4 - 3:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Explore distribution thresholds interactively')
    parser.add_argument('--asset', type=str, default='sp500',
                        choices=['btc', 'sp500', 'eurusd'],
                        help='Asset to analyze (default: sp500)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to file (e.g., output.png)')
    args = parser.parse_args()

    # Load data
    returns, asset_name = load_data(args.asset)
    print(f"\nLoaded {len(returns):,} days of {asset_name} returns")
    print(f"Mean: {np.mean(returns):+.3f}%, Std: {np.std(returns):.3f}%")

    # Print table
    print_threshold_table(returns, asset_name)

    # Create figure
    print("\nGenerating visualization...")
    fig = create_step_through_figure(returns, asset_name)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {args.save}")

    plt.show()


if __name__ == '__main__':
    main()
