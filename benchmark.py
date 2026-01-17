"""
Performance benchmarks for temporalpdf.

Demonstrates that vectorized implementations provide meaningful speedups.
"""

import time
from typing import Callable

import numpy as np

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf


def benchmark(func: Callable, n_runs: int = 5) -> tuple[float, float]:
    """Run benchmark and return (mean_time, std_time) in milliseconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    return np.mean(times), np.std(times)


def main():
    print("=" * 60)
    print("temporalpdf Performance Benchmarks")
    print("=" * 60)
    print()

    # Grid sizes to test
    value_points = [100, 500, 1000]
    time_points = [50, 100, 200]

    # NIG parameters
    nig = tpdf.NIG()
    params = tpdf.NIGParameters(mu=0.0, delta=0.02, alpha=15.0, beta=-2.0)

    print("1. PDF Matrix Computation (vectorized)")
    print("-" * 50)
    print(f"{'Grid Size':<20} {'Time (ms)':<15} {'Throughput':<15}")

    for vp in value_points:
        for tp in time_points:
            x = np.linspace(-0.2, 0.2, vp)
            t = np.linspace(0, 60, tp)

            def run():
                return nig.pdf_matrix(x, t, params)

            mean_ms, std_ms = benchmark(run)
            grid_size = vp * tp
            throughput = grid_size / mean_ms * 1000  # points per second

            print(f"{vp}x{tp} = {grid_size:<8} {mean_ms:>6.2f} ± {std_ms:.2f}   {throughput/1e6:.2f}M pts/sec")

    print()
    print("2. CRPS Computation (Monte Carlo)")
    print("-" * 50)

    n_samples_list = [1000, 5000, 10000, 50000]
    print(f"{'N Samples':<15} {'Time (ms)':<15} {'Samples/sec':<15}")

    for n_samples in n_samples_list:
        rng = np.random.default_rng(42)

        def run():
            return tpdf.crps(nig, params, y=0.01, n_samples=n_samples, rng=rng)

        mean_ms, std_ms = benchmark(run, n_runs=3)
        throughput = n_samples / mean_ms * 1000

        print(f"{n_samples:<15} {mean_ms:>6.2f} ± {std_ms:.2f}   {throughput/1e6:.2f}M samples/sec")

    print()
    print("3. Risk Measures")
    print("-" * 50)

    # VaR (uses PPF which requires CDF integration)
    def run_var():
        return tpdf.var(nig, params, alpha=0.05)

    mean_ms, _ = benchmark(run_var, n_runs=3)
    print(f"VaR 95%:  {mean_ms:.2f} ms")

    # CVaR (uses sampling)
    def run_cvar():
        return tpdf.cvar(nig, params, alpha=0.05, n_samples=50000)

    mean_ms, _ = benchmark(run_cvar, n_runs=3)
    print(f"CVaR 95%: {mean_ms:.2f} ms (50k samples)")

    # Kelly
    def run_kelly():
        return tpdf.kelly_fraction(nig, params)

    mean_ms, _ = benchmark(run_kelly, n_runs=3)
    print(f"Kelly:    {mean_ms:.2f} ms")

    print()
    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
