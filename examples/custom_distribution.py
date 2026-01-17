"""
Example: Creating a custom distribution.

This example shows how to extend temporalpdf with your own
distribution implementations.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

import sys
sys.path.insert(0, "../src")
import temporalpdf as tpdf
from temporalpdf.core.distribution import TimeEvolvingDistribution, DistributionParameters


# Step 1: Define your parameter class
@dataclass(frozen=True)
class CauchyParameters(DistributionParameters):
    """
    Parameters for time-evolving Cauchy distribution.

    The Cauchy distribution has very heavy tails (no defined mean or variance).
    Useful for modeling data with extreme outliers.

    Attributes:
        x0: Location parameter (median)
        gamma_0: Initial scale parameter (half-width at half-maximum)
        delta: Location drift rate
        beta: Scale growth rate
    """
    x0: float
    gamma_0: float
    delta: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.gamma_0 <= 0:
            raise ValueError("gamma_0 must be positive")


# Step 2: Implement the distribution class
class CauchyDistribution(TimeEvolvingDistribution[CauchyParameters]):
    """
    Time-evolving Cauchy (Lorentzian) distribution.

    f(x, t) = 1 / (pi * gamma(t) * (1 + ((x - x0(t)) / gamma(t))^2))

    where:
        x0(t) = x0 + delta * t
        gamma(t) = gamma_0 * (1 + beta * t)
    """

    @property
    def name(self) -> str:
        return "Cauchy (Lorentzian)"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("x0", "gamma_0", "delta", "beta")

    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: CauchyParameters,
    ) -> NDArray[np.float64]:
        """Evaluate Cauchy PDF at time t."""
        # Time-dependent parameters
        x0_t = params.x0 + params.delta * t
        gamma_t = params.gamma_0 * (1 + params.beta * t)

        # Cauchy PDF formula
        return 1 / (np.pi * gamma_t * (1 + ((x - x0_t) / gamma_t) ** 2))

    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: CauchyParameters,
    ) -> NDArray[np.float64]:
        """Evaluate over the full grid."""
        time_grid = np.asarray(time_grid)
        pdf_matrix = np.zeros((len(time_grid), len(x)))

        for i, t in enumerate(time_grid):
            pdf_matrix[i, :] = self.pdf(x, float(t), params)

        return pdf_matrix

    def cdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: CauchyParameters,
    ) -> NDArray[np.float64]:
        """Cauchy CDF."""
        x0_t = params.x0 + params.delta * t
        gamma_t = params.gamma_0 * (1 + params.beta * t)
        return 0.5 + np.arctan((x - x0_t) / gamma_t) / np.pi

    def quantile(
        self,
        p: NDArray[np.float64] | float,
        t: float,
        params: CauchyParameters,
    ) -> NDArray[np.float64] | float:
        """Cauchy quantile function (inverse CDF)."""
        x0_t = params.x0 + params.delta * t
        gamma_t = params.gamma_0 * (1 + params.beta * t)
        return x0_t + gamma_t * np.tan(np.pi * (np.asarray(p) - 0.5))


def main() -> None:
    """Demonstrate custom distribution usage."""

    # Step 3: Register the distribution (optional, for string-based access)
    tpdf.DistributionRegistry.register("cauchy", CauchyDistribution)

    print("Custom Distribution Example: Cauchy Distribution")
    print("=" * 60)

    # Create instance directly
    cauchy = CauchyDistribution()
    params = CauchyParameters(
        x0=0.0,       # Centered at 0
        gamma_0=0.05, # Initial scale
        delta=0.001,  # Slow drift
        beta=0.02,    # Scale grows over time
    )

    # Evaluate
    grid = tpdf.EvaluationGrid.from_ranges(
        value_range=(-0.5, 0.5),
        time_range=(0, 60),
    )

    pdf_matrix = cauchy.pdf_matrix(grid.value_grid, grid.time_grid, params)

    result = tpdf.PDFResult(
        pdf_matrix=pdf_matrix,
        value_grid=grid.value_grid,
        time_grid=grid.time_grid,
        distribution_name=cauchy.name,
        parameters=params.__dict__,
    )

    print(f"Distribution: {result.distribution_name}")
    print(f"Grid shape: {result.shape}")
    print()

    # Compare with Normal distribution
    print("Comparing Cauchy vs Normal at t=30:")
    print("-" * 40)

    t = 30.0
    x, cauchy_pdf = result.slice_at_time(t)

    normal = tpdf.Normal()
    normal_params = tpdf.NormalParameters(
        mu_0=0.0,
        sigma_0=0.05,
        delta=0.001,
        beta=0.02,
    )
    normal_pdf = normal.pdf(x, t, normal_params)

    # Show tail behavior
    print(f"  At x=0 (center):   Cauchy={cauchy_pdf[100]:.4f}, Normal={normal_pdf[100]:.4f}")
    print(f"  At x=0.3 (tail):   Cauchy={cauchy_pdf[160]:.6f}, Normal={normal_pdf[160]:.6f}")
    print(f"  At x=-0.3 (tail):  Cauchy={cauchy_pdf[40]:.6f}, Normal={normal_pdf[40]:.6f}")
    print()
    print("Note: Cauchy has much heavier tails than Normal!")
    print()

    # Visualize
    plotter = tpdf.PDFPlotter()

    fig = plotter.surface_3d(result, title="Custom Cauchy Distribution")
    plotter.save(fig, "output_custom_cauchy.png")
    print("Saved: output_custom_cauchy.png")

    # Can also use string-based creation after registration
    print()
    print("Using registered name:")
    cauchy2 = tpdf.DistributionRegistry.create("cauchy")
    print(f"  Created: {cauchy2.name}")


if __name__ == "__main__":
    main()
