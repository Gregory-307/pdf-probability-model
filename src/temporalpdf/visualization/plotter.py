"""High-level visualization interface for PDF results."""

from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from ..core.result import PDFResult
from .styles import PlotStyle, DEFAULT_STYLE


class PDFPlotter:
    """
    High-level visualization interface for PDF results.

    Provides a fluent API for creating various visualizations
    of time-evolving probability distributions.

    Example:
        >>> plotter = PDFPlotter(style=PlotStyle.PUBLICATION)
        >>> fig = plotter.surface_3d(result, title="My Distribution")
        >>> plotter.save(fig, "output.png")

    Attributes:
        style: PlotStyle configuration
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        """
        Initialize the plotter with a style configuration.

        Args:
            style: PlotStyle instance for consistent styling
        """
        self.style = style
        self._apply_style()

    def _apply_style(self) -> None:
        """Apply matplotlib style settings."""
        plt.rcParams.update(self.style.to_rcparams())

    def surface_3d(
        self,
        result: PDFResult,
        title: str | None = None,
        xlabel: str = "Time",
        ylabel: str = "Value",
        zlabel: str = "Probability Density",
        cmap: str | None = None,
        figsize: tuple[int, int] = (12, 8),
        elevation: float = 30,
        azimuth: float = -60,
        show_cumulative_ex: bool = True,
    ) -> Figure:
        """
        Create a 3D surface plot of the PDF.

        Args:
            result: PDFResult containing the PDF matrix
            title: Plot title (defaults to distribution name)
            xlabel, ylabel, zlabel: Axis labels
            cmap: Colormap name (defaults to style's cmap)
            figsize: Figure size in inches
            elevation: Viewing elevation angle
            azimuth: Viewing azimuth angle
            show_cumulative_ex: Whether to show cumulative E[X] in title

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        X, T = np.meshgrid(result.value_grid, result.time_grid)
        ax.plot_surface(
            T, X, result.pdf_matrix,
            cmap=cmap or self.style.cmap,
            edgecolor="none",
            alpha=0.9,
        )

        # Build title
        plot_title = title or f"3D PDF: {result.distribution_name}"
        if show_cumulative_ex:
            cum_ex = result.cumulative_expected_value
            plot_title += f"\nCumulative E[X]: {cum_ex:.4f}"

        ax.set_title(plot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.view_init(elev=elevation, azim=azimuth)

        plt.tight_layout()
        return fig

    def heatmap(
        self,
        result: PDFResult,
        title: str | None = None,
        xlabel: str = "Value",
        ylabel: str = "Time",
        cmap: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        colorbar_label: str = "Probability Density",
    ) -> Figure:
        """
        Create a 2D heatmap of the PDF.

        Args:
            result: PDFResult containing the PDF matrix
            title: Plot title (defaults to distribution name)
            xlabel, ylabel: Axis labels
            cmap: Colormap name
            figsize: Figure size in inches
            colorbar_label: Label for the colorbar

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            result.pdf_matrix,
            aspect="auto",
            extent=[
                result.value_grid.min(),
                result.value_grid.max(),
                result.time_grid.min(),
                result.time_grid.max(),
            ],
            origin="lower",
            cmap=cmap or self.style.cmap,
        )

        plt.colorbar(im, ax=ax, label=colorbar_label)
        ax.set_title(title or f"PDF Heatmap: {result.distribution_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.tight_layout()
        return fig

    def slice_at_time(
        self,
        result: PDFResult,
        t: float,
        observed_value: float | None = None,
        expected_value: float | None = None,
        title: str | None = None,
        xlabel: str = "Value",
        ylabel: str = "Probability Density",
        figsize: tuple[int, int] = (10, 6),
        show_stats: bool = True,
    ) -> Figure:
        """
        Plot a PDF slice at a specific time point.

        Args:
            result: PDFResult containing the PDF matrix
            t: Time point to slice at
            observed_value: Optional observed value to mark
            expected_value: Optional expected value to mark (computed if None and show_stats=True)
            title: Plot title
            xlabel, ylabel: Axis labels
            figsize: Figure size in inches
            show_stats: Whether to compute and show E[X] at this time

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        x, pdf = result.slice_at_time(t)
        ax.plot(x, pdf, label="PDF", color=self.style.primary_color, linewidth=2)

        # Mark observed value if provided
        if observed_value is not None:
            ax.axvline(
                observed_value,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Observed: {observed_value:.4f}",
            )

        # Mark expected value
        if show_stats:
            exp_val = expected_value or result.expected_value_at_time(t)
            ax.axvline(
                exp_val,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"E[X]: {exp_val:.4f}",
            )
        elif expected_value is not None:
            ax.axvline(
                expected_value,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"E[X]: {expected_value:.4f}",
            )

        ax.set_title(title or f"PDF at t={t:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=self.style.grid_alpha)

        plt.tight_layout()
        return fig

    def compare_distributions(
        self,
        results: Sequence[PDFResult],
        time_point: float,
        labels: Sequence[str] | None = None,
        title: str = "Distribution Comparison",
        xlabel: str = "Value",
        ylabel: str = "Probability Density",
        figsize: tuple[int, int] = (12, 6),
    ) -> Figure:
        """
        Overlay multiple distributions at a specific time point.

        Args:
            results: Sequence of PDFResult objects to compare
            time_point: Time point for comparison
            labels: Labels for each distribution (defaults to distribution names)
            title: Plot title
            xlabel, ylabel: Axis labels
            figsize: Figure size in inches

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        labels = labels or [r.distribution_name for r in results]
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # type: ignore[attr-defined]

        for result, label, color in zip(results, labels, colors):
            x, pdf = result.slice_at_time(time_point)
            ax.plot(x, pdf, label=label, color=color, linewidth=2)

        ax.set_title(f"{title} at t={time_point:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=self.style.grid_alpha)

        plt.tight_layout()
        return fig

    def multi_scenario(
        self,
        results: Sequence[PDFResult],
        titles: Sequence[str] | None = None,
        layout: tuple[int, int] = (2, 2),
        figsize: tuple[int, int] = (16, 12),
        cmap: str | None = None,
        show_cumulative_ex: bool = True,
    ) -> Figure:
        """
        Create a multi-panel plot for comparing scenarios.

        Args:
            results: Sequence of PDFResult objects
            titles: Titles for each subplot (defaults to distribution names)
            layout: Grid layout (rows, cols)
            figsize: Figure size in inches
            cmap: Colormap name
            show_cumulative_ex: Whether to show cumulative E[X]

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(
            layout[0],
            layout[1],
            figsize=figsize,
            subplot_kw={"projection": "3d"},
        )

        titles = titles or [r.distribution_name for r in results]
        cmap = cmap or self.style.cmap

        for ax, result, title in zip(axes.flatten(), results, titles):
            X, T = np.meshgrid(result.value_grid, result.time_grid)
            ax.plot_surface(T, X, result.pdf_matrix, cmap=cmap, edgecolor="none")

            plot_title = title
            if show_cumulative_ex:
                cum_ex = result.cumulative_expected_value
                plot_title += f"\nCumulative E[X]: {cum_ex:.4f}"

            ax.set_title(plot_title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_zlabel("Density")

        plt.tight_layout()
        return fig

    def expected_value_over_time(
        self,
        result: PDFResult,
        title: str | None = None,
        xlabel: str = "Time",
        ylabel: str = "Expected Value",
        figsize: tuple[int, int] = (10, 6),
        show_cumulative: bool = True,
    ) -> Figure:
        """
        Plot expected value E[X] over time.

        Args:
            result: PDFResult containing the PDF matrix
            title: Plot title
            xlabel, ylabel: Axis labels
            figsize: Figure size in inches
            show_cumulative: Whether to show cumulative E[X] in legend

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        expected_values = result.expected_values

        label = "E[X]"
        if show_cumulative:
            cum_ex = result.cumulative_expected_value
            label += f" (Cumulative: {cum_ex:.4f})"

        ax.plot(
            result.time_grid,
            expected_values,
            label=label,
            color=self.style.primary_color,
            linewidth=2,
        )

        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.set_title(title or f"Expected Value: {result.distribution_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=self.style.grid_alpha)

        plt.tight_layout()
        return fig

    def confidence_bands(
        self,
        result: PDFResult,
        confidence_levels: Sequence[float] = (0.5, 0.9, 0.95),
        title: str | None = None,
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: tuple[int, int] = (12, 6),
    ) -> Figure:
        """
        Plot confidence bands showing how uncertainty grows over time.

        Args:
            result: PDFResult containing the PDF matrix
            confidence_levels: Confidence levels to plot (e.g., 0.5 for 50%)
            title: Plot title
            xlabel, ylabel: Axis labels
            figsize: Figure size in inches

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate quantiles for each time point
        expected_values = result.expected_values
        ax.plot(
            result.time_grid,
            expected_values,
            label="E[X]",
            color=self.style.primary_color,
            linewidth=2,
        )

        # Sort confidence levels for proper layering
        sorted_levels = sorted(confidence_levels, reverse=True)
        alphas = np.linspace(0.1, 0.3, len(sorted_levels))

        for level, alpha in zip(sorted_levels, alphas):
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q

            lowers = []
            uppers = []

            for i in range(len(result.time_grid)):
                pdf = result.pdf_matrix[i, :]
                # Normalize to ensure it's a valid PDF
                pdf_norm = pdf / np.trapezoid(pdf, result.value_grid)
                cdf = np.cumsum(pdf_norm) * (result.value_grid[1] - result.value_grid[0])
                cdf = cdf / cdf[-1]  # Ensure CDF ends at 1

                lower_idx = np.searchsorted(cdf, lower_q)
                upper_idx = np.searchsorted(cdf, upper_q)

                lower_idx = min(lower_idx, len(result.value_grid) - 1)
                upper_idx = min(upper_idx, len(result.value_grid) - 1)

                lowers.append(result.value_grid[lower_idx])
                uppers.append(result.value_grid[upper_idx])

            ax.fill_between(
                result.time_grid,
                lowers,
                uppers,
                alpha=alpha,
                color=self.style.primary_color,
                label=f"{int(level * 100)}% CI",
            )

        ax.set_title(title or f"Confidence Bands: {result.distribution_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=self.style.grid_alpha)

        plt.tight_layout()
        return fig

    @staticmethod
    def save(fig: Figure, path: str, dpi: int = 300, transparent: bool = False) -> None:
        """
        Save figure to file.

        Args:
            fig: matplotlib Figure to save
            path: Output file path
            dpi: Resolution in dots per inch
            transparent: Whether to use transparent background
        """
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close(fig)

    @staticmethod
    def show(fig: Figure) -> None:
        """
        Display figure interactively.

        Args:
            fig: matplotlib Figure to display
        """
        plt.show()

    @staticmethod
    def close(fig: Figure) -> None:
        """
        Close figure to free memory.

        Args:
            fig: matplotlib Figure to close
        """
        plt.close(fig)
