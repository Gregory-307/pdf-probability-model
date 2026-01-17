"""Interactive visualization using Plotly for rotatable 3D plots."""

from typing import Sequence
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.result import PDFResult


class InteractivePlotter:
    """
    Interactive visualization using Plotly.

    Creates rotatable, zoomable 3D plots that work in Jupyter notebooks
    and can be exported as standalone HTML files.

    Example:
        >>> plotter = InteractivePlotter()
        >>> fig = plotter.surface_3d(result)
        >>> fig.show()  # Interactive in notebook
        >>> plotter.save_html(fig, "plot.html")  # Standalone file
    """

    def __init__(self, colorscale: str = "Viridis"):
        """
        Initialize the interactive plotter.

        Args:
            colorscale: Plotly colorscale name (Viridis, Plasma, Blues, etc.)
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive plots. "
                "Install with: pip install plotly"
            )
        self.colorscale = colorscale

    def surface_3d(
        self,
        result: PDFResult,
        title: str | None = None,
        width: int = 800,
        height: int = 600,
        show_colorbar: bool = True,
    ) -> "go.Figure":
        """
        Create an interactive 3D surface plot.

        The plot can be rotated by clicking and dragging, zoomed with scroll,
        and panned with right-click drag.

        Args:
            result: PDFResult containing the PDF matrix
            title: Plot title (defaults to distribution name)
            width: Figure width in pixels
            height: Figure height in pixels
            show_colorbar: Whether to show the color scale bar

        Returns:
            Plotly Figure object (call .show() to display)
        """
        X, T = np.meshgrid(result.value_grid, result.time_grid)

        fig = go.Figure(data=[
            go.Surface(
                x=T,
                y=X,
                z=result.pdf_matrix,
                colorscale=self.colorscale,
                showscale=show_colorbar,
                colorbar=dict(title="Density") if show_colorbar else None,
            )
        ])

        plot_title = title or f"3D PDF: {result.distribution_name}"
        cum_ex = result.cumulative_expected_value
        plot_title += f"<br><sub>Cumulative E[X]: {cum_ex:.4f}</sub>"

        fig.update_layout(
            title=dict(text=plot_title, x=0.5),
            width=width,
            height=height,
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Value",
                zaxis_title="Probability Density",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        return fig

    def multi_surface(
        self,
        results: Sequence[PDFResult],
        titles: Sequence[str] | None = None,
        rows: int = 2,
        cols: int = 2,
        width: int = 1200,
        height: int = 900,
    ) -> "go.Figure":
        """
        Create multiple interactive 3D surfaces in a grid.

        Args:
            results: Sequence of PDFResult objects
            titles: Titles for each subplot
            rows: Number of rows in grid
            cols: Number of columns in grid
            width: Total figure width
            height: Total figure height

        Returns:
            Plotly Figure with subplots
        """
        titles = titles or [r.distribution_name for r in results]

        # Create subplot specs for 3D
        specs = [[{"type": "surface"} for _ in range(cols)] for _ in range(rows)]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        for i, result in enumerate(results):
            row = i // cols + 1
            col = i % cols + 1

            X, T = np.meshgrid(result.value_grid, result.time_grid)

            fig.add_trace(
                go.Surface(
                    x=T,
                    y=X,
                    z=result.pdf_matrix,
                    colorscale=self.colorscale,
                    showscale=False,
                ),
                row=row,
                col=col,
            )

        # Update all scene layouts
        for i in range(len(results)):
            scene_name = f"scene{i+1}" if i > 0 else "scene"
            fig.update_layout(**{
                scene_name: dict(
                    xaxis_title="Time",
                    yaxis_title="Value",
                    zaxis_title="Density",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                )
            })

        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        return fig

    def heatmap(
        self,
        result: PDFResult,
        title: str | None = None,
        width: int = 800,
        height: int = 500,
    ) -> "go.Figure":
        """
        Create an interactive heatmap.

        Args:
            result: PDFResult containing the PDF matrix
            title: Plot title
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=result.pdf_matrix,
            x=result.value_grid,
            y=result.time_grid,
            colorscale=self.colorscale,
            colorbar=dict(title="Density"),
        ))

        fig.update_layout(
            title=dict(text=title or f"PDF Heatmap: {result.distribution_name}", x=0.5),
            xaxis_title="Value",
            yaxis_title="Time",
            width=width,
            height=height,
        )

        return fig

    def compare_slices(
        self,
        results: Sequence[PDFResult],
        time_point: float,
        labels: Sequence[str] | None = None,
        title: str = "Distribution Comparison",
        width: int = 800,
        height: int = 500,
    ) -> "go.Figure":
        """
        Compare multiple distributions at a specific time point.

        Args:
            results: Sequence of PDFResult objects
            time_point: Time point for comparison
            labels: Labels for each distribution
            title: Plot title
            width: Figure width
            height: Figure height

        Returns:
            Plotly Figure with overlaid PDFs
        """
        labels = labels or [r.distribution_name for r in results]

        fig = go.Figure()

        for result, label in zip(results, labels):
            x, pdf = result.slice_at_time(time_point)
            fig.add_trace(go.Scatter(
                x=x,
                y=pdf,
                mode='lines',
                name=label,
                line=dict(width=2),
            ))

        fig.update_layout(
            title=dict(text=f"{title} at t={time_point:.1f}", x=0.5),
            xaxis_title="Value",
            yaxis_title="Probability Density",
            width=width,
            height=height,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            hovermode='x unified',
        )

        return fig

    def expected_value_trajectory(
        self,
        result: PDFResult,
        title: str | None = None,
        width: int = 800,
        height: int = 400,
        show_bands: bool = True,
    ) -> "go.Figure":
        """
        Plot expected value over time with optional confidence bands.

        Args:
            result: PDFResult containing the PDF matrix
            title: Plot title
            width: Figure width
            height: Figure height
            show_bands: Whether to show confidence bands

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        expected_values = result.expected_values

        # Add confidence bands if requested
        if show_bands:
            # Calculate approximate confidence bands
            stds = []
            for i in range(len(result.time_grid)):
                pdf = result.pdf_matrix[i, :]
                if np.sum(pdf) > 0:
                    pdf_norm = pdf / np.sum(pdf)
                    mean = np.sum(result.value_grid * pdf_norm)
                    var = np.sum((result.value_grid - mean)**2 * pdf_norm)
                    stds.append(np.sqrt(var))
                else:
                    stds.append(0)
            stds = np.array(stds)

            # 95% confidence band
            upper = expected_values + 1.96 * stds
            lower = expected_values - 1.96 * stds

            fig.add_trace(go.Scatter(
                x=np.concatenate([result.time_grid, result.time_grid[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=True,
            ))

        # Main E[X] line
        fig.add_trace(go.Scatter(
            x=result.time_grid,
            y=expected_values,
            mode='lines',
            name='E[X]',
            line=dict(color='blue', width=2),
        ))

        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        cum_ex = result.cumulative_expected_value
        plot_title = title or f"Expected Value: {result.distribution_name}"
        plot_title += f" (Cumulative: {cum_ex:.4f})"

        fig.update_layout(
            title=dict(text=plot_title, x=0.5),
            xaxis_title="Time",
            yaxis_title="Expected Value",
            width=width,
            height=height,
            hovermode='x unified',
        )

        return fig

    @staticmethod
    def save_html(fig: "go.Figure", path: str, include_plotlyjs: bool = True) -> None:
        """
        Save figure as standalone HTML file.

        The HTML file can be opened in any browser and remains interactive.

        Args:
            fig: Plotly Figure to save
            path: Output file path (.html)
            include_plotlyjs: Whether to include plotly.js in the file
                             (True = standalone, False = smaller but needs CDN)
        """
        fig.write_html(
            path,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
        )

    @staticmethod
    def to_html(fig: "go.Figure", include_plotlyjs: str = "cdn") -> str:
        """
        Convert figure to HTML string for embedding.

        Args:
            fig: Plotly Figure
            include_plotlyjs: 'cdn' for external JS, True for inline

        Returns:
            HTML string
        """
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)
