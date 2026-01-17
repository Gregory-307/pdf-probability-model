"""Style configuration for visualizations."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PlotStyle:
    """
    Configuration for plot styling.

    Provides consistent styling across all visualizations.
    Can be customized or use one of the preset styles.

    Attributes:
        name: Style name for identification
        font_family: Font family for text
        font_size: Base font size
        title_size: Title font size
        label_size: Axis label font size
        primary_color: Primary color for lines/markers
        secondary_color: Secondary color
        cmap: Default colormap for surfaces/heatmaps
        grid_alpha: Grid line transparency
        figure_dpi: Figure DPI for rendering
        line_width: Default line width
    """

    name: str
    font_family: str = "sans-serif"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    cmap: str = "viridis"
    grid_alpha: float = 0.3
    figure_dpi: int = 100
    line_width: float = 1.5

    def to_rcparams(self) -> dict[str, Any]:
        """
        Convert to matplotlib rcParams dict.

        Returns:
            Dictionary suitable for plt.rcParams.update()
        """
        return {
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "figure.dpi": self.figure_dpi,
            "lines.linewidth": self.line_width,
            "grid.alpha": self.grid_alpha,
        }


# Preset styles
DEFAULT_STYLE = PlotStyle(
    name="default",
    font_family="sans-serif",
    font_size=12,
    title_size=14,
    cmap="viridis",
)

PUBLICATION_STYLE = PlotStyle(
    name="publication",
    font_family="serif",
    font_size=11,
    title_size=12,
    label_size=11,
    figure_dpi=300,
    line_width=1.0,
    cmap="viridis",
)

PRESENTATION_STYLE = PlotStyle(
    name="presentation",
    font_family="sans-serif",
    font_size=14,
    title_size=18,
    label_size=14,
    figure_dpi=150,
    line_width=2.0,
    cmap="plasma",
)

DARK_STYLE = PlotStyle(
    name="dark",
    font_family="sans-serif",
    font_size=12,
    title_size=14,
    primary_color="#00d4ff",
    secondary_color="#ff6b6b",
    cmap="magma",
    figure_dpi=100,
)
