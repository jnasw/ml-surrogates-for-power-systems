"""Matplotlib style helpers for DTU thesis-ready scientific figures.

The style is intentionally small and explicit. It uses DTU brand colours as a
controlled scientific palette while preserving clean axes, readable labels, and
editable vector text for thesis and publication exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cycler import cycler


DTU_COLORS = {
    "corporate_red": "#990000",
    "white": "#FFFFFF",
    "black": "#000000",
    "blue": "#2F3EEA",
    "bright_green": "#1FD082",
    "navy_blue": "#030F4F",
    "yellow": "#F6D04D",
    "orange": "#FC7634",
    "pink": "#F7BBB1",
    "grey": "#DADADA",
    "red": "#E83F48",
    "green": "#008835",
    "purple": "#79238E",
}


NEUTRAL_COLORS = {
    "text": "#202020",
    "axis": "#303030",
    "grid": "#DADADA",
    "light_grid": "#E8E8E8",
    "muted": "#6A6A6A",
}


# Ordered for scientific plots: enough contrast for lines and markers while
# avoiding a one-note DTU-red figure family.
THESIS_COLOR_CYCLE = [
    DTU_COLORS["blue"],
    DTU_COLORS["green"],
    DTU_COLORS["orange"],
    DTU_COLORS["purple"],
    DTU_COLORS["corporate_red"],
    DTU_COLORS["navy_blue"],
    DTU_COLORS["red"],
    DTU_COLORS["bright_green"],
]


THESIS_METHOD_COLORS = {
    "baseline": NEUTRAL_COLORS["muted"],
    "proposed": DTU_COLORS["corporate_red"],
    "lhs_static": NEUTRAL_COLORS["muted"],
    "qbc_deep_ensemble": DTU_COLORS["blue"],
    "marker_directed": DTU_COLORS["green"],
    "qbc_marker_hybrid": DTU_COLORS["orange"],
    "uniform_lhs": NEUTRAL_COLORS["text"],
    "random_resampling": DTU_COLORS["purple"],
    "rad": DTU_COLORS["corporate_red"],
    "data_only": NEUTRAL_COLORS["muted"],
    "static_uniform": DTU_COLORS["blue"],
    "static_tuned": DTU_COLORS["navy_blue"],
    "data_warmup_static": DTU_COLORS["green"],
    "ma": DTU_COLORS["orange"],
    "id": DTU_COLORS["purple"],
    "dn": DTU_COLORS["red"],
    "ntk": DTU_COLORS["corporate_red"],
    "softadapt": DTU_COLORS["orange"],
    "relobralo": DTU_COLORS["green"],
    "lra": DTU_COLORS["purple"],
    "gradnorm": DTU_COLORS["red"],
}


THESIS_BUDGET_COLORS = {
    "b256": DTU_COLORS["blue"],
    "b512": DTU_COLORS["green"],
    "b1024": DTU_COLORS["orange"],
    "b2048": DTU_COLORS["purple"],
    "b4096": DTU_COLORS["corporate_red"],
    "b8192": DTU_COLORS["navy_blue"],
}


THESIS_STYLE = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    # Figure and export
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.facecolor": "white",
    # Keep vector text editable in PDF/SVG exports.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Axes
    "axes.linewidth": 0.8,
    "axes.edgecolor": NEUTRAL_COLORS["axis"],
    "axes.labelcolor": NEUTRAL_COLORS["text"],
    "axes.titlecolor": NEUTRAL_COLORS["text"],
    "axes.titleweight": "normal",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=THESIS_COLOR_CYCLE),
    # Grid
    "grid.color": NEUTRAL_COLORS["grid"],
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    # Ticks
    "xtick.color": NEUTRAL_COLORS["text"],
    "ytick.color": NEUTRAL_COLORS["text"],
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    # Lines, markers, and patches
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "patch.linewidth": 0.7,
    "patch.edgecolor": NEUTRAL_COLORS["axis"],
    # Legend
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.45,
    "legend.borderaxespad": 0.4,
    "legend.labelspacing": 0.35,
}


def set_thesis_style() -> None:
    """Apply the DTU thesis Matplotlib style globally."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(THESIS_STYLE)


def thesis_style_context():
    """Return a Matplotlib rc context for temporary use of the thesis style."""
    import matplotlib.pyplot as plt

    return plt.rc_context(THESIS_STYLE)


def save_figure(
    fig,
    path: str | Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure in consistent thesis export formats.

    Parameters
    ----------
    fig:
        Matplotlib figure object.
    path:
        Output path without a suffix, or with a suffix to save one primary file.
    formats:
        Export formats used when `path` has no suffix.
    dpi:
        Raster resolution for PNG and other raster formats.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix:
        fig.savefig(output_path, dpi=dpi)
        return [output_path]

    saved_paths = []
    for fmt in formats:
        fmt = fmt.lstrip(".")
        candidate = output_path.with_suffix(f".{fmt}")
        fig.savefig(candidate, dpi=dpi)
        saved_paths.append(candidate)
    return saved_paths


def despine(ax, sides: Iterable[str] = ("top", "right")) -> None:
    """Hide selected spines for a cleaner scientific figure."""
    for side in sides:
        ax.spines[side].set_visible(False)


def apply_thesis_axis_style(ax, despine_sides: Iterable[str] = ("top", "right")) -> None:
    """Apply small axis-level polish that is not expressible via rcParams."""
    despine(ax, sides=despine_sides)
    ax.tick_params(which="both", top=False, right=False)


def method_color(name: str, default: str | None = None) -> str:
    """Return a stable thesis colour for a method or strategy key."""
    if name in THESIS_METHOD_COLORS:
        return THESIS_METHOD_COLORS[name]
    if default is not None:
        return default
    return NEUTRAL_COLORS["muted"]


def log_minor_grid(
    ax,
    *,
    axis: str = "y",
    subs=None,
    color: str | None = None,
    linewidth: float = 0.35,
    alpha: float = 0.6,
) -> None:
    """Enable sub-decade minor gridlines on a log-scale axis.

    ax.grid(which='minor') is a no-op before the first draw because minor-tick
    objects are created lazily.  This helper sets the locator and formatter,
    forces a canvas pass to materialise those objects, then enables the grid.

    Call after set_yscale/set_xscale and set_ylim/set_xlim, before
    plt.show() / save_and_show().

    Parameters
    ----------
    axis : 'x', 'y', or 'both'
    subs : sub-decade multipliers; defaults to [2, 3, …, 9]
    color : grid colour; defaults to NEUTRAL_COLORS['light_grid']
    linewidth, alpha : grid line appearance
    """
    import numpy as np
    from matplotlib.ticker import LogLocator, NullFormatter

    if subs is None:
        subs = np.arange(2, 10)
    if color is None:
        color = NEUTRAL_COLORS["light_grid"]

    targets = []
    if axis in ("y", "both"):
        targets.append(ax.yaxis)
    if axis in ("x", "both"):
        targets.append(ax.xaxis)

    for axobj in targets:
        axobj.set_minor_locator(LogLocator(base=10.0, subs=subs, numticks=100))
        axobj.set_minor_formatter(NullFormatter())

    # Force tick objects into existence so ax.grid can act on them.
    ax.figure.canvas.draw()

    ax.grid(True, which="minor", axis=axis, color=color, linewidth=linewidth, alpha=alpha)


def adjust_scatter_labels(
    texts: list,
    ax=None,
    *,
    arrowprops: dict | None = None,
    **kwargs,
) -> None:
    """Reposition scatter annotation texts to minimise overlap.

    Wraps ``adjustText.adjust_text`` with thesis-appropriate defaults.
    Falls back silently (labels stay at their initial positions) when the
    package is not installed.

    Parameters
    ----------
    texts:
        Annotation objects collected from ``ax.annotate`` calls.
    ax:
        The Axes the texts belong to.
    arrowprops:
        Leader-line style.  Defaults to a thin muted connector so adjusted
        labels remain traceable to their points.
    **kwargs:
        Forwarded to ``adjust_text``.
    """
    try:
        from adjustText import adjust_text
    except ImportError:
        return

    if arrowprops is None:
        arrowprops = {"arrowstyle": "-", "color": NEUTRAL_COLORS["muted"], "lw": 0.5}

    kw: dict = {"arrowprops": arrowprops}
    if ax is not None:
        kw["ax"] = ax
    kw.update(kwargs)
    adjust_text(texts, **kw)
