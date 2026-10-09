from __future__ import annotations

"""Unified publication/PPT plotting contract.

The heatmap palette follows the current MCID report's D-measure-inspired
blue→cyan→green→yellow→red convention (matplotlib ``jet``).  Text/lines/markers
are deliberately large for projection and paper figures.
"""

import matplotlib as mpl

DMEASURE_CMAP = mpl.colormaps["jet"]
NC_CMAP = DMEASURE_CMAP
METHOD_COLORS = {
    "Hodge": "#E64B35",
    "Proposed": "#E64B35",
    "DMeasure": "#4DBBD5",
    "NetLSD": "#00A087",
    "Portrait": "#3C5488",
    "Count": "#F39B7F",
}

FIG_DPI = 300
TITLE_SIZE = 25
SUPTITLE_SIZE = 29
LABEL_SIZE = 22
TICK_SIZE = 17
LEGEND_SIZE = 17
LINE_WIDTH = 3.0
MARKER_SIZE = 8.5
MST_NODE_SIZE = 1500
MST_NODE_SIZE_GLOBAL = 1050
MST_EDGE_WIDTH = 2.4
MST_LABEL_SIZE = 12


def apply_style() -> None:
    mpl.rcParams.update({
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
