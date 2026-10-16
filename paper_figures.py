"""The figure style of the paper: every figure is drawn at its printed width (IEEE column 3.5 in, page 7.16 in) in a Times-like serif,
with the panel label inside the axes at the top left and legends inside the axes or below the figure."""
import matplotlib.pyplot as plt

COLUMN, PAGE = 3.5, 7.16


def use():
    """Matplotlib's defaults plus the sizes of the paper; called at the top of every figure cell."""
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                         "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4.5, "lines.markeredgewidth": 0.5,
                         "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5, "axes.axisbelow": True,
                         "legend.frameon": False, "legend.handlelength": 2.0, "legend.columnspacing": 1.0,
                         "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def panel(ax, tag):
    """The panel label, inside the axes at the top left."""
    ax.text(0.025, 0.975, f"({tag})", transform=ax.transAxes, ha="left", va="top", fontsize=9)


def inside_legend(ax, **kw):
    """A legend inside the axes, on a white box so that grid lines and curves do not run through it."""
    return ax.legend(frameon=True, framealpha=0.92, edgecolor="0.75", fancybox=False, borderpad=0.4, **kw)
