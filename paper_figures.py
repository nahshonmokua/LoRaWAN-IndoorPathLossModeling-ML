"""Figure style: every figure is drawn at its printed width (IEEE column 3.5 in, page 7.16 in) in a Times-like serif, with ticks
inside the axes, legends inside the panels and a subcaption "(a) ..." below each panel. The six model families keep one colour and
one marker in every figure; the colours are told apart under every colour-vision deficiency, pair by pair. Everything that is not a
family (baselines, estimators, datasets) is drawn in ink and greys, and LightGBM on the reference rung in amber."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLUMN, PAGE = 246 / 72.27, 504 / 72.27                    # the printed column and text widths (246 pt and 504 pt), in inches

FAMILIES = ["Linear", "RF", "XGB", "LGBM", "KNN", "ANN"]
NAME = {"Linear": "Linear", "RF": "RF", "XGB": "XGBoost", "LGBM": "LightGBM", "KNN": "$k$-NN", "ANN": "ANN"}
COLOUR = {"Linear": "#0072b2", "RF": "#009988", "XGB": "#d55e00", "LGBM": "#eda100", "KNN": "#882255", "ANN": "#56b4e9"}
MARKER = {"Linear": "o", "RF": "s", "XGB": "^", "LGBM": "D", "KNN": "v", "ANN": "p"}
MARKER_SIZE = {"o": 4.2, "s": 3.8, "^": 4.4, "D": 3.6, "v": 4.4, "p": 4.6}     # sizes that give the six shapes the same visual weight
INK, INK2 = "#1a1a1a", "#4d4d4d"
AMBER, AMBER_DARK = COLOUR["LGBM"], "#a86f00"

# The rungs on an axis: the rung, then what it adds
RUNG_LABEL = {"static": "static", "H1": "H1\nhistory", "H2": "H2\nchannel", "H3": "H3\nclock", "H4": "H4\nSF", "H5": "H5\nSNR",
              "H6": "H6\nsensors", "H7": "H7\ngeometry"}


def use():
    """Matplotlib's defaults plus the printed sizes; called at the top of every figure cell."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "in", "ytick.direction": "in",
        "lines.linewidth": 1.1, "lines.markersize": 4, "lines.markeredgewidth": 0.5,
        "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.45, "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.1, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 100})


def family_line(ax, x, y, fam, **kw):
    """One family's series: its colour and its marker, the marker edged in ink so that the light colours hold on white and in print."""
    m = MARKER[fam]
    style = dict(color=COLOUR[fam], marker=m, ms=MARKER_SIZE[m], mfc=COLOUR[fam], mec=INK, mew=0.45, lw=1.1, label=NAME[fam], zorder=3)
    style.update(kw)
    return ax.plot(x, y, **style)


def family_handles():
    """Legend entries for the six families, drawn as family_line draws them."""
    return [Line2D([], [], color=COLOUR[f], marker=MARKER[f], ms=MARKER_SIZE[MARKER[f]], mfc=COLOUR[f], mec=INK, mew=0.45, lw=1.1, label=NAME[f])
            for f in FAMILIES]


def subcaptions(fig, axes, texts, pad_pt=5):
    """Subcaptions "(a) ...", centred under each panel and aligned on one baseline for each row of panels."""
    fig.canvas.draw()
    renderer, to_fig = fig.canvas.get_renderer(), fig.transFigure.inverted()
    rows = {}
    for ax, text in zip(axes, texts):
        bottom = to_fig.transform((0, ax.get_tightbbox(renderer).y0))[1]
        rows.setdefault(round(ax.get_position().y0, 3), []).append((ax, text, bottom))
    pad = pad_pt / 72 / fig.get_figheight()
    for row in rows.values():
        y = min(bottom for _, _, bottom in row) - pad
        for ax, text, _ in row:
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2, y, text, ha="center", va="top", fontsize=8)


def row_regions(fig, ax, groups, right, half, gap=2.0, margin=0.75):
    """Shade groups of rows as regions behind the labels and the panels, from the left edge of the saved image to `right` (a figure
    fraction). groups: (colour, first row, last row) in row order, the rows at integer y on ax, whose y axis runs downwards; half: the half
    height of the bars in rows. Two neighbouring regions are parted by a white gap of `gap` pt, centred between the bars and the label of
    the last row of the one and those of the first row of the other. Checks that every label and bar lies at least `margin` pt inside its
    region, the labels also from the region's left edge."""
    from matplotlib.patches import Rectangle
    fig.canvas.draw()
    renderer, pt = fig.canvas.get_renderer(), fig.dpi / 72
    y = lambda row: ax.transData.transform((0, row))[1]                       # display y, upwards
    label = {int(round(t)): s.get_window_extent(renderer) for t, s in zip(ax.get_yticks(), ax.get_yticklabels())}
    top = lambda row: max(y(row - half), label[row].y1)                        # the highest point of a row's bar and label
    bottom = lambda row: min(y(row + half), label[row].y0)
    edges = [max(y(groups[0][1] - 0.46), top(groups[0][1]) + margin * pt)]
    for (_, _, last), (_, first, _) in zip(groups, groups[1:]):
        mid = (bottom(last) + top(first)) / 2
        assert bottom(last) - mid >= (gap / 2 + margin) * pt and mid - top(first) >= (gap / 2 + margin) * pt, f"no room for a gap between rows {last} and {first}"
        edges += [mid + gap / 2 * pt, mid - gap / 2 * pt]
    edges.append(min(y(groups[-1][2] + 0.46), bottom(groups[-1][2]) - margin * pt))
    pad = plt.rcParams["savefig.pad_inches"] / fig.get_figwidth()             # the saved image starts this far before the figure's content
    to_fig = fig.transFigure.inverted()
    for (colour, _, _), upper, lower in zip(groups, edges[0::2], edges[1::2]):
        y0, y1 = to_fig.transform((0, lower))[1], to_fig.transform((0, upper))[1]
        fig.patches.append(Rectangle((pad, y0), right - pad, y1 - y0, transform=fig.transFigure, facecolor=colour, edgecolor="none", zorder=-10))
    assert min(s.x0 for s in label.values()) >= (pad * fig.get_figwidth() * 72 + 2) * pt, "a label leaves the shading at the left"


def break_marks(ax_top, ax_bottom, size=0.012):
    """Diagonal marks on both spines where a y-axis is broken between two stacked axes."""
    for ax, y in ((ax_top, 0), (ax_bottom, 1)):
        h = size * ax_top.get_position().height / ax.get_position().height * 3
        for x in (0, 1):
            ax.plot([x - size, x + size], [y - h, y + h], transform=ax.transAxes, color=INK2, lw=0.6, clip_on=False)


def panel(ax, tag):
    """The panel label, inside the axes at the top left (the Q-Q plots of notebook 08)."""
    ax.text(0.025, 0.975, f"({tag})", transform=ax.transAxes, ha="left", va="top")
