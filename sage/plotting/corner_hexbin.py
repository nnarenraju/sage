"""Corner plot of hexbins coloured by a value per point, with histograms on the diagonal."""

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator


def plot_corner_hexbin(points, values, *, reduce=np.mean, labels=None, histograms=None, colorbar_label=None,
                       cmap="viridis", vmin=None, vmax=None, gridsize=32, bins=40, panel_size=4.0,
                       save_path=None):
    """
    Corner plot: hexbins coloured by the values of the points in each cell, histograms on the diagonal.

    Examples
    --------
    Worst mismatch per cell::

        plot_corner_hexbin({"mass1": m1, "mass2": m2, "lambda1": l1}, np.log10(mismatch),
                           reduce=np.max, colorbar_label="log10 mismatch")

    Two sets of detections, coloured by the difference in counts::

        plot_corner_hexbin(params, found_by_sage.astype(float),
                           reduce=lambda z: np.sum(z == 1) - np.sum(z == 0),
                           histograms={"PyCBC": ~found_by_sage, "Sage": found_by_sage})

    Parameters
    ----------
    points : dict of str to array, shape (N,)
        Coordinates of each point; the order sets the rows and columns.
    values : array, shape (N,)
        Value of each point.
    reduce : callable
        Colour of a cell from the array of its points' values.
    labels : dict of str to str or None
        Axis label of each parameter; its name when absent.
    histograms : dict of str to bool array, shape (N,), or None
        Points of each diagonal histogram, by legend label; None draws one histogram of all points.
    colorbar_label : str or None
        Label of the colour bar.
    cmap : str or Colormap
        Colour map of the hexbins.
    vmin, vmax : float or None
        Colour limits; None takes the extremes over all panels.
    gridsize : int
        Hexagons across each panel.
    bins : int
        Bins of the diagonal histograms.
    panel_size : float
        Width and height of each panel, in inches.
    save_path : str or None
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(points)
    k = len(names)
    labels = {**{name: name for name in names}, **(labels or {})}
    values = np.asarray(values, dtype=float)
    groups = histograms or {None: np.ones(len(values), dtype=bool)}
    turbo = mpl.colormaps["turbo"]
    colours = [turbo(0.05), turbo(0.95)] if len(groups) == 2 else [turbo(x) for x in np.linspace(0.05, 0.95, len(groups))]
    styles = ["dashed", "solid"] if len(groups) == 2 else ["solid"] * len(groups)

    limits = {name: (np.min(points[name]), np.max(points[name])) for name in names}

    with plt.rc_context({"font.size": 18}):
        fig = plt.figure(figsize=(panel_size * k, panel_size * k), facecolor="white")
        grid = gridspec.GridSpec(k, k, figure=fig, wspace=0, hspace=0)
        cells = []
        for row, y in enumerate(names):
            for col, x in enumerate(names[:row + 1]):
                ax = fig.add_subplot(grid[row, col])
                ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
                # Pruned ticks keep labels of neighbouring panels apart where the panels meet
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
                ax.set_xlim(limits[x])
                if row == col:
                    edges = np.histogram(points[x], bins=bins, range=limits[x])[1]
                    for (label, mask), colour, style in zip(groups.items(), colours, styles):
                        ax.hist(np.asarray(points[x])[mask], bins=edges, histtype="step", color=colour,
                                linestyle=style, linewidth=1.5, label=label)
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
                    ax.set_ylabel("N", fontsize=16)
                else:
                    cells.append(ax.hexbin(points[x], points[y], C=values,
                                           reduce_C_function=lambda z: reduce(np.asarray(z)),
                                           gridsize=gridsize, cmap=cmap, extent=(*limits[x], *limits[y])))
                    ax.set_ylim(limits[y])
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
                    if col == 0:
                        ax.set_ylabel(labels[y])
                        ax.tick_params(axis="y", labelrotation=45)
                    else:
                        ax.set_yticklabels([])
                if row == k - 1:
                    ax.set_xlabel(labels[x])
                    ax.tick_params(axis="x", labelrotation=45)
                else:
                    ax.set_xticklabels([])

        filled = np.concatenate([c.get_array() for c in cells]) if cells else np.array([0.0])
        norm = mpl.colors.Normalize(np.nanmin(filled) if vmin is None else vmin,
                                    np.nanmax(filled) if vmax is None else vmax)
        for c in cells:
            c.set_norm(norm)

        # Colour bar and legend in the empty upper triangle
        top_right = grid[0, k - 1].get_position(fig)
        middle = grid[max(0, k // 2 - 1), k - 1].get_position(fig)
        cax = fig.add_axes([top_right.x0 + 0.35 * top_right.width, middle.y0,
                            0.08 * top_right.width, top_right.y1 - middle.y0])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        if colorbar_label:
            colorbar.set_label(colorbar_label)
        if histograms:
            handles = [Line2D([], [], color=colour, linestyle=style, linewidth=3.0, label=label)
                       for label, colour, style in zip(groups, colours, styles)]
            fig.legend(handles=handles, loc="upper center", frameon=False,
                       bbox_to_anchor=(grid[0, k // 2].get_position(fig).x1, top_right.y1))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig
