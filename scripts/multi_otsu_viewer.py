"""Minimal viewer for multi_otsu results: image, class overlay, class toggles.

Shows a grayscale image with a semi-transparent colour per intensity class
and one checkbox per class to show or hide it, as in find_threshold.py.
Takes only the image and the (thresholds, labels) pair returned by
multi_otsu_jax.multi_otsu; no thresholding is done here.

Example:
    from multi_otsu_jax import multi_otsu
    from multi_otsu_viewer import show_classes

    thresholds, labels = multi_otsu(image, classes=4)
    show_classes(image, thresholds, labels)

Hovering over the image reports the pixel's class next to its intensity.

Dependencies: numpy and matplotlib (>= 3.7 for coloured checkboxes).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import CheckButtons


def _class_colors(cmap, classes):
    """Return one RGBA row per class, dark class first.

    tab20 stores hues as dark/light pairs, so taking it in order would give
    neighbouring intensity classes nearly the same hue. Its dark shades are
    used first, then the light ones, so adjacent classes stay distinct.
    Other listed colormaps are cycled in order; continuous ones are sampled
    evenly from end to end.
    """
    cmap = matplotlib.colormaps.get_cmap(cmap)
    if isinstance(cmap, ListedColormap):
        order = np.arange(cmap.N)
        if cmap.name == "tab20":
            order = np.r_[0:cmap.N:2, 1:cmap.N:2]
        return cmap(order[np.arange(classes) % cmap.N])
    return cmap(np.linspace(0, 1, classes))


def _class_names(thresholds, counts):
    """Label each class with its intensity range and share of pixels.

    Thresholds are bin edges with ties going to the brighter class, so each
    class covers a half-open interval [lower, upper).
    """
    fractions = counts / counts.sum() * 100
    if len(thresholds) == 0:
        return [f"0: all  ({fractions[0]:.1f}%)"]
    bounds = [f"< {thresholds[0]:.6g}"]
    bounds += [f"[{lo:.6g}, {hi:.6g})" for lo, hi in zip(thresholds[:-1], thresholds[1:])]
    bounds += [f"≥ {thresholds[-1]:.6g}"]
    return [f"{i}: {b}  ({f:.1f}%)" for i, (b, f) in enumerate(zip(bounds, fractions))]


def show_classes(image, thresholds, labels, *, selected=None, cmap="tab20",
                 alpha=0.45, show=True):
    """Display image with a toggleable per-class overlay.

    Parameters
    ----------
    image : numpy.ndarray, shape (Y, X)
        Intensities to display, e.g. float32. Contrast is stretched between
        the 0.5th and 99.5th percentiles so hot pixels do not dominate.
    thresholds : numpy.ndarray, shape (classes-1,)
        Increasing thresholds from multi_otsu; used for the checkbox labels.
    labels : numpy.ndarray of int, shape (Y, X)
        Class per pixel, 0 (darkest) through classes-1, from multi_otsu.
    selected : iterable of int or None
        Classes shown initially. None shows every class.
    cmap : str or matplotlib.colors.Colormap, default "tab20"
        Overlay colours, one per class.
    alpha : float, default 0.45
        Opacity of the overlay for shown classes.
    show : bool, default True
        Call plt.show(). Pass False to add to or save the figure first.

    Returns
    -------
    fig : matplotlib.figure.Figure
    checks : matplotlib.widgets.CheckButtons
        Keep a reference to this: matplotlib widgets stop responding once
        they are garbage collected.
    """
    image = np.asarray(image)
    labels = np.asarray(labels)
    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    classes = len(thresholds) + 1
    if image.ndim != 2:
        raise ValueError("image must be a 2-D array")
    if labels.shape != image.shape:
        raise ValueError("labels must have the same shape as image")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer array")
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError(f"labels must lie in 0..{classes - 1} for {len(thresholds)} thresholds")

    visible = np.zeros(classes, dtype=bool)
    visible[list(range(classes)) if selected is None else list(selected)] = True
    colors = _class_colors(cmap, classes)
    # A per-class lookup table means a toggle only rewrites the alpha column
    # and re-indexes it by labels, rather than recolouring each class mask.
    lut = np.round(colors * 255).astype(np.uint8)
    lut[:, 3] = round(alpha * 255)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.88])
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax <= vmin:
        vmin, vmax = image.min(), image.max()
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    overlay = ax.imshow(np.where(visible[:, None], lut, 0)[labels],
                        interpolation="nearest")
    # Report intensity from the grayscale image, not the overlay's RGBA.
    overlay.set_mouseover(False)
    ax.format_coord = lambda x, y: _format_coord(x, y, labels)
    ax.set_title(f"Multi-Otsu: {classes} classes")
    ax.set_axis_off()

    # One checkbox per class, filled with that class's overlay colour.
    height = min(0.88, 0.05 * classes + 0.02)
    check_ax = fig.add_axes([0.66, 0.93 - height, 0.33, height])
    check_ax.set_title("Classes shown", loc="left", fontsize=10)
    counts = np.bincount(labels.ravel(), minlength=classes)
    checks = CheckButtons(
        check_ax, _class_names(thresholds, counts), actives=list(visible),
        frame_props={"facecolor": colors, "edgecolor": "k", "s": 150},
        check_props={"facecolor": "k", "s": 80})

    def toggle(_):
        visible[:] = checks.get_status()
        overlay.set_data(np.where(visible[:, None], lut, 0)[labels])
        fig.canvas.draw_idle()

    checks.on_clicked(toggle)
    if show:
        plt.show()
    return fig, checks


def _format_coord(x, y, labels):
    """Status-bar text with the class under the cursor."""
    col, row = int(round(x)), int(round(y))
    if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
        return f"x={col}, y={row}, class={labels[row, col]}"
    return f"x={x:.1f}, y={y:.1f}"
