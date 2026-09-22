"""

Accepts one 2-D uint16 or float32 image. Bin count and class count are
configurable.

The dynamic program maximizes the multi-Otsu between-class variance over
contiguous, nonempty histogram classes in O(classes * nbins**2) work.
Candidate intervals are evaluated in tiles, using O(tile_size * nbins +
classes * nbins) workspace instead of a full quadratic score matrix.

NumPy constructs the histogram and assigns output labels on the CPU. Only
the small histogram is transferred to the JAX device for the compiled
threshold search. CPU, CUDA, and Apple's experimental jax-metal backend
use the same JAX operations.

Thresholds are histogram BIN EDGES in original intensity units, with pixels
equal to a threshold assigned to the brighter class. This keeps pixel labels
consistent with the optimized histogram partition. It is not scikit-image's
bin-center threshold convention, so thresholds/labels need not be identical.

Example:
    import numpy as np
    from multi_otsu_jax import multi_otsu

    image = np.load("image.npy")
    thresholds, labels = multi_otsu(image, classes=4, nbins=1024)
    foreground = np.isin(labels, [2, 3])

"""

from functools import partial
import operator
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from tqdm import tqdm


@partial(jax.jit, static_argnames=("classes", "tile_size"))
def _solve_histogram(histogram, *, classes, tile_size):
    """Return optimal split indices and score for a uniform-bin histogram.

    Each split is the first bin in the brighter class. Exact computed ties
    choose the earliest last split, recursively through the earlier classes.
    All arrays have static shapes so the same code can compile for GPUs.
    """
    nbins = histogram.shape[0]
    dtype = histogram.dtype

    # Normalize intensity coordinates and counts to limit dynamic range.
    # An affine intensity transform leaves the optimal partition unchanged.
    weights = histogram / jnp.sum(histogram)
    centers = (jnp.arange(nbins, dtype=dtype) + 0.5) / nbins - 0.5
    zero = jnp.zeros(1, dtype=dtype)
    mass = jnp.concatenate((zero, jnp.cumsum(weights)))
    moment = jnp.concatenate((zero, jnp.cumsum(weights * centers)))
    starts = jnp.arange(nbins, dtype=jnp.int32)
    offsets = jnp.arange(tile_size, dtype=jnp.int32)
    tiles = jnp.arange((nbins + tile_size - 1) // tile_size, dtype=jnp.int32)
    initial = jnp.full(nbins + 1, -jnp.inf, dtype=dtype).at[0].set(0)

    def layer(previous, class_number):
        def evaluate_tile(tile):
            ends = tile * tile_size + offsets + 1
            safe_ends = jnp.minimum(ends, nbins)
            interval_mass = mass[safe_ends, None] - mass[None, :-1]
            interval_moment = moment[safe_ends, None] - moment[None, :-1]

            # A class must have positive mass; zero-count bins may belong
            # to a class but must not become an empty class of their own.
            valid = ((starts[None, :] < ends[:, None])
                     & (starts[None, :] >= class_number - 1)
                     & (ends[:, None] <= nbins)
                     & (interval_mass > 0))
            score = (previous[None, :-1] + interval_moment**2
                     / jnp.where(interval_mass > 0, interval_mass, 1))
            score = jnp.where(valid, score, -jnp.inf)
            splits = jnp.argmax(score, axis=1).astype(jnp.int32)
            best = jnp.take_along_axis(score, splits[:, None], axis=1)[:, 0]
            return best, splits

        scores, splits = lax.map(evaluate_tile, tiles)
        current = jnp.concatenate((jnp.full(1, -jnp.inf, dtype=dtype),
                                   scores.reshape(-1)[:nbins]))
        return current, splits.reshape(-1)[:nbins]

    final, parents = lax.scan(layer, initial, jnp.arange(1, classes + 1))
    breaks = jnp.zeros(classes - 1, dtype=jnp.int32)

    def backtrack(step, state):
        end, result = state
        split = parents[classes - step - 1, end - 1]
        result = result.at[classes - step - 2].set(split)
        return split, result

    _, breaks = lax.fori_loop(0, classes - 1, backtrack, (nbins, breaks))
    return breaks, final[-1]


def _positive_integer(value, name):
    """Accept integer parameters without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def threshold_multiotsu(image, classes=3, nbins=512, *, device=None,
                       precision="float32", tile_size=128, progress=True):
    """Return classes-1 optimal histogram-edge thresholds for one image.

    Parameters
    ----------
    image : numpy.ndarray, shape (Y, X), dtype uint16 or float32
        Finite input intensities. Processing does not modify the image.
    classes : int, default 3
        Number of nonempty intensity classes. One class returns no thresholds.
    nbins : int, default 512
        Uniform bins spanning the image minimum through maximum. Applies to
        uint16 as well as float32; integer inputs do not force 65536 bins.
    device : jax.Device or None
        Target device; None uses JAX's default device.
    precision : {"float32", "float64"}
        Search arithmetic. Float64 requires JAX x64 and a compatible backend.
    tile_size : int, default 128
        Maximum endpoints evaluated in one tile. Smaller values reduce
        temporary memory; larger values expose more parallel work.
    progress : bool, default True
        Show terminal progress through histogram and search stages. Stages
        have unequal durations; no intermediate search progress is available.

    Returns
    -------
    thresholds : numpy.ndarray, shape (classes-1,), dtype float64
        Increasing bin edges in original intensity units. Float64 output
        preserves the CPU histogram edges, independently of search precision.

    Raises
    ------
    ValueError
        For invalid inputs or fewer occupied bins than requested classes.
        A constant image supports only classes=1.
    FloatingPointError
        If search precision cannot represent a valid nonempty partition.
    """
    with _stage_progress(2, progress) as bar:
        thresholds = _threshold_multiotsu(
            image, classes, nbins, device=device, precision=precision,
            tile_size=tile_size, bar=bar)
        bar.set_description_str("Multi-Otsu complete")
        return thresholds


def _stage_progress(total, enabled):
    """Create a terminal-only bar without a misleading time estimate."""
    return tqdm(total=total, desc="Histogram", unit="stage", ascii=True,
                disable=not enabled or sys.stderr is None,
                bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} stages [{elapsed}]")


def _threshold_multiotsu(image, classes, nbins, *, device, precision,
                        tile_size, bar):
    """Share threshold computation between the two public progress bars."""
    classes = _positive_integer(classes, "classes")
    nbins = _positive_integer(nbins, "nbins")
    tile_size = _positive_integer(tile_size, "tile_size")
    if classes > nbins:
        raise ValueError("classes must not exceed nbins")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if precision == "float64" and not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before requesting float64")
    if precision == "float64":
        target = device or jax.config.jax_default_device or jax.devices()[0]
        if target.platform.lower() in ("metal", "mps"):
            raise ValueError("Metal does not support float64; use precision='float32'")

    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2-D array")
    if image.dtype not in (np.dtype("uint16"), np.dtype("float32")):
        raise ValueError("image dtype must be uint16 or float32")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite intensities")
    if classes == 1:
        bar.set_description_str("One class: histogram and search skipped")
        bar.update(2)
        return np.empty(0, dtype=np.float64)
    if image.min() == image.max():
        raise ValueError("a constant image cannot form multiple nonempty classes")

    # Float64 histogram edges preserve bin resolution even when float32
    # values have a large offset or a narrow intensity range.
    histogram, edges = np.histogram(image.astype(np.float64), bins=nbins)
    if np.count_nonzero(histogram) < classes:
        raise ValueError("fewer occupied histogram bins than requested classes")
    bar.update(1)
    bar.set_description_str("Threshold search (includes compilation on first call)")
    counts = jax.device_put(histogram.astype(precision), device)
    breaks, score = _solve_histogram(counts, classes=classes,
                                     tile_size=min(tile_size, nbins))
    breaks = np.asarray(breaks)
    if not np.isfinite(np.asarray(score)):
        raise FloatingPointError("no valid partition at the requested precision")

    # Validate nonempty classes in the original integer counts as well.
    totals = np.concatenate(([0], np.cumsum(histogram)))
    boundaries = np.concatenate(([0], breaks, [nbins]))
    if np.any(np.diff(boundaries) <= 0) or np.any(np.diff(totals[boundaries]) <= 0):
        raise FloatingPointError("search returned an empty class; try float64")
    bar.update(1)
    return edges[breaks]


def multi_otsu(image, classes=3, nbins=512, *, device=None,
               precision="float32", tile_size=128, progress=True):
    """Return thresholds and a same-shape int32 intensity-class image.

    Labels run from 0 (darkest) through classes-1 (brightest). Parameters and
    validation match threshold_multiotsu. Pixels at a threshold enter the
    brighter class. 
    """
    with _stage_progress(3, progress) as bar:
        thresholds = _threshold_multiotsu(
            image, classes, nbins, device=device, precision=precision,
            tile_size=tile_size, bar=bar)
        bar.set_description_str("Assigning intensity-class labels")
        labels = np.searchsorted(thresholds, np.asarray(image), side="right").astype(np.int32)
        bar.update(1)
        bar.set_description_str("Multi-Otsu complete")
        return thresholds, labels
