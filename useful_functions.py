"""

All functions needed by more than one script live here so that the logic is
defined and maintained in one place. The individual run-scripts
(centroid_align_xy.py, movie_from_nd2.py, find_threshold.py) are thin entry
points that handle argument parsing and I/O only.


"""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import nd2
import numpy as np
import tifffile
import yaml

# Import PyTorch inside the helpers that need it so CPU-only tools do not
# load its OpenMP runtime alongside the NumPy runtime at startup.
from scipy.ndimage import gaussian_filter, label
from tqdm import tqdm
import cv2  # TODO: Add skimage.filters.threshold_otsu fallback for outside distribution


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nd2(file_path):
    """Load a Nikon ND2 file and return the full raw array plus acquisition metadata.

    Returns the array as (T, P, Z, C, Y, X) in the file's native dtype (usually uint16).
    Callers that need a 2-D representation should pass the result to max_project_z() below.

    We return the native dtype to save memory and avoid premature casting.
    Callers should cast to float32 if they need (e.g. for Gaussian smoothing,any tensor operations, etc.). The raw ND2 values are typically
    uint16.

    Parameters
    ----------
    file_path : str or Path
        Absolute or relative path to the .nd2 file.

    Returns
    -------
    data : ndarray, shape (T, P, Z, C, Y, X), dtype (native, usually uint16)
    channel_names : list of str
        Human-readable names from the microscope metadata (e.g. "Venus",
        "Brightfield"). Preserved so output filenames and OME-TIFF metadata
        are readable without looking up index numbers.
    vox : VoxelSize
        Physical pixel size in µm (.x, .y, .z). Written into the OME-TIFF metadata
        so downstream tools (Fiji, napari,etc.) can display images at correct scale.
    period_s : float or None
        Time between frames in seconds. None if the file has no TimeLoop block
        (i.e. it is not a timelapse acquisition).
    """
    f = nd2.ND2File(file_path)

    # asarray() loads the entire file into memory as a NumPy array.
    # The axis order (T, P, Z, C, Y, X) is the nd2 library's default for
    # multi-position timelapse experiments and is the convention assumed
    # throughout this pipeline.
    data = f.asarray()
    if "P" not in f.sizes:
        # In case of ND2s where there is no P axis, insert a singleton (set with only one element) P axis
        # so the returned array always has shape (T, P, Z, C, Y, X).
        data = data[:, np.newaxis]

    channel_names = [ch.channel.name for ch in f.metadata.channels]

    vox = f.voxel_size()

    # Extract acquisition period from the TimeLoop experiment block if present.
    # We convert from ms to s here so callers always work in SI units.
    # next(..., None) avoids a StopIteration exception when the block is absent.
    period_s = next(
        (loop.parameters.periodMs / 1000.0
         for loop in f.experiment if loop.type == "TimeLoop"),
        None,
    )

    # Close the file handle now that the array is in memory.
    # Leaving it open would hold a file lock for the duration of the script.
    f.close()

    return data, channel_names, vox, period_s


def load_nd2_metadata(file_path):
    """Load channel names, voxel size, and acquisition period from an ND2 file
    without reading any image data into memory.

    load_nd2 calls f.asarray() which loads the entire image volume — up to
    several gigabytes for a typical timelapse. Scripts that only need metadata
    (channel names, physical voxel size, time interval) should call this
    function instead to avoid. The z-alignment script, for example,
    reads image data from already-written OME-TIFFs and only needs the ND2
    for the physical calibration values that were stored there at acquisition
    time.

    The return signature is a subset of load_nd2 (minus the data array) so
    that callers can switch between the two functions without restructuring
    their unpacking.

    Parameters
    ----------
    file_path : str or Path
        Path to the .nd2 file.

    Returns
    -------
    channel_names : list of str
    vox           : VoxelSize  (.x, .y, .z in µm)
    period_s      : float or None
    """
    f = nd2.ND2File(file_path)

    channel_names = [ch.channel.name for ch in f.metadata.channels]

    vox = f.voxel_size()

    # Same TimeLoop extraction as load_nd2
    period_s = next(
        (loop.parameters.periodMs / 1000.0
         for loop in f.experiment if loop.type == "TimeLoop"),
        None,
    )

    # Close the file handle now that we have the metadata. 
    f.close()

    return channel_names, vox, period_s


def load_tif_metadata(file_path):
    """Load channel names, voxel size, and acquisition period from an
    OME-TIFF written by save_ome_tiff, without reading any image data.

    Reads the JSON stored in the ImageDescription tag by save_ome_tiff.
    Returns the same (channel_names, vox, period_s) signature as
    load_nd2_metadata so callers can switch between the two without
    restructuring their unpacking.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    channel_names : list of str
    vox           : SimpleNamespace  (.x, .y, .z in µm)
    period_s      : float or None
    """
    import json
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace

    with tifffile.TiffFile(file_path) as tif:
        C = tif.series[0].shape[1]  # TCZYX — C is axis 1
        try:
            # save_ome_tiff stores the metadata dict as JSON in the ImageDescription
            # tag of the first page. tif.pages[0].description decodes that tag as a string.
            meta = json.loads(tif.pages[0].description)
            channel_names = meta["Channel"]["Name"]
            # SimpleNamespace provides the .x, .y, .z attribute interface that save_ome_tiff expects.
            vox = SimpleNamespace(
                x=meta["PhysicalSizeX"],
                y=meta["PhysicalSizeY"],
                z=meta["PhysicalSizeZ"],
            )
            period_s = meta.get("TimeIncrement")
        except Exception as e:
            print(f"  Could not read save_ome_tiff JSON metadata ({e}), trying OME-XML ...")
            try:
                # tif.ome_metadata returns the raw OME-XML string embedded by external
                # software (e.g. Fiji, Imaris). Parse it with the stdlib XML parser.
                root = ET.fromstring(tif.ome_metadata)
                # OME-XML tags are namespace-qualified, e.g.
                # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels.
                # Extract the namespace URI from the root tag so XPath queries work
                # regardless of which OME schema version the file uses.
                ns = root.tag.split('}')[0].lstrip('{') if '}' in root.tag else ''
                # Find the Pixels element anywhere in the tree using a namespace-aware path.
                p = root.find(f'.//{{{ns}}}Pixels')
                channel_names = [
                    ch.get('Name', f'ch{i}')
                    for i, ch in enumerate(p.findall(f'{{{ns}}}Channel'))
                ]
                # SimpleNamespace provides the .x, .y, .z attribute interface that save_ome_tiff expects.
                vox = SimpleNamespace(
                    x=float(p.get('PhysicalSizeX')),
                    y=float(p.get('PhysicalSizeY')),
                    z=float(p.get('PhysicalSizeZ')),
                )
                # TimeIncrement is optional in OME-XML absent means no timelapse.
                period_s = float(p.get('TimeIncrement')) if p.get('TimeIncrement') else None
            except Exception as e2:
                print(f"  Could not read OME-XML metadata ({e2}), using dummy values.")
                channel_names = [f"ch{i}" for i in range(C)]
                vox = SimpleNamespace(x=1.0, y=1.0, z=1.0)
                period_s = None

    return channel_names, vox, period_s


class LazyTifReader:
    """Lazy reader for OME-TIFFs that loads one timepoint at a time.

    Opens the TIF file once and reads individual timepoints on demand via
    tifffile page indexing, avoiding the need to load the entire volume into
    memory. Each call to read_frame returns a (C, Z, Y, X) array in the
    file's native dtype.

    Parameters
    ----------
    file_path : str or Path
        Path to an OME-TIFF written by save_ome_tiff with shape (T, C, Z, Y, X).
    """

    def __init__(self, file_path):
        self.tif = tifffile.TiffFile(file_path)
        self.series = self.tif.series[0]
        self.T, self.C, self.Z, self.Y, self.X = self.series.shape
        self._pages_per_t = self.C * self.Z
        self.dtype = self.series.dtype

    def read_frame(self, t):
        """Return timepoint t as a (C, Z, Y, X) array in native dtype."""
        start = t * self._pages_per_t
        return np.stack(
            [self.series.pages[start + i].asarray() for i in range(self._pages_per_t)]
        ).reshape(self.C, self.Z, self.Y, self.X)

    def close(self):
        """Close the underlying TIF file handle."""
        self.tif.close()

class LazyNd2Reader:
    """Lazy reader for ND2 files that loads one timepoint/position at a time.

    Uses dask (bundled with the nd2 package) for lazy indexing so that only
    the requested (C, Z, Y, X) subvolume is read from disk.

    Parameters
    ----------
    file_path : str or Path
        Path to a .nd2 file with shape (T, P, Z, C, Y, X).
    """

    def __init__(self, file_path):
        self._nd2 = nd2.ND2File(file_path)
        self._dask = self._nd2.to_dask()
        sizes = self._nd2.sizes
        if "P" not in sizes:
            # Match load_nd2() and nd2_to_tif.py: expose every ND2 as
            # (T, P, Z, C, Y, X), using a singleton P axis for single-position
            # files whose native dask array is only (T, Z, C, Y, X).
            self._dask = self._dask[:, np.newaxis]
        self.T = sizes.get("T", 1)
        self.P = sizes.get("P", 1)
        self.Z = sizes.get("Z", 1)
        self.C = sizes.get("C", 1)
        self.Y = sizes.get("Y", 0)
        self.X = sizes.get("X", 0)
        self.channel_names = [ch.channel.name for ch in self._nd2.metadata.channels]
        self.dtype = self._nd2.dtype

    def read_frame(self, t, p=0):
        """Return one timepoint and position as a (C, Z, Y, X) array.

        The ND2 native axis order after indexing out T and P is (Z, C, Y, X).
        We transpose to (C, Z, Y, X) to match LazyTifReader's convention.
        """
        frame = self._dask[t, p].compute()  # (Z, C, Y, X)
        return frame.transpose(1, 0, 2, 3)  # (C, Z, Y, X)

    def close(self):
        """Close the underlying ND2 file handle."""
        self._nd2.close()


def max_project_z(data):
    return data.max(axis=2)

# ── XY centroid detection ─────────────────────────────────────────────────────
#
# All functions in this section operate on 2-D (Y, X) images and return
# quantities in pixel coordinates (y, x). we are detecting and correcting lateral drift only.

# ── Helper threshold functions ───────────────────────────────────────────────

def normalize_threshold_params(params):
    """Return a complete threshold parameter dict with defaults filled in."""
    out = dict(params)
    out["method"] = out.get("method", "percentile")
    out["block_size"] = out.get("block_size", 64)
    out["invert"] = out.get("invert", False)
    out["levels"] = out.get("levels", 3)
    out["selected_levels"] = out.get("selected_levels", None)
    return out


def get_threshold_params_for_timepoint(cfg, t):
    """Return threshold parameters for timepoint t from old or new YAML schema.

    New YAML files may include a fully materialized timepoint_parameters list
    where index t applies to timepoint t. Older YAML files only have the
    top-level parameters block; those continue to use the same parameters for
    every timepoint.
    """
    per_t = cfg.get("timepoint_parameters")
    if per_t is None:
        return normalize_threshold_params(cfg["parameters"])
    if t < 0 or t >= len(per_t):
        raise IndexError(
            f"timepoint_parameters has {len(per_t)} entries, cannot read T={t}"
        )
    return normalize_threshold_params(per_t[t])

def _threshold_percentile(smoothed, percentile):
    return smoothed > np.percentile(smoothed, percentile)

def _threshold_global_otsu(smoothed):
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmax > vmin:
        uint8_img = np.clip((smoothed - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    else:
        uint8_img = smoothed.astype(np.uint8)
        
    thresh_uint8, _ = cv2.threshold(uint8_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_float = thresh_uint8 / 255.0 * (vmax - vmin) + vmin
    return smoothed > thresh_float

MAX_OTSU_LEVELS = 10
MULTIOTSU_NBINS = 1024


def multiotsu_thresholds_and_classes(smoothed, levels):
    """Return JAX multi-Otsu boundaries and intensity classes for one image.

    Preview and alignment share the same bin count and threshold convention.
    Boundaries are histogram bin edges in original intensity units; pixels
    equal to an edge enter the brighter class. Uniform or insufficiently
    varied frames return no boundaries and an all-zero class map.
    """
    # Load JAX only when multi-level Otsu is requested. Other threshold
    # methods do not need to initialize a JAX backend.
    from multi_otsu_jax import multi_otsu

    levels = int(np.clip(levels, 2, MAX_OTSU_LEVELS))
    image = np.asarray(smoothed)
    if image.dtype not in (np.dtype("uint16"), np.dtype("float32")):
        image = image.astype(np.float32)
    try:
        return multi_otsu(image, classes=levels, nbins=MULTIOTSU_NBINS, progress=False)
    except ValueError:
        # Preserve the existing fallback for blank or saturated frames.
        # The default brightest-class selection then produces an empty mask.
        return np.array([], dtype=np.float64), np.zeros(image.shape, dtype=int)


def multiotsu_class_map(smoothed, levels):
    """Split an image into intensity classes using multi-level Otsu.

    Returns an integer array the same shape as the input where each pixel holds
    the index of the class it falls into, 0 for the darkest class up to
    levels-1 for the brightest. Callers decide which classes count as
    foreground, so the same class map serves both the mask and the per-class
    overlay in find_threshold.py.

    The JAX dynamic program uses MULTIOTSU_NBINS uniform histogram bins.
    Levels remain capped at MAX_OTSU_LEVELS to preserve the existing control
    range. Histogram boundaries and class assignments match the preview.

    Parameters
    ----------
    smoothed : ndarray, shape (Y, X)
        Gaussian-blurred 2-D image, same input as find_largest_mask_xy.
    levels : int
        Number of intensity classes to split the image into, from 2 to
        MAX_OTSU_LEVELS. Two levels reproduces a plain global Otsu threshold.

    Returns
    -------
    class_map : ndarray of int, shape (Y, X)
        Class index per pixel, in the range 0 to levels-1.
    """
    _, class_map = multiotsu_thresholds_and_classes(smoothed, levels)
    return class_map


def _threshold_multiotsu(smoothed, levels, selected_levels, class_map=None):
    if not selected_levels:
        return np.zeros(smoothed.shape, dtype=bool)
    if class_map is None:
        class_map = multiotsu_class_map(smoothed, levels)
    return np.isin(class_map, list(selected_levels))


def _threshold_percentile_otsu_roi(smoothed, percentile):
    binary_base = _threshold_percentile(smoothed, percentile)
    labeled_base, _ = label(binary_base)
    sizes_base = np.bincount(labeled_base.ravel())
    sizes_base[0] = 0
    
    if sizes_base.max() == 0:
        return binary_base
        
    roi_mask = labeled_base == sizes_base.argmax()
    roi_pixels = smoothed[roi_mask]
    
    vmin, vmax = roi_pixels.min(), roi_pixels.max()
    if vmax > vmin:
        uint8_roi = np.clip((roi_pixels - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        uint8_roi_2d = uint8_roi.reshape(-1, 1)
        thresh_uint8, _ = cv2.threshold(uint8_roi_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_float = thresh_uint8 / 255.0 * (vmax - vmin) + vmin
        return smoothed > thresh_float
    return binary_base

def _threshold_local_otsu_torch(smoothed, percentile, block_size=32, interp=True):
    import torch
    import torch.nn.functional as F

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    H, W = smoothed.shape
    vmin, vmax = smoothed.min(), smoothed.max()
    
    if vmax <= vmin:
        return smoothed > vmin
        
    smoothed_t = torch.tensor(smoothed, dtype=torch.float32, device=device)
    img_scaled = ((smoothed_t - vmin) / (vmax - vmin) * 255.0).clamp(0, 255).to(torch.uint8)
    bins = torch.arange(256, device=device, dtype=torch.float32)
    
    def batched_otsu(patches):
        B, P = patches.shape
        hist = torch.zeros(B, 256, device=device)
        hist.scatter_add_(1, patches.long(), torch.ones_like(patches, dtype=torch.float32))
        
        weight1 = hist.cumsum(dim=1)
        weight2 = P - weight1
        
        cumsum_sum = (hist * bins.unsqueeze(0)).cumsum(dim=1)
        total_mean = cumsum_sum[:, -1:]
        
        mean1 = cumsum_sum / (weight1 + 1e-8)
        mean2 = (total_mean - cumsum_sum) / (weight2 + 1e-8)
        
        variance12 = weight1 * weight2 * (mean1 - mean2).pow(2)
        return variance12.argmax(dim=1)
        
    if not interp:
        # Pixel-by-pixel (Sliding window) Processed row-by-row to try and avoid memory OOM
        pad_left = block_size // 2
        pad_right = pad_left - 1 if block_size % 2 == 0 else pad_left
        pad_top = block_size // 2
        pad_bottom = pad_top - 1 if block_size % 2 == 0 else pad_top
        
        padded = F.pad(img_scaled.float().unsqueeze(0).unsqueeze(0), 
                       (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').squeeze()
        out_thresh = torch.zeros((H, W), dtype=torch.float32, device=device)
        
        for y in range(H):
            band = padded[y : y + block_size, :] 
            band_4d = band.unsqueeze(0).unsqueeze(0)
            patches = F.unfold(band_4d, kernel_size=block_size) 
            patches = patches.squeeze(0).transpose(0, 1) 
            row_thresh = batched_otsu(patches) 
            out_thresh[y] = row_thresh.float()
            
        thresh_float = (out_thresh / 255.0) * (vmax - vmin) + vmin
        local_mask = smoothed > thresh_float.cpu().numpy()
        global_mask = smoothed > np.percentile(smoothed, percentile)
        return local_mask & global_mask
        
    else:
        # Block-Interpolated
        grid_h, grid_w = max(1, H // block_size), max(1, W // block_size)
        h_clean = grid_h * block_size
        w_clean = grid_w * block_size
        clean_img = img_scaled[:h_clean, :w_clean].unsqueeze(0).unsqueeze(0).float()
        
        patches = F.unfold(clean_img, kernel_size=block_size, stride=block_size)
        patches = patches.squeeze(0).transpose(0, 1) 
        
        block_thresh = batched_otsu(patches) 
        block_thresh_2d = block_thresh.view(1, 1, grid_h, grid_w).float()
        
        upscaled = F.interpolate(block_thresh_2d, size=(H, W), mode='bilinear', align_corners=False)
        upscaled = upscaled.squeeze()
        
        thresh_float = (upscaled / 255.0) * (vmax - vmin) + vmin
        local_mask = smoothed > thresh_float.cpu().numpy()
        global_mask = smoothed > np.percentile(smoothed, percentile)
        return local_mask & global_mask


def find_largest_mask_xy(smoothed, percentile, method='percentile', block_size=32, invert=False,
                         levels=3, selected_levels=None, class_map=None):
    """Threshold a 2-D image and return the mask and centroid of the largest blob.

    This is the core detection step shared by find_threshold.py (for overlay
    visualisation) and centroid_align_xy.py (for computing shifts). Defining
    it once ensures that what the user sees in the Streamlit preview is exactly
    what the alignment script will detect.

    The method parameter selects between percentile-only (the default), a global
    Otsu threshold across the entire frame, or applying an Otsu refinement step
    only after determining the ROI using the given percentile.

    We take the *largest* connected component rather than all components above
    threshold because in a multi-embryo field there can be small bright
    artifacts or reflections. The embryo body is almost always the largest
    object in the frame.

    Both the mask and the centroid are returned because find_threshold.py needs
    the mask for the red overlay visualisation, while centroid_align_xy.py only
    needs the centroid coordinates. Returning both avoids running the
    connected-component labelling twice when both are needed.

    Parameters
    ----------
    smoothed : ndarray, shape (Y, X)
        Gaussian-blurred 2-D image. The caller is responsible for blurring
        before calling this function so that the blur radius can be tuned
        independently from the threshold percentile.
    percentile : float
        Pixels above this percentile of the image are included in the binary
        mask. Typical values are 85–95 depending on how bright the embryo is
        relative to background. Used when method 'percentile' or 'percentile_otsu_roi' is active.
    method : str
        One of 'percentile' (default), 'multiotsu', 'global_otsu',
        'percentile_otsu_roi', 'local_otsu_interp', or 'local_otsu_pixel'.
    levels : int
        Number of intensity classes for the 'multiotsu' method. Ignored by
        every other method.
    selected_levels : list of int or None
        Which multi-level Otsu classes count as foreground, given as class
        indices from 0 for the darkest to levels-1 for the brightest. Any
        combination is accepted, including non-adjacent classes. None selects
        the brightest class alone. Ignored by every other method.
    class_map : ndarray of int or None
        Pre-computed output of multiotsu_class_map for this image. Callers
        that already hold one pass it in to skip repeating the threshold
        search, which is what find_threshold.py does so that toggling a class
        checkbox redraws immediately. None computes it as needed.

    Returns
    -------
    mask : ndarray of bool, shape (Y, X)
        True where the pixel belongs to the largest connected component.
    centroid : ndarray, shape (2,)
        [cy, cx] — row, column coordinates of the centroid. If no component
        is found (e.g. a blank frame or threshold too high), returns the frame
        centre so that the downstream shift is zero and the frame is unchanged.
    """
    if method == "multiotsu":
        if selected_levels is None:
            selected_levels = [int(np.clip(levels, 2, MAX_OTSU_LEVELS)) - 1]
        binary = _threshold_multiotsu(smoothed, levels, selected_levels, class_map=class_map)
    elif method == "global_otsu":
        binary = _threshold_global_otsu(smoothed)
    elif method == "percentile_otsu_roi":
        binary = _threshold_percentile_otsu_roi(smoothed, percentile)
    elif method == "local_otsu_interp":
        binary = _threshold_local_otsu_torch(smoothed, percentile, block_size=block_size, interp=True)
    elif method == "local_otsu_pixel":
        binary = _threshold_local_otsu_torch(smoothed, percentile, block_size=block_size, interp=False)
    else:
        binary = _threshold_percentile(smoothed, percentile)
        
    if invert:
        binary = ~binary

    # Label connected components. Default 4-connectivity is sufficient for
    # detecting embryo blobs. 8-connectivity would merge diagonally adjacent
    # objects but is not needed here.
    labeled, _ = label(binary,structure=np.ones((3, 3)))

    # Count pixels per component. Component 0 is background so zeroing it out
    # ensures argmax never selects the background as the "largest component".
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    if sizes.max() == 0:
        # No foreground component found, can happen in very early/late frames
        # if the embryo is out of focus or if the threshold is too aggressive.
        # Returning the frame centre means the downstream shift will be (0, 0),
        # leaving the frame unchanged rather than crashing or producing garbage.
        cy, cx = smoothed.shape[0] / 2, smoothed.shape[1] / 2
        mask = np.zeros(smoothed.shape, dtype=bool)
        return mask, np.array([cy, cx])

    # Build a boolean mask containing only the largest component.
    mask = labeled == sizes.argmax()

    # Centroid = mean (row, col) position of all True pixels.
    # np.argwhere returns the indices of True elements as an (N, 2) array,
    # so .mean(axis=0) gives [mean_row, mean_col] = [cy, cx].
    centroid = np.argwhere(mask).mean(axis=0)

    return mask, centroid


def compute_shift_xy(frame, sigma, percentile, ch_idx, method='percentile', block_size=32, invert=False,
                    levels=3, selected_levels=None):
    """Return the (dy, dx) translation that moves the embryo centroid to the frame centre.

    This function answers: "by how many pixels must we shift this frame so
    that the embryo ends up centred?" the shift is
    done by align_frame_xy(), or directly with scipy.ndimage.shift in the
    --enlarge_canvas two-pass workflow.

    Separating shift computation from application is important for the
    --enlarge_canvas workflow in centroid_align_xy.py, where all shifts across
    all timepoints must be known before the canvas can be padded. Fusing
    computation and application would require either a second pass over the
    data or storing large intermediate arrays.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X)
        A single timepoint for a single embryo, with all channels and Z-slices.
    sigma : float
        Gaussian blur radius in pixels. Higher values smooth over noise but
        reduce sensitivity to fine embryo boundary details. Typical values
        are 15–40 for PSM imaging.
    percentile : float
        Threshold percentile passed to find_largest_mask_xy.
    ch_idx : int
        Index of the channel used for centroid detection (e.g. Venus = 0).
        We use a single channel for detection so that the computed shift is
        consistent across all channels

    Returns
    -------
    dy : float
        Pixels to shift in Y. Positive = move image content downward
        (scipy convention: positive shift moves the array values in the
        positive-index direction).
    dx : float
        Pixels to shift in X. Positive = move image content rightward.
    """
    Y, X = frame.shape[2], frame.shape[3]

    # Max-project the detection channel over Z to collapse to a 2-D image.
    # We project only the one detection channel
    projection = frame[ch_idx].max(axis=0)  # shape (Y, X)

    # Gaussian blur suppresses noise and isolated bright spots that could
    # pull the centroid away from the embryo body.

    smoothed = gaussian_filter(projection, sigma=sigma)

    # Detect the largest blob and get its centroid in pixel coordinates.
    # The mask is discarded here; only the centroid coordinates are needed.
    _, centroid = find_largest_mask_xy(smoothed, percentile, method=method, block_size=block_size,
                                       invert=invert, levels=levels, selected_levels=selected_levels)
    cy, cx = centroid[0], centroid[1]

    # The target position is the frame centre (Y/2, X/2).
    # dy > 0 when the embryo is above centre → we shift the image downward.
    # dy < 0 when the embryo is below centre → we shift the image upward.
    dy = Y / 2 - cy
    dx = X / 2 - cx

    # Always print the computed shift so any caller — whether a batch script,
    # a real-time autofocus loop, or an interactive notebook — automatically
    # gets a record of what the centroid detector decided, without needing to
    # add logging at every call site.
    # tqdm.write is used instead of print so that this output does not
    # visually corrupt any tqdm progress bar that may be running in the caller.
    # If no tqdm bar is active, tqdm.write behaves identically to print.
    # Format: +7.2f means always show the sign (+ or -), right-pad to 7
    # characters wide so columns line up when many lines are printed, and
    # show 2 decimal places.
    tqdm.write(f"dy={dy:+7.2f}  dx={dx:+7.2f}")

    return dy, dx


def compute_shift_xy_gpu(frame, sigma, percentile, ch_idx, method='percentile', block_size=32, invert=False,
                    levels=3, selected_levels=None):
    """GPU-accelerated version of compute_shift_xy.

    Gaussian blur runs on GPU; connected-component labeling stays on CPU.
    Parameters are identical to compute_shift_xy.
    """
    import torch
    import torch.nn.functional as F

    Y, X = frame.shape[2], frame.shape[3]

    device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
) # Determine the best available device for GPU acceleration, else fall back to CPU.
    # Move the entire frame to GPU once at the start.
    t_frame = torch.from_numpy(frame).to(device) # (C, Z, Y, X)

    # Extract the detection channel and max-project over Z to get a 2-D image.
    projection = t_frame[ch_idx].max(dim=0).values  # (Y, X)

    # Gaussian blur suppresses noise and isolated bright spots that could
    # pull the centroid away from the embryo body. conv2d expects (N, C_in, H, W).
    kernel = _gaussian_kernel_2d(sigma, device=device)
    padding = kernel.shape[-1] // 2
    proj_4d = projection.reshape(1, 1, Y, X)
    smoothed_gpu = F.conv2d(proj_4d, kernel, padding=padding)

    # Squeeze to (Y, X) and pull to CPU for connected-component labeling.
    smoothed = smoothed_gpu.squeeze().cpu().numpy()

    _, centroid = find_largest_mask_xy(smoothed, percentile, method=method, block_size=block_size,
                                       invert=invert, levels=levels, selected_levels=selected_levels)
    cy, cx = centroid[0], centroid[1]

    # dy > 0 when the embryo is above centre → shift image downward.
    # dy < 0 when the embryo is below centre → shift image upward.
    dy = Y / 2 - cy
    dx = X / 2 - cx
    tqdm.write(f"dy={dy:+7.2f}  dx={dx:+7.2f}")

    return dy, dx


def align_frame_xy(frame, sigma, percentile, ch_idx, method='percentile', block_size=32, invert=False,
                    levels=3, selected_levels=None):
    """Compute the centring shift and apply it to all channels and Z-slices.

    This is the single-pass alignment function used in the default (no
    --enlarge_canvas) workflow. It wraps compute_shift_xy and the scipy shift
    call into one step so callers do not have to manage the shift value
    separately when they do not need to inspect it before applying it.

    All channels and Z-slices are shifted by the same (dy, dx) because lateral
    embryo position is the same in every channel and Z-slice within one
    timepoint. Shifting them together keeps channels in registration with each
    other, which matters for any downstream co-localisation analysis.

    We use order=1 (bilinear) interpolation. order=0 (nearest-neighbour) is
    faster but creates staircase artefacts on diagonal edges. order=3 (cubic)
    is smoother but noticeably slower, and the improvement is not visible at
    typical display resolutions

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X)
    sigma, percentile, ch_idx : see compute_shift_xy

    Returns
    -------
    shifted : ndarray, shape (C, Z, Y, X)
        Frame shifted so the embryo centroid is at the frame centre.
        Pixels that shift outside the original canvas boundary are filled
        with 0 (black), which is visually unambiguous and does not introduce
        false fluorescence signal.
    dy, dx : float
        The shift that was applied.
    """
    dy, dx = compute_shift_xy(frame, sigma, percentile, ch_idx, method=method, block_size=block_size,
                              invert=invert, levels=levels, selected_levels=selected_levels)

    # Apply the same (dy, dx) shift to every channel and Z-slice simultaneously.
    # The (0, 0) entries for the C and Z axes ensure those axes are untouched.
    # mode="constant", cval=0 fills newly exposed border pixels with black.
    shifted = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0)

    return shifted, dy, dx


def _gaussian_kernel_2d(sigma, device="mps"):
    """Build a normalised 2D Gaussian kernel as a torch tensor.

    Kernel radius is 4*sigma (rounded up), matching scipy's default truncate=4.
    """
    import torch

    radius = int(np.ceil(sigma * 4))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = g1d[:, None] * g1d[None, :]
    kernel /= kernel.sum()
    # conv2d expects (out_channels, in_channels, height, width).
    # Single input and output channel since we blur one grayscale image.
    return kernel.reshape(1, 1, size, size)


def align_frame_xy_gpu(frame, sigma, percentile, ch_idx, method='percentile', block_size=32, invert=False,
                    levels=3, selected_levels=None):
    """GPU-accelerated version of align_frame_xy.

    Gaussian blur and integer-pixel shift run on GPU. Connected-component
    labeling (find_largest_mask_xy) stays on CPU because scipy.ndimage.label
    has no torch equivalent.

    Returns the shifted frame as a torch tensor on the GPU device.
    dy and dx are the raw centroid offsets before rounding.

    Parameters are identical to align_frame_xy.
    """
    import torch
    import torch.nn.functional as F

    C, Z, Y, X = frame.shape

    # Select the best available device: CUDA GPU > Apple MPS GPU > CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

   
    # Move entire frame to GPU once. it stays there until the final result.
        
    t_frame = torch.from_numpy(frame).to(device)  # (C, Z, Y, X)

    # Max-project the detection channel over Z to get a 2D image.
    projection = t_frame[ch_idx].max(dim=0).values  # (Y, X)

    # Gaussian blur on GPU. conv2d expects (N, C_in, H, W).
    kernel = _gaussian_kernel_2d(sigma, device=device)
    padding = kernel.shape[-1] // 2  # same-size output
    proj_4d = projection.reshape(1, 1, Y, X)
    smoothed_gpu = F.conv2d(proj_4d, kernel, padding=padding)

    # Pull the blurred 2D image to CPU for connected-component labeling.
    smoothed = smoothed_gpu.squeeze().cpu().numpy()  # (Y, X)

    _, centroid = find_largest_mask_xy(smoothed, percentile, method=method, block_size=block_size,
                                       invert=invert, levels=levels, selected_levels=selected_levels)
    cy, cx = centroid[0], centroid[1]

    dy = Y / 2 - cy
    dx = X / 2 - cx
    tqdm.write(f"dy={dy:+7.2f}  dx={dx:+7.2f}")

    # Round to integer pixels. sub-pixel interpolation is not needed here.
    idy = int(round(dy))
    idx = int(round(dx))

    # Pad zeros on the side content shifts away from then slice back to
    # original size. Exposed edges become zero, no wrapping occurs.
    padded = F.pad(t_frame, (max(idx, 0), max(-idx, 0), max(idy, 0), max(-idy, 0)))
    shifted_gpu = padded[:, :, max(-idy, 0):max(-idy, 0)+Y, max(-idx, 0):max(-idx, 0)+X]

    return shifted_gpu, dy, dx


def apply_shift_xy_gpu(frame, dy, dx):
    """Apply a precomputed (dy, dx) shift to a frame on GPU.

    Used in the --enlarge_canvas two-pass workflow, where shifts are
    precomputed in Pass 1 and applied in Pass 2.

    The shift is rounded to integer pixels, consistent with align_frame_xy_gpu.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X)
        A single timepoint for a single embryo.
    dy : float
        Shift in Y (positive = move content downward).
    dx : float
        Shift in X (positive = move content rightward).

    Returns
    -------
    shifted_gpu : torch.Tensor, shape (C, Z, Y, X), on GPU
        Caller is responsible for moving to CPU when done.
    """
    import torch
    import torch.nn.functional as F

    C, Z, Y, X = frame.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    t_frame = torch.from_numpy(frame).to(device)  # (C, Z, Y, X)

    idy = int(round(dy))
    idx = int(round(dx))

    # Pad zeros on the side content shifts away from, then slice back to
    # original size. Mirrors the same operation in align_frame_xy_gpu.
    padded = F.pad(t_frame, (max(idx, 0), max(-idx, 0), max(idy, 0), max(-idy, 0)))
    shifted_gpu = padded[:, :, max(-idy, 0):max(-idy, 0)+Y, max(-idx, 0):max(-idx, 0)+X]

    return shifted_gpu


# ── 3-D registration ────────────────────────────────────────────────────────

def compute_z_profile(frame, ch_idx):
    """Return the mean intensity per Z-slice for the chosen channel.

    Averaging over all pixels in each (Y, X) plane produces a 1-D profile
    of signal strength versus Z position. This profile is the input to
    compute_centroid_z and captures how the fluorescent signal is distributed
    along the optical axis.

    We use mean rather than sum so that the profile values are independent
    of image size and can be compared across experiments with different
    field-of-view dimensions.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X), float32
        A single timepoint for a single embryo position.
    ch_idx : int
        Index of the channel to use for Z detection.

    Returns
    -------
    profile : ndarray, shape (Z,), float64
        Mean pixel intensity at each Z-slice.
    """
    # Select the detection channel and average over the spatial axes (Y, X).
    # axis=(1, 2) collapses Y and X simultaneously, leaving one value per
    # Z-slice. We compute over the full (Y, X) frame rather than a subregion
    # because the signal fills the embryo body and using the full frame
    # gives a more stable average than any manually chosen subregion.
    return frame[ch_idx].mean(axis=(1, 2))


def compute_centroid_z(profile):
    """Return the intensity-weighted mean Z position (centroid) of a Z profile.

    The centroid is the continuous analogue of argmax: instead of returning
    the single slice with the highest intensity, it returns the weighted
    average Z position, which is more stable when the peak is broad or when
    two adjacent slices have similar intensities.

    Before computing the weighted average, the minimum value of the profile
    is subtracted from every element. This removes any uniform background
    floor, signal that is present at the same level across all Z-slices and
    therefore carries no information about the embryo's Z position. Without
    this subtraction, a high background would pull the centroid toward Z/2
    regardless of where the actual signal peak is.

    If the background-subtracted profile is all zeros (blank frame or
    signal entirely below background level), the centroid falls back to
    the midpoint of the Z range. This keeps the downstream shift at zero
    rather than producing a NaN or an extreme value that would corrupt the
    alignment.

    Parameters
    ----------
    profile : ndarray, shape (Z,)
        Mean intensity per Z-slice, as returned by compute_z_profile.

    Returns
    -------
    centroid : float
        Intensity-weighted mean Z position in slice units. Can be fractional.
    """
    # Subtract the minimum to remove background before computing the centroid.
    # The minimum is the baseline signal present even in out-of-signal slices;
    # only the excess above this baseline reflects actual Z-localised signal.
    above_background = profile - profile.min()

    total = above_background.sum()

    if total == 0:
        # All slices are equally bright (or all zero) so the profile carries
        # no Z position information. Returning the midpoint means the computed
        # shift will be zero, leaving the frame unchanged rather than crashing.
        return len(profile) / 2.0

    # np.arange gives the slice index for each element of the profile.
    # The weighted sum (index * weight) / total_weight is the standard
    # formula for centre of mass, applied here along the Z axis.
    z_indices = np.arange(len(profile), dtype=float)
    return float((above_background * z_indices).sum() / total)


def compute_shift_z(reference, moving,proj_axis=None,upsample_factor=10):
    """Return the (dz, dy, dx) translation that aligns *moving* to *reference*.

    Uses FFT-based phase cross-correlation on the full 3-D volume rather
    than a 1-D projection or a single representative slice. Working in 3-D
    means the shift estimate incorporates spatial structure along all three
    axes, which makes it more robust than projecting down to Z intensity
    profiles when the embryo signal is sparse or unevenly distributed across
    slices.

    The function returns sub-pixel shifts. Callers that need integer shifts
    (e.g. for discrete slice indexing) should round the result themselves so
    that the rounding policy is explicit at the call site rather than buried
    inside this function.

    Parameters
    ----------
    reference : ndarray, shape (Z, Y, X)
        The volume to align against.
    moving : ndarray, shape (Z, Y, X)
        The volume to be aligned.

    Returns
    -------
    dz : float
        Shift along Z. Positive means the moving volume must move toward
        higher Z indices to match the reference.
    dy : float
        Shift along Y.
    dx : float
        Shift along X.
    """
    from skimage.registration import phase_cross_correlation
    if proj_axis is None:
        shift, _, _ = phase_cross_correlation(reference, moving, upsample_factor=upsample_factor)
        dz, dy, dx = shift
    elif proj_axis =='x':
        # max project along x to get yz plane
        ref_proj = reference.max(axis=2)
        mov_proj = moving.max(axis=2)
        shift, _, _ = phase_cross_correlation(ref_proj, mov_proj, upsample_factor=upsample_factor)
        # Since we are projecting along x, the shift order is (dz, dy) instead of (dz, dy, dx).
        dz, dy = shift
        dx = 0.0  # No shift along x since we projected it out
    elif proj_axis =='y':
        # max project along y to get xz plane
        ref_proj = reference.max(axis=1)
        mov_proj = moving.max(axis=1)
        shift, _, _ = phase_cross_correlation(ref_proj, mov_proj, upsample_factor=upsample_factor)
        # Since we are projecting along y, the shift order is (dz, dx) instead of (dz, dy, dx).
        dz, dx = shift
        dy = 0.0  # No shift along y since we projected it out
    elif proj_axis =='xy':
        # max project along x and y to get z profile
        ref_proj = reference.max(axis=(1,2))
        mov_proj = moving.max(axis=(1,2))
        shift, _, _ = phase_cross_correlation(ref_proj, mov_proj, upsample_factor=upsample_factor)
        dz = shift[0]  # Only shift along z since x and y are projected out
        dy, dx = 0.0, 0.0
    elif proj_axis == 'z':
        # max project along z to get yx plane
        ref_proj = reference.max(axis=0)
        mov_proj = moving.max(axis=0)
        shift, _, _ = phase_cross_correlation(ref_proj, mov_proj, upsample_factor=upsample_factor)
        # Since we are projecting along z, the shift order is (dy, dx) instead of (dz, dy, dx).
        dy, dx = shift
        dz = 0.0  # No shift along z since we projected it out
    else:
        print(f"Invalid proj_axis {proj_axis}, should be one of None, 'x', 'y', 'z', or 'xy'. Defaulting to 'x' ")
    
    return dz, dy, dx


def save_shifts_csv(filepath, shifts):
    """Write per-timepoint (dz, dy, dx) shifts to a CSV file.

    Each row corresponds to one timepoint. The pairwise shifts (dz, dy, dx)
    are stored alongside running totals (total_dz, total_dy, total_dx) that
    accumulate from t=0. All values are in raw pixel and slice units without
    any physical unit conversion, so the CSV is independent of voxel
    calibration. Callers that need physical distances can multiply by the
    appropriate voxel sizes themselves.

    Parameters
    ----------
    filepath : str or Path
        Destination CSV path.
    shifts : array-like, shape (T, 3)
        Each row is (dz, dy, dx) for one timepoint.
    """
    shifts = np.asarray(shifts)
    # Cumulative drift from t=1, rounded to match upsample precision.
    cumulative = np.round(np.cumsum(shifts, axis=0), 1)
    # Timepoints are 1-indexed to match the microscope convention.
    t_col = np.arange(1, len(shifts) + 1).reshape(-1, 1)
    # Each row: t, dz, dy, dx, total_dz, total_dy, total_dx
    rows = np.hstack([t_col, shifts, cumulative])

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'dz', 'dy', 'dx', 'total_dz', 'total_dy', 'total_dx'])
        writer.writerows(rows.tolist())


# ── Format conversion ─────────────────────────────────────────────────────────

def save_ome_tiff(filepath, volume, channel_names, vox, period_s,
                  shape=None, dtype=None):
    """Write a (T, C, Z, Y, X) volume as an OME-TIFF with full physical metadata.

    OME-TIFF is chosen as the output format because it embeds physical pixel
    size, channel names, and acquisition timing in a standardised XML block
    that Fiji (Bio-Formats) and napari can read without any manual calibration.

    The function takes a filepath rather than a directory + naming components
    so that it is not coupled to any particular filename convention. The caller
    decides the name and this function only handles the writing.

    Parameters
    ----------
    filepath : str or Path
        Full destination path including filename, e.g.
        "/data/aligned_nd1188/nd1188_P0.ome.tif".
    volume : ndarray or generator
        When an ndarray, shape (T, C, Z, Y, X). When a generator, must yield
        one (C, Z, Y, X) frame per timepoint, and shape/dtype must be provided.
    channel_names : list of str
        Channel names in the same order as the C axis (e.g. ["Venus", "BF"]).
        Written into the OME metadata so channels are labelled correctly in
        Fiji's channel manager.
    vox : VoxelSize
        Physical voxel size from load_nd2 (.x, .y, .z in µm). Written so
        downstream tools display images at the correct physical scale rather
        than in arbitrary pixel units.
    period_s : float or None
        Time between frames in seconds from load_nd2. Written as TimeIncrement
        so the time axis is correctly calibrated. None is safe to pass -- Fiji
        will simply leave the time axis uncalibrated.
    shape : tuple, optional
        Required when volume is a generator. The full (T, C, Z, Y, X) shape
        so tifffile can write the OME-XML header before consuming frames.
    dtype : numpy dtype, optional
        Required when volume is a generator. The pixel dtype (e.g. np.uint16).
    """
    metadata = {
        "axes": "TCZYX",
        "PhysicalSizeX": vox.x, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": vox.y, "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": vox.z, "PhysicalSizeZUnit": "µm",
        "TimeIncrement": period_s, "TimeIncrementUnit": "s",
        "Channel": {"Name": channel_names},
    }
    # When volume is a generator, shape and dtype must be declared upfront so
    # tifffile can write the OME-XML header before consuming any frames.
    # bigtiff is required for streaming because the final size is not known
    # and will likely exceed the 4 GB limit of tiff.
    kwargs = {}
    if shape is not None:
        kwargs["shape"] = shape
        kwargs["dtype"] = dtype
        kwargs["bigtiff"] = True
    tifffile.imwrite(
        filepath,
        volume,
        photometric="minisblack",
        metadata=metadata,
        **kwargs,
    )



def concatenate_tifs(tif_paths, output_path, axis=0):
    """Concatenate multiple OME-TIFFs along a chosen axis.

    Joins a set of OME-TIFF files by concatenating along the specified
    axis (default 0, which is T in TCZYX volumes). Input files are sorted
    by filename before concatenation — ND2 files are named sequentially
    by the microscope (nd1188, nd1189, ...), so alphabetical order
    corresponds to acquisition time order.

    All dimensions except the concatenation axis must match across files.
    A mismatch is raised as an error.

    OME metadata is compared across files. Inconsistencies are printed as
    warnings but do not prevent concatenation. The OME-XML from the first
    file is passed through to the output.

    Data is streamed to the output one input file at a time using
    tifffile's append mode, so only one input file's data is in memory
    at any point.

    Parameters
    ----------
    tif_paths : list of str or Path
        Paths to OME-TIFF files.
    output_path : str or Path
        Destination path for the concatenated OME-TIFF.
    axis : int, default 0
        Axis along which to concatenate. 0=T, 1=C, 2=Z, 3=Y, 4=X for
        TCZYX volumes.
    """
    # Sort by filename so that files from sequentially named ND2s
    # (nd1188, nd1189, ...) are concatenated in acquisition order.
    sorted_paths = sorted(tif_paths, key=lambda p: Path(p).name)

    # Collect array shapes and OME-XML metadata from each file before
    # loading any pixel data. This pass is used to verify that all files
    # have compatible dimensions and consistent metadata before committing
    # to the concatenation.
    shapes = []
    ome_xmls = []
    for path in sorted_paths:
        # TiffFile opens the file and parses the TIFF header and any
        # OME-XML in the ImageDescription tag. series[0].shape gives the
        # full array shape without reading pixel data into memory.
        # ome_metadata returns the OME-XML as a string if present, or
        # None for plain TIFFs that have no OME block.
        f = tifffile.TiffFile(path)
        shapes.append(f.series[0].shape)
        ome_xmls.append(f.ome_metadata)
        f.close()

    # Verify that all dimensions except the concatenation axis match.
    # For example, when concatenating along axis 0 (T) with shape
    # (T, C, Z, Y, X), the C, Z, Y, X values must be identical across
    # all files. Files with different spatial dimensions or channel counts
    # cannot be concatenated into a single coherent volume.
    ref_shape = shapes[0]
    for i, shape in enumerate(shapes[1:], start=1):
        for dim in range(len(ref_shape)):
            if dim == axis:
                continue
            if shape[dim] != ref_shape[dim]:
                raise ValueError(
                    f"Dimension mismatch on axis {dim}: "
                    f"{Path(sorted_paths[0]).name} has {ref_shape[dim]} "
                    f"but {Path(sorted_paths[i]).name} has {shape[dim]}."
                )

    # Compare OME metadata across files. The OME-XML string encodes
    # voxel sizes, channel names, and time calibration. Files from the
    # same imaging session should have identical metadata, but slight
    # differences (e.g. floating-point rounding in voxel size between
    # acquisition runs) can occur without affecting the image data.
    # These are printed as warnings rather than raised as errors so the
    # user is aware but concatenation still proceeds.
    ref_ome = ome_xmls[0]
    if ref_ome is not None:
        for i, ome in enumerate(ome_xmls[1:], start=1):
            if ome is None:
                tqdm.write(
                    f"Warning: {Path(sorted_paths[i]).name} has no OME metadata"
                )
            elif ome != ref_ome:
                tqdm.write(
                    f"Warning: OME metadata in {Path(sorted_paths[i]).name} "
                    f"differs from {Path(sorted_paths[0]).name}"
                )

    total_along_axis = sum(s[axis] for s in shapes)
    tqdm.write(f"Concatenating {len(sorted_paths)} files -> {total_along_axis} total along axis {axis}")

    # Parse the first file's OME-XML to extract physical calibration
    # (voxel sizes, time interval, channel names). These values are
    # passed to TiffWriter as a metadata dict so that tifffile generates
    # a fresh OME-XML header whose dimension counts (SizeT, SizeC, SizeZ)
    # match the actual number of pages in the concatenated output.
    # Passing the raw XML string directly would preserve the original
    # SizeT from a single ND2 file, causing Fiji to misread the axes.
    import xml.etree.ElementTree as ET

    ome_metadata = {"axes": "TCZYX"}
    if ref_ome is not None:
        root = ET.fromstring(ref_ome)
        # OME-XML wraps every tag in its namespace, e.g.
        # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels.
        # Extracting the namespace from the root tag lets us find
        # child elements without assuming a specific schema version.
        ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
        pixels = root.find(f".//{ns}Pixels")
        if pixels is not None:
            for attr in ["PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement"]:
                val = pixels.get(attr)
                if val is not None:
                    ome_metadata[attr] = float(val)
            for attr in ["PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit", "TimeIncrementUnit"]:
                val = pixels.get(attr)
                if val is not None:
                    ome_metadata[attr] = val

        # Channel names (Venus, mCherry, TD, etc.) so Fiji labels them
        # correctly in the channel manager instead of showing "C=0".
        channels = root.findall(f".//{ns}Channel")
        if channels:
            names = [ch.get("Name") for ch in channels if ch.get("Name")]
            if names:
                ome_metadata["Channel"] = {"Name": names}

    # Build the full output shape by summing along the concatenation
    # axis. The remaining dimensions (C, Z, Y, X) come from the first
    # file since they were already verified to match across all inputs.
    out_shape = list(ref_shape)
    out_shape[axis] = total_along_axis
    out_shape = tuple(out_shape)

    def page_generator():
        """Yield 2D (Y, X) pages from each input file in order.

        Each file's 5D array (T, C, Z, Y, X) is flattened to (T*C*Z, Y, X)
        so that every confocal plane becomes one TIFF page. Reading and
        flattening one file at a time keeps peak memory at the size of a
        single input file rather than the full concatenated volume.
        """
        for path in tqdm(sorted_paths, desc="Concatenating", unit="file"):
            data = tifffile.imread(path).astype(np.float32)
            yield from data.reshape(-1, data.shape[-2], data.shape[-1])
            del data

    # imwrite with a generator and an explicit shape lets tifffile
    # stream pages to disk while knowing the full 5D dimensions
    # upfront. It generates a single OME Image element with the
    # correct SizeT, SizeC, SizeZ, voxel sizes, and channel names.
    tifffile.imwrite(
        output_path,
        page_generator(),
        shape=out_shape,
        dtype=np.float32,
        bigtiff=True,
        photometric="minisblack",
        metadata=ome_metadata,
    )

    tqdm.write(f"Done. Output: {output_path}")


# ── Contrast enhancement ──────────────────────────────────────────────────────

def apply_clahe_2d(frame_2d, clip_limit, tile_size):
    """Apply CLAHE to a single 2D grayscale image and return the result.

    Uses OpenCV's CLAHE implementation, which operates on uint16 natively and
    runs in C++ without holding the Python GIL.

    Parameters
    ----------
    frame_2d : ndarray, shape (Y, X), dtype uint16
        A single grayscale plane.
    clip_limit : float
        Maximum slope of the cumulative distribution function within any tile.
        Values above this threshold are clipped and redistributed uniformly
        across the histogram. Higher values allow stronger contrast enhancement
        but amplify noise proportionally. Typical range: 1.0 – 4.0.
    tile_size : int
        Side length in pixels of each histogram tile. The image is divided into
        a regular grid of non-overlapping (tile_size × tile_size) tiles, each
        equalized independently. Smaller values increase local adaptation at
        the cost of more visible tile boundaries. Must be a positive integer
        that divides evenly into the image dimensions for best results.

    Returns
    -------
    ndarray, shape (Y, X), dtype uint16
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(frame_2d)


def apply_clahe_volume(frame_czyx, clip_limit, tile_size):
    """Apply CLAHE independently to every (Y, X) slice in a (C, Z, Y, X) volume.

    Iterates over all channel and Z-plane combinations and calls apply_clahe_2d
    on each 2D slice. The result preserves the input shape and dtype.

    Parameters
    ----------
    frame_czyx : ndarray, shape (C, Z, Y, X), dtype uint16
    clip_limit : float
        Passed to apply_clahe_2d. See its docstring.
    tile_size : int
        Passed to apply_clahe_2d. See its docstring.

    Returns
    -------
    ndarray, shape (C, Z, Y, X), dtype uint16
    """
    out = np.empty_like(frame_czyx)
    for c in range(frame_czyx.shape[0]):
        for z in range(frame_czyx.shape[1]):
            out[c, z] = apply_clahe_2d(frame_czyx[c, z], clip_limit, tile_size)
    return out


# ── Workflow helpers ─────────────────────────────────────────────

def run_nd2_folder_conversion(nd2_folder, output_dir=None, position=None, parallel=True):
    """Convert all ND2 files in a folder to per-position OME-TIFFs and concatenate.

    This function mirrors nd2_to_tif.py so that GUI code
    can reuse the exact conversion flow without modifying existing scripts.
    """
    nd2_folder = Path(nd2_folder)
    if not nd2_folder.is_dir():
        raise ValueError(f"Not a directory: {nd2_folder}")

    nd2_files = sorted(nd2_folder.glob("*.nd2"))
    if not nd2_files:
        raise ValueError(f"No .nd2 files found in {nd2_folder}")

    output_dir = Path(output_dir) if output_dir else nd2_folder / "tifs"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_positions = None
    for nd2_path in nd2_files:
        channel_names, vox, period_s = load_nd2_metadata(nd2_path)
        f = nd2.ND2File(nd2_path)
        data = f.to_dask()
        _, P, _, _, _, _ = data.shape
        target_positions = [position] if position is not None else list(range(P))
        base = nd2_path.stem

        def _convert_position(p):
            volume = data[:, p].compute().transpose(0, 2, 1, 3, 4)  # (T, C, Z, Y, X)
            out_path = output_dir / f"{base}_P{p}.ome.tif"
            save_ome_tiff(out_path, volume, channel_names, vox, period_s)

        if parallel and position is None:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(_convert_position, target_positions))
        else:
            for p in target_positions:
                _convert_position(p)

        f.close()

    if target_positions is None:
        return output_dir

    for p in target_positions:
        tifs_for_position = sorted(output_dir.glob(f"*_P{p}.ome.tif"))
        if len(tifs_for_position) < 2:
            continue
        concat_path = output_dir / f"P{p}_concat.ome.tif"
        concatenate_tifs(tifs_for_position, concat_path)

    return output_dir

def run_xy_alignment_from_yaml(
    yaml_file,
    use_nd2=False,
    fps=2.0,
    no_enlarge_canvas=False,
    use_gpu=False,
    low_memory=False,
    progress_cb=None,
):
    """Run XY alignment in-process using the same config logic as centroid_align_xy.py."""
    yaml_path = Path(yaml_file)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    tif_file = None
    if use_nd2:
        nd2_path = Path(cfg["source"]["file"])
        if not nd2_path.is_file():
            raise FileNotFoundError(f"ND2 file not found: {nd2_path}")
        data, channel_names, vox, period_s = load_nd2(nd2_path)
        data = data.astype(np.float32)
        T, P, Z, C, Y, X = data.shape
        base = nd2_path.stem
        out_dir = nd2_path.parent / f"aligned_{base}"
    else:
        src_file = Path(cfg["source"]["file"])
        if src_file.is_file() and src_file.suffix.lower() in [".tif", ".tiff"]:
            tif_path = src_file
            base = tif_path.stem[:-4] if tif_path.stem.endswith(".ome") else tif_path.stem
        else:
            try:
                p_id = "P" + yaml_path.stem.split("_P")[1].split("_")[0]
            except IndexError as e:
                raise ValueError(
                    f"Could not derive embryo ID from {yaml_path.name} and source file not found."
                ) from e
            tif_candidates = sorted(yaml_path.parent.glob(f"{p_id}*.ome.tif"))
            if not tif_candidates:
                raise FileNotFoundError(f"No TIF found for {p_id} in {yaml_path.parent}")
            tif_path = tif_candidates[0]
            base = p_id

        channel_names, vox, period_s = load_tif_metadata(tif_path)
        tif_file = tifffile.TiffFile(tif_path)
        series = tif_file.series[0]
        T, C, Z, Y, X = series.shape
        P = 1
        out_dir = tif_path.parent / f"aligned_{base}"

        def _read_tif_frame(t, h, w):
            return np.stack([series.pages[t * C * Z + i].asarray() for i in range(C * Z)]).reshape(C, Z, h, w).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _progress(current, total, phase, detail=None):
        if progress_cb is not None:
            progress_cb(current, total, phase, detail)

    try:
        if not no_enlarge_canvas:
            shifts = np.zeros((P, T, 2), dtype=np.float32)
            precompute_total = max(1, P * T)
            precompute_idx = 0
            for p in range(P):
                for t in range(T):
                    if use_nd2:
                        frame = data[t, p].transpose(1, 0, 2, 3)
                    else:
                        frame = _read_tif_frame(t, Y, X)
                    params = get_threshold_params_for_timepoint(cfg, t)
                    sigma = params["sigma"]
                    percentile = params["percentile"]
                    ch_idx = params["channel_index"]
                    method = params.get("method", "percentile")
                    block_size = params.get("block_size", 64)
                    invert = params.get("invert", False)
                    if use_gpu:
                        dy, dx = compute_shift_xy_gpu(frame, sigma, percentile, ch_idx, method=method, block_size=block_size, invert=invert)
                    else:
                        dy, dx = compute_shift_xy(frame, sigma, percentile, ch_idx, method=method, block_size=block_size,
                              invert=invert, levels=levels, selected_levels=selected_levels)
                    shifts[p, t] = [dy, dx]
                    precompute_idx += 1
                    _progress(precompute_idx, precompute_total, "xy_pass1_precompute")

            dy_all = shifts[:, :, 0]
            dx_all = shifts[:, :, 1]
            pad_top = int(np.ceil(max(0, -dy_all.min())))
            pad_bottom = int(np.ceil(max(0, dy_all.max())))
            pad_left = int(np.ceil(max(0, -dx_all.min())))
            pad_right = int(np.ceil(max(0, dx_all.max())))
            y_orig, x_orig = Y, X

            if use_nd2:
                data = np.pad(
                    data,
                    ((0, 0), (0, 0), (0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                )
                T, P, Z, C, Y, X = data.shape
            else:
                Y = y_orig + pad_top + pad_bottom
                X = x_orig + pad_left + pad_right

            for p in range(P):
                fpath = out_dir / (f"{base}_P{p}.ome.tif" if use_nd2 else f"{base}_xy.ome.tif")
                align_total = max(1, T)
                if low_memory:
                    def _generate():
                        for t in range(T):
                            dy, dx = shifts[p, t]
                            if use_nd2:
                                frame = data[t, p].transpose(1, 0, 2, 3)
                            else:
                                raw = _read_tif_frame(t, y_orig, x_orig)
                                frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                            if use_gpu:
                                yield apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                            else:
                                yield shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                            _progress(t + 1, align_total, "xy_pass2_align", f"position={p}")
                    save_ome_tiff(fpath, _generate(), channel_names, vox, period_s, shape=(T, C, Z, Y, X), dtype=np.uint16)
                else:
                    volume = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
                    for t in range(T):
                        dy, dx = shifts[p, t]
                        if use_nd2:
                            frame = data[t, p].transpose(1, 0, 2, 3)
                        else:
                            raw = _read_tif_frame(t, y_orig, x_orig)
                            frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                        if use_gpu:
                            volume[t] = apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                        else:
                            volume[t] = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                        _progress(t + 1, align_total, "xy_pass2_align", f"position={p}")
                    save_ome_tiff(fpath, volume, channel_names, vox, period_s)
        else:
            for p in range(P):
                fpath = out_dir / (f"{base}_P{p}.ome.tif" if use_nd2 else f"{base}_xy.ome.tif")
                align_total = max(1, T)
                if low_memory:
                    def _generate():
                        for t in range(T):
                            frame = data[t, p].transpose(1, 0, 2, 3) if use_nd2 else _read_tif_frame(t, Y, X)
                            params = get_threshold_params_for_timepoint(cfg, t)
                            sigma = params["sigma"]
                            percentile = params["percentile"]
                            ch_idx = params["channel_index"]
                            method = params.get("method", "percentile")
                            block_size = params.get("block_size", 64)
                            invert = params.get("invert", False)
                            if use_gpu:
                                shifted, _, _ = align_frame_xy_gpu(frame, sigma, percentile, ch_idx, method=method, block_size=block_size, invert=invert)
                                yield shifted.cpu().numpy().astype(np.uint16)
                            else:
                                shifted, _, _ = align_frame_xy(frame, sigma, percentile, ch_idx, method=method, block_size=block_size, invert=invert)
                                yield shifted.clip(0, 65535).astype(np.uint16)
                            _progress(t + 1, align_total, "xy_align", f"position={p}")
                    save_ome_tiff(fpath, _generate(), channel_names, vox, period_s, shape=(T, C, Z, Y, X), dtype=np.uint16)
                else:
                    volume = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
                    for t in range(T):
                        frame = data[t, p].transpose(1, 0, 2, 3) if use_nd2 else _read_tif_frame(t, Y, X)
                        params = get_threshold_params_for_timepoint(cfg, t)
                        sigma = params["sigma"]
                        percentile = params["percentile"]
                        ch_idx = params["channel_index"]
                        method = params.get("method", "percentile")
                        block_size = params.get("block_size", 64)
                        invert = params.get("invert", False)
                        if use_gpu:
                            shifted, _, _ = align_frame_xy_gpu(frame, sigma, percentile, ch_idx, method=method, block_size=block_size, invert=invert)
                            volume[t] = shifted.cpu().numpy().astype(np.uint16)
                        else:
                            shifted, _, _ = align_frame_xy(frame, sigma, percentile, ch_idx, method=method, block_size=block_size, invert=invert)
                            volume[t] = shifted.clip(0, 65535).astype(np.uint16)
                        _progress(t + 1, align_total, "xy_align", f"position={p}")
                    save_ome_tiff(fpath, volume, channel_names, vox, period_s)

                for c in range(C):
                    ch_name = channel_names[c]
                    if use_nd2:
                        mp4_aligned = out_dir / f"{base}_P{p}_{ch_name}_aligned.mp4"
                        mp4_unaligned = out_dir / f"{base}_P{p}_{ch_name}_unaligned.mp4"
                        src_tif = None
                    else:
                        mp4_aligned = out_dir / f"{base}_xy_{ch_name}_aligned.mp4"
                        mp4_unaligned = out_dir / f"{base}_{ch_name}_unaligned.mp4"
                        src_tif = tif_path
                    tiff_to_mp4(fpath, mp4_aligned, channel_index=c, fps=fps)
                    _progress(c + 1, C, "xy_mp4_aligned", f"position={p}")
                    if use_nd2:
                        continue
                    tiff_to_mp4(src_tif, mp4_unaligned, channel_index=c, fps=fps)
                    _progress(c + 1, C, "xy_mp4_unaligned", f"position={p}")
    finally:
        if tif_file is not None:
            tif_file.close()


def resolve_xy_tif_from_yaml(yaml_file):
    """Resolve XY alignment input TIF and config using centroid_align_xy.py rules."""
    yaml_path = Path(yaml_file)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    src_file = Path(cfg["source"]["file"])
    if src_file.is_file() and src_file.suffix.lower() in [".tif", ".tiff"]:
        tif_path = src_file
    else:
        try:
            p_id = "P" + yaml_path.stem.split("_P")[1].split("_")[0]
        except IndexError as e:
            raise ValueError(
                f"Could not derive embryo ID from {yaml_path.name} and source file not found."
            ) from e
        tif_candidates = sorted(yaml_path.parent.glob(f"{p_id}*.ome.tif"))
        if not tif_candidates:
            raise FileNotFoundError(f"No TIF found for {p_id} in {yaml_path.parent}")
        tif_path = tif_candidates[0]
    return tif_path, cfg


def apply_manual_xy_from_yaml(
    yaml_file,
    manual_rows,
    use_gpu=False,
    low_memory=False,
    no_enlarge_canvas=False,
    fps=2.0,
    progress_cb=None,
):
    """Apply user-entered manual XY shifts and write a new aligned movie."""
    tif_path, cfg = resolve_xy_tif_from_yaml(yaml_file)
    channel_names, vox, period_s = load_tif_metadata(tif_path)
    tif_file = tifffile.TiffFile(tif_path)
    series = tif_file.series[0]
    T, C, Z, Y, X = series.shape

    def _read_tif_frame(t, h, w):
        return np.stack([series.pages[t * C * Z + i].asarray() for i in range(C * Z)]).reshape(C, Z, h, w).astype(np.float32)

    shifts = np.zeros((T, 2), dtype=np.float32)
    for row in manual_rows:
        t1 = int(row["t1"])
        t2 = int(row["t2"])
        y1, x1 = row["p1"]
        y2, x2 = row["p2"]
        if t1 < 0 or t1 >= T or t2 < 0 or t2 >= T:
            raise ValueError(f"Manual row has out-of-range timepoint: t1={t1}, t2={t2}, T={T}")
        shifts[t2] = [float(y1 - y2), float(x1 - x2)]  # align t2 point back to t1 point

    stem = tif_path.stem[:-4] if tif_path.stem.endswith(".ome") else tif_path.stem
    out_dir = tif_path.parent / f"aligned_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{stem}_xy_manual.ome.tif"

    def _progress(current, total, phase, detail=None):
        if progress_cb is not None:
            progress_cb(current, total, phase, detail)

    try:
        y_orig, x_orig = Y, X
        if not no_enlarge_canvas:
            dy_all = shifts[:, 0]
            dx_all = shifts[:, 1]
            pad_top = int(np.ceil(max(0, -dy_all.min())))
            pad_bottom = int(np.ceil(max(0, dy_all.max())))
            pad_left = int(np.ceil(max(0, -dx_all.min())))
            pad_right = int(np.ceil(max(0, dx_all.max())))
            Y = y_orig + pad_top + pad_bottom
            X = x_orig + pad_left + pad_right
        else:
            pad_top = pad_bottom = pad_left = pad_right = 0

        total = max(1, T)
        if low_memory:
            def _generate():
                for t in range(T):
                    dy, dx = shifts[t]
                    raw = _read_tif_frame(t, y_orig, x_orig)
                    frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                    if use_gpu:
                        aligned = apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                    else:
                        aligned = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                    _progress(t + 1, total, "manual_xy_apply")
                    yield aligned

            save_ome_tiff(fpath, _generate(), channel_names, vox, period_s, shape=(T, C, Z, Y, X), dtype=np.uint16)
        else:
            volume = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
            for t in range(T):
                dy, dx = shifts[t]
                raw = _read_tif_frame(t, y_orig, x_orig)
                frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                if use_gpu:
                    volume[t] = apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                else:
                    volume[t] = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                _progress(t + 1, total, "manual_xy_apply")
            save_ome_tiff(fpath, volume, channel_names, vox, period_s)

        for c in range(C):
            ch_name = channel_names[c]
            mp4_aligned = out_dir / f"{stem}_xy_manual_{ch_name}_aligned.mp4"
            tiff_to_mp4(fpath, mp4_aligned, channel_index=c, fps=fps)
            _progress(c + 1, C, "manual_xy_mp4")
        return fpath
    finally:
        tif_file.close()


def apply_manual_xy_to_tif(
    tif_path,
    manual_rows,
    use_gpu=False,
    low_memory=False,
    no_enlarge_canvas=False,
    fps=2.0,
    progress_cb=None,
):
    """Apply user-entered manual XY shifts directly from a chosen TIF file."""
    tif_path = Path(tif_path)
    if not tif_path.is_file():
        raise FileNotFoundError(f"TIF not found: {tif_path}")

    channel_names, vox, period_s = load_tif_metadata(tif_path)
    tif_file = tifffile.TiffFile(tif_path)
    series = tif_file.series[0]
    T, C, Z, Y, X = series.shape

    def _read_tif_frame(t, h, w):
        return np.stack([series.pages[t * C * Z + i].asarray() for i in range(C * Z)]).reshape(C, Z, h, w).astype(np.float32)

    shifts = np.zeros((T, 2), dtype=np.float32)
    for row in manual_rows:
        t1 = int(row["t1"])
        t2 = int(row["t2"])
        y1, x1 = row["p1"]
        y2, x2 = row["p2"]
        if t1 < 0 or t1 >= T or t2 < 0 or t2 >= T:
            raise ValueError(f"Manual row has out-of-range timepoint: t1={t1}, t2={t2}, T={T}")
        shifts[t2] = [float(y1 - y2), float(x1 - x2)]

    stem = tif_path.stem[:-4] if tif_path.stem.endswith(".ome") else tif_path.stem
    out_dir = tif_path.parent / f"aligned_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{stem}_xy_manual.ome.tif"

    def _progress(current, total, phase, detail=None):
        if progress_cb is not None:
            progress_cb(current, total, phase, detail)

    try:
        y_orig, x_orig = Y, X
        if not no_enlarge_canvas:
            dy_all = shifts[:, 0]
            dx_all = shifts[:, 1]
            pad_top = int(np.ceil(max(0, -dy_all.min())))
            pad_bottom = int(np.ceil(max(0, dy_all.max())))
            pad_left = int(np.ceil(max(0, -dx_all.min())))
            pad_right = int(np.ceil(max(0, dx_all.max())))
            Y = y_orig + pad_top + pad_bottom
            X = x_orig + pad_left + pad_right
        else:
            pad_top = pad_bottom = pad_left = pad_right = 0

        total = max(1, T)
        if low_memory:
            def _generate():
                for t in range(T):
                    dy, dx = shifts[t]
                    raw = _read_tif_frame(t, y_orig, x_orig)
                    frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                    if use_gpu:
                        aligned = apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                    else:
                        aligned = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                    _progress(t + 1, total, "manual_xy_apply")
                    yield aligned

            save_ome_tiff(fpath, _generate(), channel_names, vox, period_s, shape=(T, C, Z, Y, X), dtype=np.uint16)
        else:
            volume = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
            for t in range(T):
                dy, dx = shifts[t]
                raw = _read_tif_frame(t, y_orig, x_orig)
                frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                if use_gpu:
                    volume[t] = apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                else:
                    volume[t] = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                _progress(t + 1, total, "manual_xy_apply")
            save_ome_tiff(fpath, volume, channel_names, vox, period_s)

        for c in range(C):
            ch_name = channel_names[c]
            mp4_aligned = out_dir / f"{stem}_xy_manual_{ch_name}_aligned.mp4"
            tiff_to_mp4(fpath, mp4_aligned, channel_index=c, fps=fps)
            _progress(c + 1, C, "manual_xy_mp4")
        return fpath
    finally:
        tif_file.close()


def launch_threshold_gui():
    """Launch the legacy threshold tuning GUI."""
    run_script_command(Path(__file__).with_name("find_threshold.py"))


def _clip_roi_zyx(volume_zyx, roi):
    """Crop a (Z, Y, X) volume to ROI bounds (y0, y1, x0, x1)."""
    if roi is None:
        return volume_zyx
    y0, y1, x0, x1 = [int(v) for v in roi]
    _, y_max, x_max = volume_zyx.shape
    y0 = max(0, min(y0, y_max - 1))
    y1 = max(y0 + 1, min(y1, y_max))
    x0 = max(0, min(x0, x_max - 1))
    x1 = max(x0 + 1, min(x1, x_max))
    return volume_zyx[:, y0:y1, x0:x1]


def _project_for_z_alignment(volume_zyx, projection_mode):
    """Project a (Z, Y, X) volume to the chosen representation for registration."""
    if projection_mode == "yz":
        return volume_zyx.max(axis=2)  # (Z, Y)
    if projection_mode == "xz":
        return volume_zyx.max(axis=1)  # (Z, X)
    if projection_mode == "z_profile_1d":
        return volume_zyx.max(axis=(1, 2))  # (Z,)
    raise ValueError(f"Unknown projection_mode: {projection_mode}")


def _estimate_shift_from_projection(reference_proj, moving_proj, projection_mode, upsample_factor=10):
    """Estimate shifts from projections and map them to (dz, dy, dx)."""
    from skimage.registration import phase_cross_correlation

    shift_vec, _, _ = phase_cross_correlation(reference_proj, moving_proj, upsample_factor=upsample_factor)
    if projection_mode == "yz":
        dz, dy = shift_vec
        return float(-dz), float(-dy), 0.0
    if projection_mode == "xz":
        dz, dx = shift_vec
        return float(-dz), 0.0, float(-dx)
    if projection_mode == "z_profile_1d":
        dz = shift_vec[0]
        return float(-dz), 0.0, 0.0
    raise ValueError(f"Unknown projection_mode: {projection_mode}")


def _update_roi_bounds(volume_zyx, current_roi):
    """Recenter ROI around the brightest point in max-projected YX."""
    if current_roi is None:
        return None
    y0, y1, x0, x1 = [int(v) for v in current_roi]
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    yx = volume_zyx.max(axis=0)
    cy, cx = np.unravel_index(np.argmax(yx), yx.shape)
    new_y0 = int(round(cy - h / 2))
    new_x0 = int(round(cx - w / 2))
    return (new_y0, new_y0 + h, new_x0, new_x0 + w)


def compute_z_shifts_for_tif(
    tif_path,
    ch_idx=0,
    projection_mode="yz",
    use_roi=False,
    roi=None,
    static_roi=True,
    roi_update_interval=10,
    mode="frame_to_frame",
    segments=None,
    upsample_factor=10,
    progress_cb=None,
):
    """Compute per-timepoint (dz, dy, dx) shifts from an OME-TIFF.

    Parameters mirror align_z.py behavior, with added projection and ROI controls.
    """
    reader = LazyTifReader(tif_path)
    try:
        T = reader.T
        shifts = [[0.0, 0.0, 0.0] for _ in range(T)]
        segments = segments or []
        current_roi = roi

        def _proj_at_t(t, ref_t_for_roi=None):
            nonlocal current_roi
            vol = reader.read_frame(t)[ch_idx]
            if use_roi and current_roi is not None:
                if not static_roi and ref_t_for_roi is not None and roi_update_interval > 0:
                    if (t - ref_t_for_roi) % roi_update_interval == 0:
                        current_roi = _update_roi_bounds(vol, current_roi)
                vol = _clip_roi_zyx(vol, current_roi)
            return _project_for_z_alignment(vol, projection_mode)

        def _progress(current, total, phase, detail=None):
            if progress_cb is not None:
                progress_cb(current, total, phase, detail)

        if mode == "frame_to_frame":
            total_steps = max(1, T - 1)
            for t in range(1, T):
                ref_proj = _proj_at_t(t - 1, ref_t_for_roi=0)
                mov_proj = _proj_at_t(t, ref_t_for_roi=0)
                shifts[t] = list(
                    _estimate_shift_from_projection(
                        ref_proj,
                        mov_proj,
                        projection_mode,
                        upsample_factor=upsample_factor,
                    )
                )
                _progress(t, total_steps, "z_compute")
        elif mode == "reference":
            total_steps = 0
            for ref_t, start, end in segments:
                total_steps += max(0, end - start + 1 - (1 if start <= ref_t <= end else 0))
            total_steps = max(1, total_steps)
            step = 0
            for ref_t, start, end in segments:
                ref_proj = _proj_at_t(ref_t, ref_t_for_roi=ref_t)
                for t in range(start, end + 1):
                    if t == ref_t:
                        shifts[t] = [0.0, 0.0, 0.0]
                        continue
                    mov_proj = _proj_at_t(t, ref_t_for_roi=ref_t)
                    shifts[t] = list(
                        _estimate_shift_from_projection(
                            ref_proj,
                            mov_proj,
                            projection_mode,
                            upsample_factor=upsample_factor,
                        )
                    )
                    step += 1
                    _progress(step, total_steps, "z_compute", f"ref={ref_t}")
        else:
            raise ValueError("mode must be 'frame_to_frame' or 'reference'")

        return shifts
    finally:
        reader.close()


def apply_z_shifts_to_tif(tif_path, dz_values, out_path=None, progress_cb=None):
    """Apply integer dz values to a TIF and write a z-aligned OME-TIFF."""
    tif_path = Path(tif_path)
    out_path = Path(out_path) if out_path else tif_path.with_name(f"{tif_path.stem}_z.ome.tif")

    reader = LazyTifReader(tif_path)
    channel_names, vox, period_s = load_tif_metadata(tif_path)
    try:
        def _progress(current, total, phase, detail=None):
            if progress_cb is not None:
                progress_cb(current, total, phase, detail)

        T, C, Z, Y, X = reader.T, reader.C, reader.Z, reader.Y, reader.X
        if len(dz_values) != T:
            raise ValueError(f"Expected {T} dz values, got {len(dz_values)}")

        offsets = []
        bottom_pad = 0
        new_Z = Z
        for t in range(T):
            dz = int(round(dz_values[t]))
            off = bottom_pad - dz
            if off < 0:
                extra = -off
                bottom_pad += extra
                new_Z += extra
                for i in range(len(offsets)):
                    offsets[i] += extra
                off = 0
            if off + Z > new_Z:
                new_Z = off + Z
            offsets.append(off)

        def _aligned_frames():
            for t in range(T):
                frame = reader.read_frame(t)
                padded = np.zeros((C, new_Z, Y, X), dtype=frame.dtype)
                padded[:, offsets[t]:offsets[t] + Z, :, :] = frame
                _progress(t + 1, max(1, T), "z_apply")
                yield padded

        save_ome_tiff(
            out_path,
            _aligned_frames(),
            channel_names,
            vox,
            period_s,
            shape=(T, C, new_Z, Y, X),
            dtype=reader.series.dtype,
        )
        return out_path
    finally:
        reader.close()


def save_z_shifts_csv_for_tif(tif_path, shifts, out_csv=None):
    """Save shifts CSV next to input tif and return CSV path."""
    tif_path = Path(tif_path)
    if out_csv is None:
        stem = tif_path.stem[:-4] if tif_path.stem.endswith(".ome") else tif_path.stem
        out_csv = tif_path.parent / f"{stem}_z_shifts.csv"
    save_shifts_csv(out_csv, shifts)
    return Path(out_csv)


def load_total_dz_from_csv(csv_path):
    """Load rounded total_dz values from a shifts CSV."""
    dz_ints = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dz_ints.append(round(float(row["total_dz"])))
    return dz_ints


# ── Display utilities ─────────────────────────────────────────────────────────

def auto_contrast(img, percentile=99.5):
    """Scale a float32 image to uint8 by clipping at a high percentile.

    A fixed 0-to-max normalisation would make dim frames appear very dark
    whenever a single hot pixel or camera artefact inflates the maximum.
    Clipping at the 99.5th percentile means at most 0.5% of pixels are
    saturated, which preserves perceptual contrast across frames with varying
    peak intensities.

    We use 99.5 rather than 100 because readout noise can create single-pixel outliers that would otherwise
    dominate the normalisation. 99.5 is a conventional value in fluorescence
    microscopy display pipelines.

    Parameters
    ----------
    img : ndarray, shape (Y, X), dtype float32
    percentile : float, default 99.5

    Returns
    -------
    ndarray, shape (Y, X), dtype uint8
    """
    vmax = np.percentile(img, percentile)

    # Guard against all-zero frames (blank channels, failed acquisition, or
    # frames that are entirely background after alignment padding).
    # Division by zero would produce NaN; setting vmax=1 gives a black frame,
    # which is the correct representation of "no signal present".
    if vmax == 0:
        vmax = 1

    # Clip to [0, 255] before casting to uint8. Without the clip, values
    # slightly above vmax would wrap around to 0 in uint8 arithmetic, creating
    # spurious black pixels at the brightest spots.
    return np.clip(img / vmax * 255, 0, 255).astype(np.uint8)


def make_grid_frame(images, nrows=2, ncols=2):
    """Tile a list of 2-D uint8 images into a single nrows×ncols grid image.

    We use a 2×2 grid because the microscope captures 4 embryo positions
    arranged in a 2×2 physical layout on the dish. Preserving this spatial
    arrangement in the output movie makes it straightforward to correlate
    features in the movie with their physical position.

    Images are placed in row-major order (left-to-right, top-to-bottom).
    If fewer than nrows*ncols images are provided (e.g. a run with only 3
    embryos), the remaining cells are left black (zero), which is unambiguous
    and avoids index-out-of-range errors.

    Parameters
    ----------
    images : list of ndarray, each shape (Y, X), dtype uint8
        All images must have the same (Y, X) dimensions.
    nrows, ncols : int, default 2
        Grid dimensions. Change these if the experiment uses a different
        number of embryo positions (e.g. a 1×4 strip or a 3×3 array).

    Returns
    -------
    grid : ndarray, shape (nrows*Y, ncols*X), dtype uint8
    """
    H, W = images[0].shape

    # Pre-allocate with zeros so unfilled cells appear black rather than
    # containing uninitialised memory values.
    grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)

    for i, img in enumerate(images):
        # divmod maps the flat index i to (row, col) in the grid,
        # placing images in row-major (reading) order.
        r, c = divmod(i, ncols)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = img

    return grid


def encode_mp4(frame_generator, output_path, fps=2):
    """Encode RGB frames using the bundled FFmpeg; raise on failure."""
    from ffmpeg_runtime import encode_rgb_mp4
    encode_rgb_mp4(frame_generator, output_path, fps)

def tiff_to_mp4(tif_path, output_path, channel_index, fps=2):
    """
    Reads a standard (T, C, Z, Y, X) OME-TIFF from disk, max-projects the 
    specified channel, and encodes it directly to an MP4 using encode_mp4.
    """
    # Read the metadata from the file to get the channel name
    channel_names, _, _ = load_tif_metadata(tif_path)
    ch_name = channel_names[channel_index]

    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0] # .series is a method of the TiffFile object that returns the first series of the file. 
        T, C, Z, Y, X = series.shape # shape is written by save_ome_tiff as (T, C, Z, Y, X). 

        def generate_frames():
            for t in tqdm(range(T), desc=f"    Encoding {ch_name}", unit="frame", leave=False): # leave=False means don't show the progress bar at the end.
                frame = np.stack([series.pages[t*C*Z + i].asarray() for i in range(C*Z)]).reshape(C, Z, Y, X).astype(np.float32)
                proj = auto_contrast(frame[channel_index].max(axis=0))
                yield np.stack([proj, proj, proj], axis=-1)

        tqdm.write(f"\nStarting encode to: {output_path.name}")
        encode_mp4(generate_frames(), output_path, fps)
        tqdm.write(f"Done! Video saved at: {output_path}")
