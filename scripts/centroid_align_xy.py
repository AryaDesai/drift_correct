"""centroid_align_xy.py -- align embryo movies by XY centroid tracking.

Reads a per-embryo threshold YAML produced by find_threshold.py, derives the
input file from the YAML name, shifts the embryo to the frame centre at every
timepoint, and writes one OME-TIFF and one MP4 per channel.

Default input is a per-embryo concatenated OME-TIFF produced by nd2_to_tif.py
(e.g. P0_nd1181.ome.tif). The TIF path derives the embryo position ID (e.g.
P0) from the YAML filename and globs for a matching TIF in the same directory.
Pass --use_nd2 to load directly from the raw ND2 file instead.

Two-pass canvas expansion is the default: Pass 1 computes all shifts without
modifying data, Pass 2 pads the canvas asymmetrically and applies the shifts.
Pass --no_enlarge_canvas to skip canvas expansion (some edge data will be
clipped).

All image processing functions are defined in useful_functions.py. This
script contains only argument parsing, file I/O orchestration, and progress
reporting.

Usage:
    python centroid_align_xy.py nd1181_P0_Venus_threshold.yaml
    python centroid_align_xy.py nd1181_P0_Venus_threshold.yaml --use_gpu
    python centroid_align_xy.py nd1181_P0_Venus_threshold.yaml --use_nd2
    python centroid_align_xy.py nd1181_P0_Venus_threshold.yaml --no_enlarge_canvas
"""

import argparse
import sys
from pathlib import Path
import subprocess

import numpy as np
import tifffile
import yaml
from scipy.ndimage import shift
from tqdm import tqdm

from useful_functions import (
    align_frame_xy,
    align_frame_xy_gpu,
    apply_shift_xy_gpu,
    auto_contrast,
    compute_shift_xy,
    compute_shift_xy_gpu,
    load_nd2,
    load_tif_metadata,
    save_ome_tiff,
    encode_mp4,
    tiff_to_mp4,
)


def main():
    parser = argparse.ArgumentParser(description="Centroid-align embryo movies from ND2 data.")
    parser.add_argument("yaml_file", help="Threshold YAML from find_threshold.py")
    parser.add_argument("--use_nd2", action="store_true", help="Load from ND2 file instead of per-embryo TIF (default: TIF).")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second (default: 2)")
    parser.add_argument(
        "--no_enlarge_canvas",
        action="store_true",
        help="Skip the two-pass canvas expansion and use single-pass alignment instead (some edge data may be clipped).",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU acceleration for alignment.",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Stream aligned frames directly to disk via TiffWriter instead of accumulating the full volume in RAM before saving.",
    )
    args = parser.parse_args()

    # ── 1. Load YAML config ───────────────────────────────────────────────────

    yaml_path = Path(args.yaml_file)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    sigma      = cfg["parameters"]["sigma"]
    percentile = cfg["parameters"]["percentile"]
    ch_idx     = cfg["parameters"]["channel_index"]

    # ── 2. Load image data ────────────────────────────────────────────────────

    if args.use_nd2:
        nd2_path = Path(cfg["source"]["file"])
        if not nd2_path.is_file():
            print(f"Error: ND2 file not found: {nd2_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {nd2_path} ...")
        data, channel_names, vox, period_s = load_nd2(nd2_path)
        # load_nd2 returns the native dtype (usually uint16); cast to float32
        # for consistent processing and to support GPU operations (conv2d needs float).
        data = data.astype(np.float32)
        T, P, Z, C, Y, X = data.shape
        base    = nd2_path.stem
        out_dir = nd2_path.parent / f"aligned_{base}"
        print(f"  Shape: T={T}, P={P}, Z={Z}, C={C}, Y={Y}, X={X}")
    else:
        # Check if the source file in the YAML is a TIF and exists.
        # This is the most reliable method if files haven't been moved.
        src_file = Path(cfg["source"]["file"])
        if src_file.is_file() and src_file.suffix.lower() in [".tif", ".tiff"]:
            tif_path = src_file
            print(f"Using source file from YAML: {tif_path}")
            # Use the filename stem (minus .ome) as the base for output naming.
            base = tif_path.stem
            if base.endswith(".ome"):
                base = base[:-4]
        else:
            # Fallback: derive embryo ID from the YAML filename and glob for matching TIFs.
            # e.g. nd1181_P0_Venus_threshold.yaml → p_id = "P0".
            try:
                p_id = 'P' + yaml_path.stem.split('_P')[1].split('_')[0]
            except IndexError:
                print(f"Error: Could not derive embryo ID from {yaml_path.name} and source file not found.", file=sys.stderr)
                sys.exit(1)

            tif_candidates = sorted(yaml_path.parent.glob(f"{p_id}*.ome.tif"))
            if not tif_candidates:
                print(f"Error: no TIF found for {p_id} in {yaml_path.parent}", file=sys.stderr)
                sys.exit(1)
            tif_path = tif_candidates[0]
            base = p_id

        print(f"Loading {tif_path.name} ...")
        channel_names, vox, period_s = load_tif_metadata(tif_path)
        # Open the TIF for lazy per-frame reads and is kept open for both passes and MP4 writing to avoid reopening the file multiple times.
        tif_file = tifffile.TiffFile(tif_path)
        series   = tif_file.series[0]
        T, C, Z, Y, X = series.shape  # shape is written by save_ome_tiff as (T, C, Z, Y, X)
        P = 1  # single embryo per TIF
        out_dir = tif_path.parent / f"aligned_{base}"
        print(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

        def _read_tif_frame(t, h, w):
            """Read timepoint t from the TIF as a (C, Z, h, w) float32 array."""
            return np.stack([series.pages[t*C*Z + i].asarray()
                             for i in range(C*Z)]).reshape(C, Z, h, w).astype(np.float32)

    print(f"  Channels: {channel_names}")
    print(f"  Threshold channel: {channel_names[ch_idx]} (index {ch_idx})")
    print(f"  sigma={sigma}, percentile={percentile}")

    # ── 3. Prepare output directory ───────────────────────────────────────────

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}")


    # ── 4a. Two-pass path: default; pass --no_enlarge_canvas to skip ─────────────────────

    if not args.no_enlarge_canvas:
        if args.use_gpu:
            import torch
        # Pass 1 — precompute all (dy, dx) shifts without modifying data.
        # Printing each value as it arrives lets the user spot directional
        # drift patterns before committing to any canvas expansion.
        print(f"\n--- Pass 1: precomputing shifts ---")
        shifts = np.zeros((P, T, 2))  # shifts[p, t] = [dy, dx]

        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}:")
            for t in range(T):
                if args.use_nd2:
                    # ND2 data is stored as (Z, C, Y, X) per timepoint; transpose to (C, Z, Y, X)
                    # so channel is always axis 0, which is what all downstream functions expect.
                    frame = data[t, p].transpose(1, 0, 2, 3)
                else:
                    # TIF pages are stored as individual 2-D (Y, X) planes in T × C × Z order.
                    frame = _read_tif_frame(t, Y, X)
                if args.use_gpu:
                    dy, dx = compute_shift_xy_gpu(frame, sigma, percentile, ch_idx)
                else:
                    dy, dx = compute_shift_xy(frame, sigma, percentile, ch_idx)
                shifts[p, t] = [dy, dx]

        dy_all, dx_all = shifts[:, :, 0], shifts[:, :, 1]
        print(f"\nShift summary (pixels):")
        print(f"  dy  min={dy_all.min():+.1f}  max={dy_all.max():+.1f}  mean={dy_all.mean():+.1f}")
        print(f"  dx  min={dx_all.min():+.1f}  max={dx_all.max():+.1f}  mean={dx_all.mean():+.1f}")

        # Compute asymmetric padding so that no original data is clipped.
        # scipy shift with +dy moves content downward → bottom rows are lost → pad bottom.
        # scipy shift with -dy moves content upward  → top rows are lost    → pad top.
        # Same logic applies to dx / left / right.
        pad_top    = int(np.ceil(max(0, -dy_all.min())))
        pad_bottom = int(np.ceil(max(0,  dy_all.max())))
        pad_left   = int(np.ceil(max(0, -dx_all.min())))
        pad_right  = int(np.ceil(max(0,  dx_all.max())))
        print(f"\nCanvas padding:  top={pad_top}  bottom={pad_bottom}  left={pad_left}  right={pad_right}")

        # Save the original frame dimensions before expansion so Pass 2 can
        # read each TIF frame at its original size and pad it on-the-fly.
        Y_orig, X_orig = Y, X

        # Pad only Y and X; all other axes are unchanged.
        # The precomputed shifts remain valid on the larger canvas — each
        # centroid will land at (pad_top + Y_orig/2, pad_left + X_orig/2),
        # i.e. stabilised but not re-centred in the new frame. This is
        # intentional: the embryo position is consistent across timepoints,
        # which is the goal of alignment.
        if args.use_nd2:
            # ND2: expand the full in-memory array once; axes 4 and 5 are Y and X.
            data = np.pad(
                data,
                ((0, 0), (0, 0), (0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            )
            T, P, Z, C, Y, X = data.shape
        else:
            # TIF: no in-memory array to pad; update Y and X so the output
            # volume is allocated at the correct expanded size, and each frame
            # is padded individually when it is read in Pass 2 below.
            Y = Y_orig + pad_top + pad_bottom
            X = X_orig + pad_left + pad_right
        print(f"  Expanded canvas: Y={Y}  X={X}")

        # Pass 2 — apply precomputed shifts on the expanded canvas.
        # We call scipy.ndimage.shift directly here rather than align_frame_xy
        # because the shifts are already computed and we must not recompute
        # centroids on the padded canvas (the padding zeros would shift the
        # percentile threshold and produce incorrect centroids).
        print(f"\n--- Pass 2: aligning {P} embryo(s) × {T} timepoints ---")
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}")
            if args.use_nd2:
                fpath = out_dir / f"{base}_P{p}.ome.tif"
            else:
                # Strip .ome from stem if present so we don't get .ome_xy.ome.tif
                stem = tif_path.stem
                if stem.endswith(".ome"):
                    stem = stem[:-4]
                fpath = out_dir / f"{stem}_xy.ome.tif"
            if args.low_memory:
                # Pass a generator to imwrite so frames are consumed one at a time
                # without accumulating the full volume in RAM. shape and dtype are
                # declared upfront so tifffile writes correct OME-XML before consuming
                # any frames.
                print(f"    Saving {fpath.name} (streaming) ...")
                def _generate_aligned_frames():
                    for t in tqdm(range(T), desc="    Aligning", unit="frame", leave=True):
                        dy, dx = shifts[p, t]
                        if args.use_nd2:
                            # ND2 array was already padded above; axes are (Z, C, Y, X), transpose to (C, Z, Y, X).
                            frame = data[t, p].transpose(1, 0, 2, 3)
                        else:
                            raw = _read_tif_frame(t, Y_orig, X_orig)
                            frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                        if args.use_gpu:
                            yield apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                        else:
                            yield shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)
                save_ome_tiff(fpath, _generate_aligned_frames(), channel_names, vox, period_s,
                             shape=(T, C, Z, Y, X), dtype=np.uint16)
                print(f"    Saved {fpath.name}")
            else:
                # Always allocate volume in numpy. GPU path processes each frame on
                # GPU and moves the result back to CPU immediately this avoids
                # allocating the full volume on GPU (which would OOM on large datasets).
                volume = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
                for t in tqdm(range(T), desc="    Aligning", unit="frame", leave=True):
                    dy, dx = shifts[p, t]
                    if args.use_nd2:
                        # ND2 array was already padded above; axes are (Z, C, Y, X), transpose to (C, Z, Y, X).
                        frame = data[t, p].transpose(1, 0, 2, 3)
                    else:
                        raw = _read_tif_frame(t, Y_orig, X_orig)
                        frame = np.pad(raw, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
                    if args.use_gpu:
                        volume[t] = apply_shift_xy_gpu(frame, dy, dx).cpu().numpy().clip(0, 65535).astype(np.uint16)
                    else:
                        volume[t] = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0).clip(0, 65535).astype(np.uint16)

                print(f"    Saving {fpath.name} ...")
                save_ome_tiff(fpath, volume, channel_names, vox, period_s)
                print(f"    Saved {fpath.name}")
            if not args.use_nd2:
                print(f"Closing the TIF file ...")
                tif_file.close()
                print(f"Done closing the TIF file.")
            for c in range(C):
                # For each channel, encode both the unaligned and aligned tifs to an MP4.
                ch_name = channel_names[c]
                if args.use_nd2:
                    mp4_path = out_dir / f"{base}_P{p}_{ch_name}_aligned.mp4"
                    mp4_path_unaligned = out_dir / f"{base}_P{p}_{ch_name}_unaligned.mp4"
                else:
                    # Strip .ome from stem
                    stem = tif_path.stem
                    if stem.endswith(".ome"):
                        stem = stem[:-4]
                    mp4_path = out_dir / f"{stem}_xy_{ch_name}_aligned.mp4"
                    mp4_path_unaligned = out_dir / f"{stem}_{ch_name}_unaligned.mp4"
                # fpath is the path to the OME-TIFF we just wrote to disk a few lines above.
                tiff_to_mp4(fpath, mp4_path, channel_index=c, fps=args.fps)
                print(f"    Saved {mp4_path.name}")
                print(f" Now encoding the unaligned TIF to an MP4 ...")
                mp4_path_unaligned = out_dir / f"{base}_P{p}_{ch_name}_unaligned.mp4"
                # The unaligned tiff is the one we stared with. That file is located at tif_path.
                tiff_to_mp4(tif_path, mp4_path_unaligned, channel_index=c, fps=args.fps)
                print(f"    Saved {mp4_path_unaligned.name}")
    # ── 4b. Single-pass path: --no_enlarge_canvas ────────────────────────────────────────

    else:
        print(f"\nAligning {P} embryo(s) × {T} timepoints ...")
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}")
            if args.use_nd2:
                fpath = out_dir / f"{base}_P{p}.ome.tif"
            else:
                # Strip .ome from stem if present so we don't get .ome_xy.ome.tif
                stem = tif_path.stem
                if stem.endswith(".ome"):
                    stem = stem[:-4]
                fpath = out_dir / f"{stem}_xy.ome.tif"
            if args.low_memory:
                # Pass a generator to imwrite so frames are consumed one at a time
                # without accumulating the full volume in RAM. shape and dtype are
                # declared upfront so tifffile writes correct OME-XML before consuming
                # any frames.
                print(f"    Saving {fpath.name} (streaming) ...")
                def _generate_aligned_frames():
                    for t in tqdm(range(T), desc="    Aligning timepoints", unit="frame", leave=True):
                        if args.use_nd2:
                            # Reorder from ND2 axis order (Z, C, Y, X) to pipeline order (C, Z, Y, X).
                            frame = data[t, p].transpose(1, 0, 2, 3)
                        else:
                            frame = _read_tif_frame(t, Y, X)
                        if args.use_gpu:
                            shifted, dy, dx = align_frame_xy_gpu(frame, sigma, percentile, ch_idx)
                            yield shifted.cpu().numpy().astype(np.uint16)
                        else:
                            shifted, dy, dx = align_frame_xy(frame, sigma, percentile, ch_idx)
                            yield shifted.clip(0, 65535).astype(np.uint16)
                save_ome_tiff(fpath, _generate_aligned_frames(), channel_names, vox, period_s,
                             shape=(T, C, Z, Y, X), dtype=np.uint16)
                print(f"    Saved {fpath.name}")
            else:
                # Always allocate volume in numpy. GPU path processes each frame on
                # GPU and moves the result back to CPU immediately.
                volume = np.zeros((T, C, Z, Y, X), dtype=np.uint16)
                for t in tqdm(range(T), desc="    Aligning timepoints", unit="frame", leave=True):
                    if args.use_nd2:
                        # Reorder from ND2 axis order (Z, C, Y, X) to pipeline order (C, Z, Y, X).
                        frame = data[t, p].transpose(1, 0, 2, 3)
                    else:
                        frame = _read_tif_frame(t, Y, X)
                    if args.use_gpu:
                        shifted, dy, dx = align_frame_xy_gpu(frame, sigma, percentile, ch_idx)
                        volume[t] = shifted.cpu().numpy().astype(np.uint16)  # We convert the float32 to uint16 to save disk space and time.

                    else:
                        shifted, dy, dx = align_frame_xy(frame, sigma, percentile, ch_idx)
                        volume[t] = shifted.clip(0, 65535).astype(np.uint16)  # We convert the float32 to uint16 to save disk space and time.

                print(f"    Saving {fpath.name} ...")
                save_ome_tiff(fpath, volume, channel_names, vox, period_s)
                print(f"    Saved {fpath.name}")
            if not args.use_nd2:
                print(f"Closing the TIF file ...")
                tif_file.close()
                print(f"Done closing the TIF file.")
            # Write one MP4 per channel: unaligned (left) and aligned (right).
            for c in range(C):
                ch_name = channel_names[c]
                if args.use_nd2:
                    mp4_path = out_dir / f"{base}_P{p}_{ch_name}_aligned.mp4"
                    mp4_path_unaligned = out_dir / f"{base}_P{p}_{ch_name}_unaligned.mp4"
                else:
                    # Strip .ome from stem
                    stem = tif_path.stem
                    if stem.endswith(".ome"):
                        stem = stem[:-4]
                    mp4_path = out_dir / f"{stem}_xy_{ch_name}_aligned.mp4"
                    mp4_path_unaligned = out_dir / f"{stem}_{ch_name}_unaligned.mp4"
                # fpath is the path to the OME-TIFF we just wrote to disk a few lines above.
                print(f" Encoding the aligned TIF to an MP4 ...")
                tiff_to_mp4(fpath, mp4_path, channel_index=c, fps=args.fps)
                print(f" Done encoding the aligned TIF to an MP4.")
                print(f"    Saved {mp4_path.name}")
                print(f" Now encoding the unaligned TIF to an MP4 ...")

                
                # The unaligned tiff is the one we stared with. That file is located at tif_path.
                tiff_to_mp4(tif_path, mp4_path_unaligned, channel_index=c, fps=args.fps)
                print(f" Done encoding the unaligned TIF to an MP4.")
                print(f"    Saved {mp4_path_unaligned.name}")




if __name__ == "__main__":
    main()
