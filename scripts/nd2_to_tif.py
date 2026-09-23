"""Convert a folder of ND2 files into per-embryo concatenated OME-TIFFs.

Finds all .nd2 files in the given directory, converts each to per-embryo
OME-TIFFs (one per position, split along the P axis), then concatenates
the TIFFs for each embryo position across all ND2 files along the time
axis. The final output is one OME-TIFF per embryo covering the full
imaging session.

Usage:
    python nd2_to_tif.py /path/to/nd2_folder
    python nd2_to_tif.py /path/to/nd2_folder --output_dir /path/to/output
    python nd2_to_tif.py /path/to/nd2_folder --position 0
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nd2
import numpy as np

from useful_functions import concatenate_tifs, load_nd2_metadata, save_ome_tiff


def parse_bool(value):
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of ND2 files into per-embryo concatenated OME-TIFFs."
    )
    parser.add_argument(
        "nd2_folder",
        nargs="?",
        default=".",
        help="Directory containing .nd2 files to process. "
             "Defaults to the current directory.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for output TIFFs. Defaults to a subfolder "
             "named 'tifs' inside the ND2 folder.",
    )
    parser.add_argument(
        "--parallel",
        type=parse_bool,
        default=True,
        help="Process positions in parallel using threads (default: True).",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=None,
        help="Extract only this specific embryo position index (e.g., 0).",
    )
    args = parser.parse_args()

    nd2_folder = Path(args.nd2_folder)
    if not nd2_folder.is_dir():
        print(f"Error: not a directory: {nd2_folder}", file=sys.stderr)
        sys.exit(1)

    # Find all ND2 files in the folder, sorted alphabetically so that
    # sequentially named files (nd1188, nd1189, ...) are processed in
    # acquisition order.
    nd2_files = sorted(nd2_folder.glob("*.nd2"))
    if not nd2_files:
        print(f"Error: no .nd2 files found in {nd2_folder}", file=sys.stderr)
        sys.exit(1)

    # Output directory defaults to a 'tifs' subfolder inside the ND2
    # folder, keeping converted files next to the raw data.
    output_dir = Path(args.output_dir) if args.output_dir else nd2_folder / "tifs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(nd2_files)} ND2 file(s) in {nd2_folder}")
    print(f"Output directory: {output_dir}")

    # Process one ND2 at a time: load it, split along the P (position)
    # axis, and write one OME-TIFF per embryo. The ND2 data is released
    # after writing so only one file's data is in memory at a time.
    for nd2_path in nd2_files:
        print(f"\nOpening {nd2_path.name} ...")
        f = nd2.ND2File(nd2_path)
        channel_names, vox, period_s = load_nd2_metadata(nd2_path)
        data = f.to_dask()                          # lazy array, shape (T, P, Z, C, Y, X)
        if "P" not in f.sizes:
            # In case of ND2s where there is no P axis, insert a singleton (set with only one element) P axis so the rest
            # of the pipeline (which assumes T, P, Z, C, Y, X) works unchanged.
            data = data[:, np.newaxis]
        T, P, Z, C, Y, X = data.shape
        print(f"  Shape: T={T}, P={P}, Z={Z}, C={C}, Y={Y}, X={X}")
        target_positions = [args.position] if args.position is not None else range(P)
        base = nd2_path.stem  # e.g. "nd1188"

        def convert_position(p):
            """Load one position from the dask array and write it as an OME-TIFF."""
            # Extract one embryo position and reorder from the ND2
            # axis convention (T, Z, C, Y, X) to the OME-TIFF convention
            # (T, C, Z, Y, X) that save_ome_tiff expects.
            volume = data[:, p].compute().transpose(0, 2, 1, 3, 4)  # load one position into memory
            out_path = output_dir / f"{base}_P{p}.ome.tif"
            print(f"  Writing {out_path.name} ...")
            save_ome_tiff(out_path, volume, channel_names, vox, period_s)

        if args.parallel and args.position is None:
            # The bottleneck per position is disk I/O: reading chunks from
            # the ND2 file via dask and writing the output TIF. A thread
            # pool lets one position's read overlap with another's write,
            # keeping the disk busy instead of idling between positions.
            with ThreadPoolExecutor(max_workers=2) as pool:
                # Consume the results so worker errors reach the caller.
                list(pool.map(convert_position, target_positions))
        else:
            for p in target_positions:
                convert_position(p)

        f.close()         # release the ND2 file handle

    # Group the per-embryo TIFFs by position and concatenate along T.
    # All *_P{p}.ome.tif files from different ND2s belong to the same
    # embryo and are joined into one continuous timelapse.
    print(f"\nConcatenating per-embryo TIFFs ...")
    for p in target_positions:
        tifs_for_position = sorted(output_dir.glob(f"*_P{p}.ome.tif"))
        if len(tifs_for_position) < 2:
            print(f"  P{p}: only {len(tifs_for_position)} file(s), skipping concatenation")
            continue

        concat_path = output_dir / f"P{p}_concat.ome.tif"
        print(f"  P{p}: concatenating {len(tifs_for_position)} files -> {concat_path.name}")
        concatenate_tifs(tifs_for_position, concat_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
