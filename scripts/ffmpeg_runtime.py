"""Locate a bundled FFmpeg and stream RGB frames to an HEVC movie."""

import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ENCODERS = (
    ("Apple VideoToolbox", [
    "-c:v", "hevc_videotoolbox",
    "-allow_sw", "1",
    "-q:v", "65",
    "-pix_fmt", "yuv420p",
]),
    ("NVIDIA NVENC", ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p"]),
    ("Intel Quick Sync", ["-c:v", "hevc_qsv", "-preset", "medium", "-global_quality", "23", "-pix_fmt", "nv12"]),
    ("CPU", ["-c:v", "libx265", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),

)


def input_command(executable, width, height, fps):
    return [executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]


def select_encoder(executable, first, fps):
    """Probe the actual dimensions/format without consuming the frame iterator."""
    height, width, _ = first.shape
    failures = []
    for name, options in ENCODERS:
        command = input_command(executable, width, height, fps) + options + ["-frames:v", "1", "-f", "null", "-"]
        try:
            result = subprocess.run(command, input=first.tobytes(), capture_output=True,
                                    timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode == 0:
                print(f"Video encoder: {name} ({options[1]})", flush=True)
                return options
            detail = result.stderr.decode("utf-8", errors="replace").strip()[-2000:]
        except subprocess.TimeoutExpired:
            detail = "test encode timed out after 20 seconds"
        failures.append(f"{name}: {detail}")
        print(f"Video encoder {name} unavailable; trying next encoder. {detail[:300]}", flush=True)
    raise RuntimeError("No working HEVC encoder:\n" + "\n".join(failures))


# Windows keeps the .exe suffix; macOS and Linux do not.
FFMPEG_NAME = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def search_roots():
    """Directories that may hold a vendored FFmpeg, nearest bundle first."""
    roots = []
    # _MEIPASS covers a PyInstaller onefile extraction and the Contents/Frameworks
    # payload of a .app bundle, neither of which sits beside the executable.
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        roots.append(Path(bundle))
    # __file__ covers source runs and Nuitka standalone; argv[0] covers a vendor
    # folder dropped next to the launcher.
    roots.append(Path(__file__).resolve().parent)
    launcher = Path(sys.argv[0]).resolve().parent
    roots.append(launcher)
    # Inside drift_correct.app/Contents/MacOS, also look in the bundle's
    # Resources and in the folder holding the .app itself.
    if launcher.name == "MacOS" and launcher.parent.name == "Contents":
        roots.append(launcher.parent / "Resources")
        roots.append(launcher.parent.parent.parent)
    return roots


def find_ffmpeg():
    override = os.environ.get("DRIFT_CORRECT_FFMPEG")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DRIFT_CORRECT_FFMPEG does not exist: {path}")
        return str(path.resolve())
    for root in search_roots():
        # The imageio_ffmpeg binaries carry a platform and version suffix, so
        # match them by prefix rather than by exact name.
        for path in (root / FFMPEG_NAME, root / "bin" / FFMPEG_NAME,
                     *sorted((root / "ffmpeg_vendor" / "imageio_ffmpeg" / "binaries").glob("ffmpeg*"))):
            if path.is_file():
                return str(path)
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    raise FileNotFoundError(f"FFmpeg not found. Place an ffmpeg_vendor folder next to the "
                            f"drift_correct executable, or set DRIFT_CORRECT_FFMPEG to the "
                            f"full path of {FFMPEG_NAME}.")


def encode_rgb_mp4(frames, output_path, fps=2):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be a finite positive number")
    executable = find_ffmpeg()
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError("Cannot encode a movie with no frames")
    if first.ndim != 3 or first.shape[2] != 3 or str(first.dtype) != "uint8":
        raise ValueError("Expected uint8 RGB frames with shape (height, width, 3)")
    height, width, _ = first.shape
    encoder_options = select_encoder(executable, first, fps)
    output_path = Path(output_path)
    # Commit only a successfully encoded video, leaving existing output intact
    # if FFmpeg fails or frame generation raises an exception.
    handle, temporary = tempfile.mkstemp(prefix=".encoding-", suffix=".mp4", dir=output_path.parent)
    os.close(handle)
    command = input_command(executable, width, height, fps) + encoder_options + [
        "-tag:v", "hvc1", "-movflags", "+faststart", temporary]
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=errors,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                try:
                    for frame in itertools.chain((first,), frames):
                        if frame.shape != first.shape or str(frame.dtype) != "uint8":
                            raise ValueError("All video frames must have identical shape and uint8 dtype")
                        process.stdin.write(frame.tobytes())
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                code = process.wait()
                if code:
                    errors.seek(0)
                    detail = errors.read().decode("utf-8", errors="replace")[-8000:]
                    raise RuntimeError(f"FFmpeg failed (exit {code}):\n{detail}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
