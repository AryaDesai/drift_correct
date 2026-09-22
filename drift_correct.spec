"""One-folder build: a .app bundle on macOS, a plain folder elsewhere.

Run with python -m PyInstaller drift_correct.spec. PyInstaller ad-hoc signs
the result, which runs on the machine that built it but is rejected by
Gatekeeper elsewhere.
"""

from pathlib import Path
import sys

import imageio_ffmpeg

root = Path(SPECPATH)

# Vendor the FFmpeg that imageio_ffmpeg installed for this platform. It links
# only against system frameworks, so it relocates into the bundle unchanged,
# and get_ffmpeg_exe picks the right binary and suffix per platform.
ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
if not ffmpeg_exe.is_file():
    raise RuntimeError(f"imageio_ffmpeg has no binary at {ffmpeg_exe}")
# The vendored build includes libx265, so its GPL terms travel with the app.
site_packages = Path(imageio_ffmpeg.__file__).resolve().parent.parent
licenses = sorted(site_packages.glob("imageio_ffmpeg-*.dist-info/LICENSE"))
if not licenses:
    raise RuntimeError(f"No imageio_ffmpeg LICENSE found under {site_packages}")

a = Analysis(
    [str(root / "drift_correct.py")],
    pathex=[str(root)],
    binaries=[(str(ffmpeg_exe), "ffmpeg_vendor/imageio_ffmpeg/binaries")],
    datas=[(str(licenses[0]), "licenses/imageio-ffmpeg")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    runtime_hooks=[],
    excludes=["PySide6", "PySide2", "PyQt5", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
# One folder rather than one file: every tool button relaunches this same
# executable, and a onefile build would re-extract the whole payload per click.
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="drift_correct",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="drift_correct",
)
# Built by make_icon.sh from a square source image; the build stays usable
# without one rather than failing on a missing icon.
icon_file = root / "drift_correct.icns"

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="drift_correct.app",
        icon=str(icon_file) if icon_file.is_file() else None,
        bundle_identifier="edu.ozbudaklab.driftcorrect",
        info_plist={
            "CFBundleName": "Drift Correct",
            "CFBundleDisplayName": "Drift Correct",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "11.0",
        },
    )
