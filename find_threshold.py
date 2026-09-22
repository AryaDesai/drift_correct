"""find_threshold.py — interactive threshold tuning for embryo masking.

PyQt6 desktop application for tuning Gaussian blur (sigma) and percentile
threshold parameters on microscopy data. Displays one embryo at a time with
a red mask overlay and cyan centroid crosshair so the user can visually
confirm the mask quality before saving parameters to a YAML config.

Supports two input formats:
  - ND2 files (multi-embryo timelapse): loads the full (T, P, Z, C, Y, X)
    volume via useful_functions.load_nd2, max-projects Z, and lets the user
    navigate between embryo positions with Prev/Next buttons.
  - OME-TIFF / TIFF files (single-embryo): loads a (T, C, Z, Y, X) volume
    via tifffile.imread, max-projects Z, and treats it as a single position.
    Embryo navigation is hidden in this case.dir

Each embryo's parameters are saved to its own YAML file so that different
embryos can have different sigma/percentile values. The YAML schema matches
the format expected by centroid_align_xy.py.

A live histogram beneath the image shows Gaussian-smoothed intensities,
mask membership or Otsu classes, and the thresholds used for masking.
Available methods are percentile, multi-level Otsu, and percentile-to-Otsu ROI.
Multi-level Otsu uses the JAX implementation so the preview matches
the downstream masking and alignment calculations.


Usage:
    python find_threshold.py
"""

import datetime
import sys
from pathlib import Path
from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import numpy as np
import cv2
import tifffile
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label

from useful_functions import (
    MAX_OTSU_LEVELS,
    find_largest_mask_xy,
    load_nd2,
    load_tif_metadata,
    max_project_z,
    multiotsu_thresholds_and_classes,
)


# ── Data loading ──────────────────────────────────────────────────────────────


class LazyTifAccessor:
    """Lazy accessor for TIF files that loads one timepoint at a time.

    Provides array-like indexing `[t, p, c]` to retrieve a 2-D (Y, X) image,
    but only loads and max-projects the requested timepoint from disk. This
    prevents out-of-memory errors when working with large concatenated TIFs.

    The accessor mimics the shape (T, P, C, Y, X) that the app expects, with
    P always equal to 1 for single-embryo TIFs.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.tif = tifffile.TiffFile(file_path)
        self.series = self.tif.series[0]
        # Expected shape from save_ome_tiff: (T, C, Z, Y, X)
        self.T, self.C, self.Z, self.Y, self.X = self.series.shape
        self.P = 1  # Single embryo per TIF
        self.shape = (self.T, self.P, self.C, self.Y, self.X)
        # Load channel names from OME metadata (e.g. "Venus", "mCherry").
        self.channel_names, _, _ = load_tif_metadata(file_path)
        # Cache the most recently loaded timepoint to avoid redundant reads
        # when the user switches channels without changing T.
        self._cache_t = None
        self._cache_data = None  # Shape: (C, Y, X) — max-projected

    def __getitem__(self, idx):
        """Return a 2-D (Y, X) image for the given (t, p, c) index."""
        t, p, c = idx
        if t != self._cache_t:
            # Load all C*Z pages for this timepoint and max-project Z.
            pages_per_t = self.C * self.Z
            start = t * pages_per_t
            frame_czyx = np.stack(
                [self.series.pages[start + i].asarray() for i in range(pages_per_t)]
            ).reshape(self.C, self.Z, self.Y, self.X).astype(np.float32)
            # Max-project Z → (C, Y, X)
            self._cache_data = frame_czyx.max(axis=1)
            self._cache_t = t
        return self._cache_data[c]

    def close(self):
        """Close the underlying TIF file handle."""
        self.tif.close()


def load_file(file_path):
    """Load an ND2 or TIF file and return a Z-max-projected accessor + metadata.

    Both file types are normalised to the same output shape (T, P, C, Y, X)
    so that the rest of the application does not need to distinguish between
    them. ND2 files naturally have a P (position) axis; TIF files are assumed
    to be single-embryo and get a P axis of size 1 inserted.

    Parameters
    ----------
    file_path : str
        Path to an .nd2 or .tif/.ome.tif file.

    Returns
    -------
    max_proj : ndarray or LazyTifAccessor, shape (T, P, C, Y, X)
        Z-max-projected image data. For ND2 files this is a float32 ndarray
        loaded into memory. For TIF files this is a LazyTifAccessor that
        loads one timepoint at a time to avoid out-of-memory errors.
    channel_names : list of str
        Channel names. For ND2 files these come from the microscope metadata;
        for TIF files they default to "Ch0", "Ch1", etc. because TIF metadata
        does not reliably carry channel names.
    is_nd2 : bool
        True if the file was an ND2, False if TIF. Used downstream to decide
        the YAML filename pattern and whether to show embryo navigation.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".nd2":
        data, channel_names, _, _ = load_nd2(file_path)
        # data is (T, P, Z, C, Y, X). Max-project Z to get (T, P, C, Y, X).
        max_proj = max_project_z(data)
        return max_proj, channel_names, True

    # TIF / OME-TIFF path: use lazy accessor to avoid loading entire file.
    accessor = LazyTifAccessor(file_path)

    return accessor, accessor.channel_names, False


# ── Rendering ─────────────────────────────────────────────────────────────────


# Overlay colour per intensity class, dark class first. Distinct hues rather
# than a brightness ramp so that several checked classes stay tellable apart,
# which a single-colour overlay could not do.
LEVEL_COLORS = [
    (255, 50, 50),    # red
    (50, 200, 255),   # blue
    (120, 255, 80),   # green
    (255, 220, 40),   # yellow
    (220, 100, 255),  # purple
]


def render_embryo(img, mask, centroid, class_map=None, selected_levels=None):
    """Grayscale image + red mask overlay + centroid crosshair -> RGBA PIL image.

    The red semi-transparent overlay shows which pixels the thresholding
    selected as foreground. The cyan crosshair marks the centroid of the
    largest connected component — this is the point that centroid_align_xy.py
    will use to centre the embryo.

    Parameters
    ----------
    img : ndarray, shape (Y, X), float32
        Raw (unsmoothed) max-projected image for display.
    mask : ndarray of bool, shape (Y, X)
        Binary mask from find_largest_mask_xy.
    centroid : ndarray, shape (2,)
        [cy, cx] centroid coordinates from find_largest_mask_xy.

    Returns
    -------
    composite : PIL.Image, mode RGBA
    """
    # Auto-contrast normalisation: clip at the 99.5th percentile to avoid
    # hot pixels dominating the display range. Same logic as auto_contrast
    # in useful_functions.py but producing an RGBA composite rather than
    # a bare uint8 array.
    vmax = np.percentile(img, 99.5)
    if vmax == 0:
        vmax = 1
    gray = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGBA")

    # Semi-transparent overlay on masked pixels. Alpha=100 (out of 255) lets
    # the underlying grayscale detail show through while clearly marking the
    # mask boundary.
    overlay_arr = np.zeros((*gray.shape, 4), dtype=np.uint8)
    if class_map is None:
        overlay_arr[mask] = [255, 50, 50, 100]
    else:
        # Multi-level Otsu: colour each checked class separately so the user
        # can see which class contributed which region. This shows every
        # selected pixel, whereas mask holds only the largest connected
        # component, so the two deliberately differ when a selection is
        # fragmented.
        for i in selected_levels or []:
            overlay_arr[class_map == i] = [*LEVEL_COLORS[i % len(LEVEL_COLORS)], 100]
    composite = Image.alpha_composite(base, Image.fromarray(overlay_arr, mode="RGBA"))

    # Cyan crosshair at the centroid. The arm length (s=25 pixels) and line
    # width (7 pixels) are chosen to be visible on 1024×1024 embryo images
    # without obscuring nearby anatomy.
    draw = ImageDraw.Draw(composite)
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    s = 25
    draw.line([(cx - s, cy), (cx + s, cy)], fill=(0, 255, 255, 255), width=7)
    draw.line([(cx, cy - s), (cx, cy + s)], fill=(0, 255, 255, 255), width=7)

    return composite


# ── Application ───────────────────────────────────────────────────────────────


class EmbryoLabel(QLabel):
    """Keep the image proportional when the histogram divider is dragged."""

    def __init__(self):
        super().__init__()
        self._image = QPixmap()
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setPixmap(self, pixmap):
        self._image = pixmap
        self._resize_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_image()

    def _resize_image(self):
        if not self._image.isNull():
            super().setPixmap(self._image.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


class ThresholdApp:
    """PyQt6 application for interactive threshold parameter tuning.

    The user loads an ND2 or TIF file, selects a channel, and adjusts sigma
    and percentile sliders while watching the mask overlay update in real time.
    When satisfied, they save the parameters to a YAML file that
    centroid_align_xy.py will read.

    One embryo is displayed at a time. For multi-embryo ND2 files, Prev/Next
    buttons navigate between positions. Each embryo's parameters are saved
    independently to its own YAML file so different embryos can have different
    thresholds.
    """

    def __init__(self, root):
        self.root = root
        self.root.setWindowTitle("Find threshold for masking PSM")

        # ── Data state ────────────────────────────────────────────────────
        # These are populated by _load_file and remain None until a file is
        # loaded. All rendering code checks for None before proceeding.
        self.file_path = None
        self.max_proj = None       # (T, P, C, Y, X) float32
        self.channel_names = []
        self.is_nd2 = False
        self.T = self.P = self.C = self.Y = self.X = 0
        self.custom_timepoint_params = {}

        # ── Multi-level Otsu cache ────────────────────────────────────────
        # The class map depends only on the image and the level count, not on
        # which classes are checked, so it is kept between redraws. Toggling a
        # checkbox would otherwise repeat the same JAX threshold calculation.
        self._class_map = None
        self._class_map_key = None
        self._class_thresholds = np.array([])


        # ── Animation state ───────────────────────────────────────────────
        self.playing = False
        # A single-shot timer schedules the next frame and stops on Pause.
        self.animation_timer = QTimer(self.root)
        self.animation_timer.setSingleShot(True)
        self.animation_timer.timeout.connect(self._animate)

        # ── Display scaling ───────────────────────────────────────────────
        # Each embryo image is scaled to this size in pixels for display.
        # The raw images are typically 1024×1024; displaying them at full
        # resolution would make the window too large on most screens.
        self.display_size = 512

        self._build_ui()
        # A tall window gives both square views room without exceeding the
        # available desktop area. The user can still resize it normally.
        screen = self.root.screen().availableGeometry()
        self.root.resize(min(960, int(screen.width() * 0.9)),
                         min(1200, int(screen.height() * 0.9)))

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        # Left panel: controls. The image occupies the remaining space.
        layout = QHBoxLayout(self.root)
        ctrl = QVBoxLayout()
        controls = QWidget()
        controls.setLayout(ctrl)
        controls.setMaximumWidth(340)
        layout.addWidget(controls)
        title = QLabel("Find threshold for masking PSM")
        title.setStyleSheet("font: bold 14pt Helvetica;")
        ctrl.addWidget(title)
        legend = QLabel("Red overlay = mask  |  + = centroid")
        legend.setStyleSheet("color: gray;")
        ctrl.addWidget(legend)

        # Browse opens a native dialog; the entry also accepts pasted paths.
        self._add_button(ctrl, "Browse", self._browse)
        self.path_entry = QLineEdit()
        ctrl.addWidget(self.path_entry)
        self._add_button(ctrl, "Load", self._load_file)
        ctrl.addWidget(QLabel("Channel"))
        self.channel_combo = QComboBox()
        ctrl.addWidget(self.channel_combo)
        self.channel_combo.activated.connect(lambda _: self._update())

        # Prev/Next navigate positions. Hidden for single-embryo TIFs.
        self.nav_frame = QWidget()
        nav = QHBoxLayout(self.nav_frame)
        nav.setContentsMargins(0, 0, 0, 0)
        ctrl.addWidget(self.nav_frame)
        self.embryo_idx = 0
        self._add_button(nav, "Prev", self._prev_embryo)
        self.embryo_label = QLabel("Embryo 0 / 0")
        nav.addWidget(self.embryo_label)
        self._add_button(nav, "Next", self._next_embryo)

        # T selects the timepoint; sigma sets the Gaussian blur radius.
        self.t_slider = self._add_slider(ctrl, "T", 0, 0, 0)
        self.sigma_slider = self._add_slider(ctrl, "Sigma", 1, 100, 2)
        ctrl.addWidget(QLabel("Threshold Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Percentile", "Multi-Level Otsu", "Percentile -> Otsu ROI",
        ])
        ctrl.addWidget(self.method_combo)
        self.method_combo.activated.connect(lambda _: self._on_method_change())

        # Keep the class count within the shared masking helper's range.
        self.levels_frame = QWidget()
        levels_layout = QHBoxLayout(self.levels_frame)
        levels_layout.setContentsMargins(0, 0, 0, 0)
        levels_layout.addWidget(QLabel("Levels"))
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(2, MAX_OTSU_LEVELS)
        self.levels_spin.setValue(3)
        levels_layout.addWidget(self.levels_spin)
        self.levels_spin.valueChanged.connect(lambda _: self._on_levels_change())
        ctrl.addWidget(self.levels_frame)
        self.levels_frame.hide()

        # One checkbox per class, rebuilt whenever the level count changes.
        self.levels_check_frame = QWidget()
        self.levels_check_layout = QHBoxLayout(self.levels_check_frame)
        self.levels_check_layout.setContentsMargins(0, 0, 0, 0)
        ctrl.addWidget(self.levels_check_frame)
        self.levels_check_frame.hide()
        self.level_vars = []

        # Qt sliders use integers, so percentile is stored in half percent
        # steps and converted back when reading the controls.
        self.pct_slider = self._add_slider(ctrl, "Percentile", 20, 199, 180, 0.5)
        self.invert_check = QCheckBox("Invert Mask")
        ctrl.addWidget(self.invert_check)
        self.invert_check.toggled.connect(lambda _: self._update())

        # Store controls for the current timepoint or an inclusive range.
        self._add_separator(ctrl)
        self._add_button(ctrl, "Apply to current T", self._apply_to_current_t)
        range_layout = QHBoxLayout()
        ctrl.addLayout(range_layout)
        range_layout.addWidget(QLabel("Range"))
        self.range_start_spin = QSpinBox()
        self.range_end_spin = QSpinBox()
        for spin in (self.range_start_spin, self.range_end_spin):
            spin.setRange(0, 0)
            range_layout.addWidget(spin)
        self._add_button(range_layout, "Apply", self._apply_to_range)
        self.override_status_label = QLabel("Customized: 0 / 0 T")
        self.override_status_label.setStyleSheet("color: gray;")
        ctrl.addWidget(self.override_status_label)

        # Animation delay is measured in seconds.
        self._add_separator(ctrl)
        anim = QHBoxLayout()
        ctrl.addLayout(anim)
        self.play_btn = self._add_button(anim, "Play", self._toggle_play)
        anim.addWidget(QLabel("Delay (s):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 5.0)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setDecimals(1)
        self.delay_spin.setValue(0.5)
        anim.addWidget(self.delay_spin)

        # Save YAML and display load/save confirmation messages.
        self._add_separator(ctrl)
        self._add_button(ctrl, "Save thresholds to YAML", self._save_yaml)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: green;")
        self.status_label.setWordWrap(True)
        ctrl.addWidget(self.status_label)
        ctrl.addStretch()

        # Right panel: equal initial heights give both square views room.
        # Later redraws preserve the user's draggable divider position.
        self.display_frame = QWidget()
        display = QVBoxLayout(self.display_frame)
        layout.addWidget(self.display_frame, 1)
        self.display_splitter = QSplitter(Qt.Orientation.Vertical)
        self.display_splitter.setChildrenCollapsible(False)
        display.addWidget(self.display_splitter)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.display_splitter.addWidget(image_panel)
        self.header_label = QLabel()
        self.header_label.setWordWrap(True)
        self.header_label.setStyleSheet("font: 12pt Helvetica;")
        image_layout.addWidget(self.header_label)
        self.img_label = EmbryoLabel()
        image_layout.addWidget(self.img_label, 1)
        self.cap_label = QLabel()
        self.cap_label.setWordWrap(True)
        self.cap_label.setStyleSheet("color: gray;")
        image_layout.addWidget(self.cap_label)

        histogram_panel = QWidget()
        histogram_layout = QVBoxLayout(histogram_panel)
        histogram_layout.setContentsMargins(0, 0, 0, 0)
        histogram_layout.addWidget(QLabel("Gaussian-smoothed intensity histogram"))
        self.hist_figure = Figure(figsize=(5, 5), layout="constrained")
        self.hist_canvas = FigureCanvasQTAgg(self.hist_figure)
        self.hist_canvas.setMinimumHeight(130)
        self.hist_ax = self.hist_figure.add_subplot(111)
        self.hist_ax.set_box_aspect(1)
        self.hist_ax.set_axis_off()
        self.hist_ax.text(0.5, 0.5, "Load a file to display its histogram",
                          ha="center", va="center", transform=self.hist_ax.transAxes)
        histogram_layout.addWidget(self.hist_canvas, 1)
        self.hist_note = QLabel()
        self.hist_note.setWordWrap(True)
        histogram_layout.addWidget(self.hist_note)
        self.display_splitter.addWidget(histogram_panel)
        self.display_splitter.setStretchFactor(0, 1)
        self.display_splitter.setStretchFactor(1, 1)
        QTimer.singleShot(0, self._initialize_display_split)

    def _initialize_display_split(self):
        """Set the initial split after Qt has assigned the panel its height."""
        height = self.display_splitter.height()
        self.display_splitter.setSizes([height // 2, height // 2])

    def _add_button(self, layout, text, callback):
        """Add a button connected to the supplied callback."""
        button = QPushButton(text)
        button.clicked.connect(lambda _: callback())
        layout.addWidget(button)
        return button

    def _add_separator(self, layout):
        """Add a horizontal divider between control groups."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

    def _add_slider(self, layout, title, minimum, maximum, value, scale=1):
        """Add a slider with a label showing its current numeric value."""
        label = QLabel(f"{title}: {value * scale}")
        layout.addWidget(label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(
            lambda v: label.setText(f"{title}: {v * scale}"))
        slider.valueChanged.connect(lambda _: self._update())
        layout.addWidget(slider)
        return slider

    def _browse(self):
        """Open a native file dialog filtered to ND2 and TIF files."""
        path, _ = QFileDialog.getOpenFileName(
            self.root, "Select ND2 or TIF file", "",
            "ND2 files (*.nd2);;TIFF files (*.tif);;All files (*.*)",
        )
        if path:
            self.path_entry.setText(path)
            self._load_file()

    def _load_file(self):
        """Load the file specified in the path entry and update all controls."""
        path = self.path_entry.text().strip()
        if not path:
            return
        if not Path(path).is_file():
            QMessageBox.critical(self.root, "Error", f"File not found:\n{path}")
            return

        self.status_label.setText("Loading...")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        # Close the previous TIF accessor if one exists (releases file handle).
        if hasattr(self.max_proj, 'close'):
            self.max_proj.close()

        self.file_path = path
        self.max_proj, self.channel_names, self.is_nd2 = load_file(path)
        self.T, self.P, self.C, self.Y, self.X = self.max_proj.shape
        self.custom_timepoint_params = {}
        self._class_map_key = None

        # Reset embryo index to the first position.
        self.embryo_idx = 0

        # Populate the channel dropdown with the names read from the file.
        self.channel_combo.clear()
        self.channel_combo.addItems(self.channel_names)
        self.channel_combo.setCurrentIndex(0)

        # Set the T slider range to match the number of timepoints.
        self.t_slider.setMaximum(max(0, self.T - 1))
        self.t_slider.setValue(0)
        self.range_start_spin.setValue(0)
        self.range_end_spin.setValue(0)
        self.range_start_spin.setMaximum(max(0, self.T - 1))
        self.range_end_spin.setMaximum(max(0, self.T - 1))

        # Show embryo navigation only for multi-position ND2 files.
        # Single-embryo TIFs have P=1 so navigation is meaningless.
        if self.P > 1:
            self.nav_frame.show()
        else:
            self.nav_frame.hide()
        self._update_embryo_label()

        self.status_label.setText(
            f"Loaded: T={self.T}, P={self.P}, C={self.C}, "
            f"Y={self.Y}, X={self.X}")
        self._update_override_status()
        self._update()

    # ── Embryo navigation ─────────────────────────────────────────────────

    def _prev_embryo(self):
        """Navigate to the previous embryo position, wrapping around."""
        if self.P <= 1:
            return
        self.embryo_idx = (self.embryo_idx - 1) % self.P
        self._update_embryo_label()
        self._update()

    def _next_embryo(self):
        """Navigate to the next embryo position, wrapping around."""
        if self.P <= 1:
            return
        self.embryo_idx = (self.embryo_idx + 1) % self.P
        self._update_embryo_label()
        self._update()

    def _update_embryo_label(self):
        """Update the navigation label to show the current position index."""
        self.embryo_label.setText(f"Embryo {self.embryo_idx} / {self.P - 1}")

    # ── Rendering ─────────────────────────────────────────────────────────

    def _on_method_change(self):
        """Enable/disable controls based on method and trigger update."""
        method = self.method_combo.currentText()
        self.pct_slider.setEnabled(method != "Multi-Level Otsu")

        # Hide inactive level controls because the checkbox count varies,
        # so an inactive row would be misleading clutter.
        self.levels_frame.setVisible(method == "Multi-Level Otsu")
        self.levels_check_frame.setVisible(method == "Multi-Level Otsu")
        if method == "Multi-Level Otsu" and not self.level_vars:
            self._rebuild_level_checkboxes()
        self._update()

    def _rebuild_level_checkboxes(self):
        """Recreate one checkbox per intensity class for the current level count.

        The checkboxes are destroyed and rebuilt rather than hidden because
        the number of classes changes with the Levels spinbox. Selection is
        reset to the brightest class instead of being carried over: class
        indices are relative to the level count, so class 2 of 3 covers a
        different intensity band than class 2 of 4 and preserving the old
        checks would silently change which pixels are selected.
        """
        while self.levels_check_layout.count():
            widget = self.levels_check_layout.takeAt(0).widget()
            widget.setParent(None)
            widget.deleteLater()
        self.level_vars = []

        levels = self.levels_spin.value()
        for i in range(levels):
            # Default to the brightest class alone, which reproduces a plain
            # global Otsu threshold and is the usual starting point.
            checkbox = QCheckBox(str(i))
            checkbox.setChecked(i == levels - 1)
            self.level_vars.append(checkbox)
            checkbox.toggled.connect(lambda _: self._update())
            self.levels_check_layout.addWidget(checkbox)

    def _on_levels_change(self):
        """Rebuild the class checkboxes after the level count changes."""
        self._rebuild_level_checkboxes()
        self._update()

    def _selected_levels(self):
        """Return the indices of the currently checked intensity classes."""
        return [i for i, var in enumerate(self.level_vars) if var.isChecked()]

    def _method_code(self):
        """Return the internal method string for the current UI selection."""
        method_map = {
            "Percentile": "percentile",
            "Multi-Level Otsu": "multiotsu",
            "Percentile -> Otsu ROI": "percentile_otsu_roi",
        }
        return method_map.get(self.method_combo.currentText(), "percentile")

    def _current_params(self):
        """Return threshold parameters represented by the current controls."""
        ch_idx = self.channel_combo.currentIndex()
        if ch_idx < 0:
            ch_idx = 0
        channel = self.channel_names[ch_idx] if self.channel_names else f"Ch{ch_idx}"
        method = self._method_code()
        params = {
            "channel": channel,
            "channel_index": ch_idx,
            "sigma": self.sigma_slider.value(),
            "percentile": (self.pct_slider.value() * 0.5),
            "method": method,
            "invert": self.invert_check.isChecked(),
        }
        if method == "multiotsu":
            params["levels"] = self.levels_spin.value()
            params["selected_levels"] = self._selected_levels()
        return params

    def _apply_to_current_t(self):
        """Store current controls for the currently selected timepoint."""
        if self.max_proj is None:
            return
        t = int(self.t_slider.value())
        self.custom_timepoint_params[t] = dict(self._current_params())
        self._update_override_status()

    def _apply_to_range(self):
        """Store current controls for every timepoint in an inclusive range."""
        if self.max_proj is None:
            return
        start = int(self.range_start_spin.value())
        end = int(self.range_end_spin.value())
        if start > end:
            start, end = end, start
        start = max(0, min(start, self.T - 1))
        end = max(0, min(end, self.T - 1))
        params = dict(self._current_params())
        for t in range(start, end + 1):
            self.custom_timepoint_params[t] = dict(params)
        self.range_start_spin.setValue(start)
        self.range_end_spin.setValue(end)
        self._update_override_status()

    def _update_override_status(self):
        total = self.T if self.max_proj is not None else 0
        customized = len(self.custom_timepoint_params)
        self.override_status_label.setText(f"Customized: {customized} / {total} T")

    def _materialize_timepoint_params(self, default_params):
        """Build the full indexed parameter list written to YAML."""
        params_by_t = [dict(default_params) for _ in range(self.T)]
        for t, params in self.custom_timepoint_params.items():
            if 0 <= t < self.T:
                params_by_t[t] = dict(params)
        return params_by_t

    def _update(self):
        """Recompute the mask and redraw the embryo image and histogram.

        Called whenever any parameter changes: T slider, sigma, percentile,
        channel selection, or embryo navigation. Each call applies the
        Gaussian filter and connected-component detection from scratch.
        PyQt6 updates are event-driven (not polling), so the filter only
        runs when a widget value actually changes.
        """
        if self.max_proj is None:
            return

        t = self.t_slider.value()
        p = self.embryo_idx
        ch_idx = self.channel_combo.currentIndex()
        if ch_idx < 0:
            ch_idx = 0
        sigma = self.sigma_slider.value()
        percentile = (self.pct_slider.value() * 0.5)
        channel = self.channel_names[ch_idx]
        
        method = self._method_code()
        invert = self.invert_check.isChecked()
        levels = self.levels_spin.value()
        selected_levels = self._selected_levels()

        h_str = (
            f"T = {t}  |  {channel}  |  sigma = {sigma}  |  "
            f"method = {method}  |  percentile = {percentile}"
        )
        if method == "multiotsu":
            shown = ",".join(str(i) for i in selected_levels) or "none"
            h_str += f"  |  levels = {levels}  |  selected = {shown}"
        if invert:
            h_str += "  |  INVERTED"
        self.header_label.setText(h_str)

        # Extract the 2-D image for the current (t, p, ch) combination.
        img = self.max_proj[t, p, ch_idx]

        # Gaussian blur suppresses noise and isolated bright spots before
        # thresholding, matching the processing in compute_shift_xy.
        smoothed = gaussian_filter(img, sigma=sigma)

        # Detect the largest connected component above the percentile
        # threshold. This is the same function used by centroid_align_xy.py
        # so that the mask the user sees here is exactly what the alignment
        # script will detect.
        #
        # The class map is reused for both the mask and the per-class overlay,
        # and is only recomputed when the image or the level count changes.
        class_map = None
        if method == "multiotsu":
            key = (t, p, ch_idx, sigma, levels)
            if key != self._class_map_key:
                # Reuse the shared JAX boundaries and class map for both
                # the histogram and mask, including the blank-frame fallback.
                self._class_thresholds, self._class_map = multiotsu_thresholds_and_classes(
                    smoothed, levels)
                self._class_map_key = key
            class_map = self._class_map

        mask, centroid = find_largest_mask_xy(smoothed, percentile, method=method, invert=invert,
                                              levels=levels, selected_levels=selected_levels, class_map=class_map)

        # Render the composite image and scale it for display.
        pil_img = render_embryo(img, mask, centroid,
                                class_map=class_map, selected_levels=selected_levels)
        pil_img = pil_img.resize(
            (self.display_size, self.display_size), Image.LANCZOS)
        # Copy the pixel data so the Qt image owns its buffer after the
        # temporary PIL image and byte string leave this method.
        image = QImage(
            pil_img.tobytes(), pil_img.width, pil_img.height,
            pil_img.width * 4, QImage.Format.Format_RGBA8888,
        ).copy()
        self.img_label.setPixmap(QPixmap.fromImage(image))

        # Update the caption with quantitative diagnostics.
        area = int(mask.sum())
        cx, cy = centroid[1], centroid[0]
        mean_val = float(img[mask].mean()) if mask.any() else 0.0
        cap = (
            f"Embryo {p}  |  Area: {area} px ({area / mask.size * 100:.1f}%)  |  "
            f"Centroid: ({cx:.1f}, {cy:.1f})  |  "
            f"Mean intensity: {mean_val:.1f}")

        # Report how much of the coloured overlay survives into the mask. The
        # mask keeps only the largest connected component, so a selection whose
        # classes do not touch loses everything outside the biggest piece.
        # Anything below 100% means the crosshair may sit on a different
        # structure than the overlay suggests.
        if class_map is not None and selected_levels:
            selected_px = int(np.isin(class_map, selected_levels).sum())
            if selected_px:
                cap += f"  |  Largest component: {area / selected_px * 100:.0f}% of selected"
        self.cap_label.setText(cap)
        self._update_histogram(smoothed, mask, method, percentile,
                               class_map, selected_levels, invert)

    def _update_histogram(self, smoothed, mask, method, percentile,
                          class_map, selected_levels, invert):
        """Plot full-frame smoothed intensities with the masking thresholds.

        Ordinary methods colour pixels by final mask membership. Multi-level
        Otsu instead shows every intensity class using the overlay colours,
        including unchecked classes, and identifies the selected classes.
        All groups share bins spanning the full intensity range.
        """
        ax = self.hist_ax
        ax.clear()
        ax.set_box_aspect(1)
        pixels = smoothed.ravel()
        edges = np.histogram_bin_edges(pixels, bins=80)
        markers = []
        if class_map is not None:
            groups = []
            colors = []
            labels = []
            for i in range(self.levels_spin.value()):
                groups.append(smoothed[class_map == i])
                colors.append(np.asarray(LEVEL_COLORS[i % len(LEVEL_COLORS)]) / 255)
                state = "selected" if i in selected_levels else "not displayed"
                labels.append(f"Class {i} ({state})")
            ax.hist(groups, bins=edges, stacked=True, color=colors, label=labels)
            for i, threshold in enumerate(self._class_thresholds):
                ax.axvline(threshold, color="#8a6500", linestyle="--", linewidth=1.2,
                           label="Otsu boundaries" if i == 0 else "_nolegend_")
                markers.append(f"{i}/{i + 1}: {threshold:.6g}")
            note = "Classes are numbered from 0 (darkest), matching the checkboxes. "
            note += "Colours show classes before inclusion and largest-component selection. "
            if markers:
                note += "Otsu boundaries: " + "; ".join(markers) + "."
            else:
                note += "Too few distinct intensities: all pixels assigned to class 0; no Otsu boundaries."
        else:
            # Membership comes directly from the displayed mask, so spatial
            # filtering and inversion are reflected in the histogram too.
            ax.hist([smoothed[~mask], smoothed[mask]], bins=edges, stacked=True,
                    color=["#7a8793", "#ff3232"],
                    label=["Outside final mask", "Inside final mask (red overlay)"])
            q = np.percentile(smoothed, percentile)
            gate = "Percentile threshold" if method == "percentile" else "Percentile gate"
            ax.axvline(q, color="#8a6500", linestyle="--", linewidth=1.2,
                       label=f"{gate}: P{percentile:g} = {q:.6g}")
            note = "Colours show final mask after largest-component selection. "
            if method == "percentile_otsu_roi":
                threshold = self._roi_otsu_threshold(smoothed, q)
                if threshold is not None:
                    ax.axvline(threshold, color="#007c91", linestyle=":", linewidth=1.5,
                               label=f"ROI-derived Otsu threshold: {threshold:.6g}")
                    note += "Otsu is fitted to the largest percentile region, then applied to the full frame."
                else:
                    note += "ROI empty or constant: masking uses the percentile threshold only."
            else:
                note += "Pixels must exceed the percentile threshold before inclusion in trhe final mask. "
        if invert:
            note += " Invert Mask is ON: the threshold selection is complemented before keeping the largest component."
        self.hist_note.setText(note)
        ax.set_xlabel("Gaussian-smoothed intensity ", fontsize=9)
        ax.set_ylabel("Pixel count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        # Keep the legend above the plot so it does not widen the panel.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                  fontsize=8, frameon=False, borderaxespad=0)
        self.hist_canvas.draw_idle()

    def _roi_otsu_threshold(self, smoothed, percentile_threshold):
        """Recover the ROI threshold using the masking helper's calculation."""
        # ROI construction uses default connectivity, whereas final mask
        # selection uses eight neighbours. Preserve that distinction here.
        regions, _ = label(smoothed > percentile_threshold)
        sizes = np.bincount(regions.ravel())
        sizes[0] = 0
        if sizes.max() == 0:
            return None
        roi = smoothed[regions == sizes.argmax()]
        vmin, vmax = roi.min(), roi.max()
        if vmax <= vmin:
            return None
        uint8_roi = np.clip((roi - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(uint8_roi.reshape(-1, 1), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold / 255.0 * (vmax - vmin) + vmin

    # ── Animation ─────────────────────────────────────────────────────────

    def _toggle_play(self):
        """Toggle between playing and paused states."""
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._animate()
        else:
            # Cancel the pending callback so the animation actually stops.
            self.animation_timer.stop()

    def _animate(self):
        """Advance T by one frame and schedule the next advance.

        Uses QTimer rather than time.sleep() so the PyQt6 event
        loop remains responsive during animation. The T slider wraps
        around from T-1 back to 0.
        """
        if not self.playing or self.max_proj is None:
            return
        t = (self.t_slider.value() + 1) % self.T
        self.t_slider.setValue(t)
        delay_ms = int(self.delay_spin.value() * 1000)
        self.animation_timer.start(delay_ms)

    # ── Save YAML ─────────────────────────────────────────────────────────

    def _save_yaml(self):
        """Save the current embryo's threshold parameters to a YAML file.

        Each embryo gets its own YAML file so that different embryos can
        have different sigma and percentile values. The YAML schema matches
        what centroid_align_xy.py expects: parameters (channel, channel_index,
        sigma, percentile), source (file path, image shape), and diagnostics
        (mask area, centroid, mean intensity at t=0).

        Filename pattern:
          ND2 input:  {base}_P{n}_{channel}_threshold.yaml
          TIF input:  {base}_{channel}_threshold.yaml
        """
        if self.max_proj is None:
            QMessageBox.warning(self.root, "No data", "Load a file first.")
            return

        p = self.embryo_idx
        ch_idx = self.channel_combo.currentIndex()
        if ch_idx < 0:
            ch_idx = 0
        params = self._current_params()
        channel = params["channel"]
        sigma = params["sigma"]
        percentile = params["percentile"]
        method = params["method"]

        t = self.t_slider.value()
        invert = params.get("invert", False)
        
        img = self.max_proj[t, p, ch_idx]
        smoothed = gaussian_filter(img, sigma=sigma)
        mask, centroid = find_largest_mask_xy(
            smoothed, percentile, method=method, invert=invert,
            levels=params.get("levels", self.levels_spin.value()),
            selected_levels=params.get("selected_levels"))
        area = int(mask.sum())
        
        output = {
            "parameters": params,
            "timepoint_parameters": self._materialize_timepoint_params(params),
            "source": {
                "file": str(Path(self.file_path)),
                "image_shape": {
                    "T": self.T, "P": self.P,
                    "Y": self.Y, "X": self.X,
                },
            },
            "diagnostics": {
                "time_point": t,
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "embryos": [{
                    "id": p,
                    "mask_area_px": area,
                    "mask_area_pct": round(area / mask.size * 100, 2),
                    "mean_intensity": round(float(img[mask].mean()), 2) if mask.any() else 0.0,
                    "centroid": [round(float(centroid[1]), 2),
                                 round(float(centroid[0]), 2)],
                }],
            },
        }

        base = Path(self.file_path).stem
        if self.is_nd2:
            fname = f"{base}_P{p}_{channel}_threshold.yaml"
        else:
            # TIF files are single-embryo so no position index is needed.
            # The .ome suffix (if present) is already removed by Path.stem,
            # but the base might still end with ".ome" if the file was named
            # e.g. "nd1188_P0.ome.tif". Strip it for a cleaner filename.
            if base.endswith(".ome"):
                base = base[:-4]
            fname = f"{base}_{channel}_threshold.yaml"

        # Anchor the YAML next to the input file. A bare relative name
        # resolves against the working directory, which is the read-only
        # system volume when the app is launched from Finder.
        out_path = Path(self.file_path).resolve().parent / fname
        with open(out_path, "w") as fout:
            yaml.dump(output, fout, default_flow_style=False, sort_keys=False)

        self.status_label.setText(f"Saved \u2192 {fname}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    root = QWidget()
    window = ThresholdApp(root)
    root.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
