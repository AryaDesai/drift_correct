"""Minimal PyQt6 front end for centroid_align_xy.py."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtGui import QCloseEvent, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


SCRIPT_PATH = Path(__file__).with_name("centroid_align_xy.py")
from tool_runtime import tool_command


class CentroidAlignWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.process: QProcess | None = None

        self.setWindowTitle("XY Centroid Alignment")
        self.setMinimumWidth(680)

        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("Select a threshold YAML file")
        self.yaml_edit.textChanged.connect(self._update_run_button)

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_yaml)

        yaml_row = QHBoxLayout()
        yaml_row.addWidget(self.yaml_edit, 1)
        yaml_row.addWidget(browse_button)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.01, 1000.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setValue(2.0)
        self.fps_spin.setSuffix(" fps")

        self.use_nd2_check = QCheckBox("Load directly from the raw ND2 file")
        self.use_nd2_check.setToolTip(
            "Uses the ND2 path stored in the YAML instead of the per-embryo TIFF."
        )

        self.enlarge_canvas_check = QCheckBox("Enlarge canvas to prevent clipping")
        self.enlarge_canvas_check.setChecked(True)
        self.enlarge_canvas_check.setToolTip(
            "Runs the default two-pass alignment. This path calculates the shifts first,expands the canvas, then corrects the shift. No edge data is lost. Turning this off may clip edge data."
        )

        self.use_gpu_check = QCheckBox("Use GPU acceleration")
        self.use_gpu_check.setToolTip("Uses the GPU, if available, for acceleration of the alignment process.")

        self.low_memory_check = QCheckBox("Use low-memory streaming")
        self.low_memory_check.setToolTip(
            "Writes aligned frames as they are produced instead of holding the full result in RAM. If your filesize is larger than the available RAM, turn this option on to avoid running out of memory."
        )

        form = QFormLayout()
        form.addRow("Threshold YAML:", yaml_row)
        form.addRow("Output video speed:", self.fps_spin)
        form.addRow("", self.use_nd2_check)
        form.addRow("", self.enlarge_canvas_check)
        form.addRow("", self.use_gpu_check)
        form.addRow("", self.low_memory_check)

        self.run_button = QPushButton("Run alignment")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.run_button)

        self.status_label = QLabel("Ready")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Processing output will appear here.")
        self.log.document().setMaximumBlockCount(5000)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _browse_yaml(self) -> None:
        current = Path(self.yaml_edit.text().strip()).expanduser()
        start_dir = current.parent if current.parent.is_dir() else SCRIPT_PATH.parent
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select threshold YAML",
            str(start_dir),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if filename:
            self.yaml_edit.setText(filename)

    def _update_run_button(self) -> None:
        not_running = self.process is None
        self.run_button.setEnabled(not_running and bool(self.yaml_edit.text().strip()))

    def _arguments(self, yaml_path: Path) -> list[str]:
        arguments = [str(yaml_path.resolve()), "--fps", str(self.fps_spin.value())]
        if self.use_nd2_check.isChecked():
            arguments.append("--use_nd2")
        if not self.enlarge_canvas_check.isChecked():
            arguments.append("--no_enlarge_canvas")
        if self.use_gpu_check.isChecked():
            arguments.append("--use_gpu")
        if self.low_memory_check.isChecked():
            arguments.append("--low_memory")
        return arguments

    def _run(self) -> None:
        yaml_path = Path(self.yaml_edit.text().strip()).expanduser()
        if not yaml_path.is_file():
            QMessageBox.warning(self, "Invalid YAML file", "Select an existing YAML file.")
            return
        if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
            answer = QMessageBox.question(
                self,
                "Unexpected file type",
                "The selected file does not have a .yaml or .yml extension. Continue anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        program, arguments = tool_command("align-worker", self._arguments(yaml_path))
        self.log.clear()
        self.log.appendPlainText(f"Running: {program} {arguments!r}\n")
        self.status_label.setText("Running…")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        process = QProcess(self)
        self.process = process
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._finished)
        process.start(program, arguments)

    def _read_output(self) -> None:
        if self.process is None:
            return
        output = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if output:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(output)
            self.log.ensureCursorVisible()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText("\nCould not start alignment: " + self.process.errorString())
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._read_output()
        succeeded = exit_code == 0 and _exit_status == QProcess.ExitStatus.NormalExit
        self.status_label.setText("Completed successfully" if succeeded else f"Stopped with error code {exit_code}")
        self.log.appendPlainText("\nAlignment completed." if succeeded else f"\nAlignment stopped (exit code {exit_code}).")
        if self.process is not None:
            self.process.deleteLater()
        self.process = None
        self.cancel_button.setEnabled(False)
        self._update_run_button()

    def _cancel(self) -> None:
        if self.process is None:
            return
        self.status_label.setText("Stopping…")
        self.cancel_button.setEnabled(False)
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.process is None:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Alignment is running",
            "Stop the alignment and close the window?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.process.kill()
            self.process.waitForFinished(3000)
            event.accept()
        else:
            event.ignore()


def main() -> int:
    app = QApplication(sys.argv)
    window = CentroidAlignWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
