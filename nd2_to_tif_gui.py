"""Thin PyQt6 front end for nd2_to_tif.py."""

import codecs
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

SCRIPT = Path(__file__).resolve().with_name("nd2_to_tif.py")
from tool_runtime import tool_command


class ConversionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Convert ND2 to TIFF")
        self.resize(650, 460)
        self.process = None
        self.input_folder = QLineEdit()
        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText("Default: tifs subfolder inside the input folder")
        self.all_positions = QCheckBox("All embryo positions")
        self.all_positions.setChecked(True)
        self.position = QSpinBox()
        self.position.setRange(0, 1000000)
        self.position.setEnabled(False)
        self.position.setToolTip("Zero-based embryo position: 0 is the first position.")
        self.all_positions.toggled.connect(lambda checked: self.position.setEnabled(not checked))
        self.parallel = QCheckBox("Process positions in parallel")
        self.parallel.setChecked(True)
        self.parallel.setToolTip("Uses multiple worker threads when processing all positions.")
        self.all_positions.toggled.connect(self.parallel.setEnabled)

        self.settings = QWidget()
        form = QFormLayout(self.settings)
        form.addRow("ND2 folder:", self.folder_row(self.input_folder))
        form.addRow("Output folder:", self.folder_row(self.output_folder))
        form.addRow("", self.all_positions)
        form.addRow("Position index:", self.position)
        form.addRow("", self.parallel)
        self.run_button = QPushButton("Convert")
        self.run_button.clicked.connect(self.run)
        self.status = QLabel("Ready")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(5000)
        layout = QVBoxLayout(self)
        note = QLabel("Converts ND2 files in filename order, then joins each embryo's TIFFs across time.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(self.settings)
        layout.addWidget(self.run_button)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

    def folder_row(self, field):
        row = QHBoxLayout()
        row.addWidget(field)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self.browse_folder(field))
        row.addWidget(browse)
        return row

    def browse_folder(self, field):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", field.text() or str(SCRIPT.parent))
        if folder:
            field.setText(folder)

    def arguments(self):
        args = [str(Path(self.input_folder.text().strip()).expanduser().resolve())]
        if self.output_folder.text().strip():
            args += ["--output_dir", str(Path(self.output_folder.text().strip()).expanduser().resolve())]
        if not self.all_positions.isChecked():
            args += ["--position", str(self.position.value())]
        args += ["--parallel", str(self.parallel.isChecked())]
        return args

    def run(self):
        folder = Path(self.input_folder.text().strip()).expanduser()
        if not self.input_folder.text().strip() or not folder.is_dir():
            QMessageBox.warning(self, "Select input", "Select an existing ND2 folder.")
            return
        if not any(folder.glob("*.nd2")):
            QMessageBox.warning(self, "No ND2 files", "The selected folder contains no .nd2 files.")
            return
        self.log.clear()
        self.status.setText("Converting…")
        self.settings.setEnabled(False)
        self.run_button.setEnabled(False)
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        process = QProcess(self)
        self.process = process
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_output)
        process.errorOccurred.connect(self.process_error)
        process.finished.connect(self.finished)
        program, arguments = tool_command("convert-worker", self.arguments())
        process.start(program, arguments)

    def read_output(self):
        if self.process is not None:
            text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text.replace("\r", "\n"))
            self.log.ensureCursorVisible()

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(self.process.errorString())
            self.finished(-1, QProcess.ExitStatus.CrashExit)

    def finished(self, code, status):
        self.read_output()
        succeeded = code == 0 and status == QProcess.ExitStatus.NormalExit
        self.status.setText("Completed" if succeeded else "Conversion failed — see log")
        if self.process is not None:
            self.process.deleteLater()
        self.process = None
        self.settings.setEnabled(True)
        self.run_button.setEnabled(True)

    def closeEvent(self, event):
        if self.process is not None:
            QMessageBox.information(self, "Conversion running", "Wait for conversion to finish before closing this window.")
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ConversionWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
