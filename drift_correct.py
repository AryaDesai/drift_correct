"""Entry point for the three embryo drift-correction tools."""

# nuitka-project: --user-package-configuration-file={MAIN_DIRECTORY}/ffmpeg.nuitka-package.config.yml

import argparse
import sys

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout

from tool_runtime import configure_worker_output, tool_command


class DriftCorrectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drift correction")
        self.setMinimumWidth(340)
        self.processes = {}
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(QLabel("What do you want to do?"))
        for label, filename in (
            ("Convert ND2 to TIFF", "convert-gui"),
            ("Mask embryo", "mask-gui"),
            ("Align movie", "align-gui"),
        ):
            button = QPushButton(label)
            button.setMinimumHeight(42)
            button.clicked.connect(lambda checked=False, name=filename, control=button: self.launch(name, control))
            layout.addWidget(button)

    def launch(self, mode, button):
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.processes[process] = [button, ""]
        button.setEnabled(False)
        process.readyReadStandardOutput.connect(lambda: self.read_output(process))
        process.errorOccurred.connect(lambda error: self.process_error(process, error))
        process.finished.connect(lambda code, status: self.finished(process, code, status))
        program, arguments = tool_command(mode)
        process.start(program, arguments)

    def read_output(self, process):
        if process in self.processes:
            output = bytes(process.readAllStandardOutput()).decode(errors="replace")
            self.processes[process][1] = (self.processes[process][1] + output)[-12000:]

    def process_error(self, process, error):
        if error == QProcess.ProcessError.FailedToStart and process in self.processes:
            self.processes[process][1] = process.errorString()
            self.finished(process, -1, QProcess.ExitStatus.CrashExit)

    def finished(self, process, code, status):
        if process not in self.processes:
            return
        self.read_output(process)
        button, output = self.processes.pop(process)
        button.setEnabled(True)
        process.deleteLater()
        if code != 0 or status != QProcess.ExitStatus.NormalExit:
            message = QMessageBox(self)
            message.setWindowTitle("Tool stopped")
            message.setIcon(QMessageBox.Icon.Warning)
            message.setText(f"{button.text()} stopped with an error.")
            message.setDetailedText(output or f"Exit code: {code}")
            message.exec()

    def closeEvent(self, event):
        if self.processes:
            QMessageBox.information(self, "Tools open", "Close the tool windows before closing the launcher.")
            event.ignore()
        else:
            event.accept()


def launcher_main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DriftCorrectDialog()
    window.show()
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description="Embryo drift-correction tools")
    parser.add_argument("--tool", choices=(
        "convert-gui", "mask-gui", "align-gui", "convert-worker", "align-worker"
    ))
    # Parse only the routing prefix so worker flags (including --help) are
    # handled by that worker's own parser.
    if sys.argv[1:2] == ["--tool"]:
        options = parser.parse_args(sys.argv[1:3])
        remaining = sys.argv[3:]
    else:
        options = parser.parse_args()
        remaining = []
    if remaining and options.tool not in {"convert-worker", "align-worker"}:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    # Workers retain their existing argument parsers. Remove the routing
    # arguments before handing over control to a tool.
    sys.argv = [sys.argv[0], *remaining]
    configure_worker_output()
    # Explicit imports let Nuitka discover every tool, while lazy loading
    # avoids importing imaging libraries just to display the launcher.
    if options.tool == "convert-gui":
        from nd2_to_tif_gui import main as entry
    elif options.tool == "mask-gui":
        from find_threshold import main as entry
    elif options.tool == "align-gui":
        from centroid_align_xy_gui import main as entry
    elif options.tool == "convert-worker":
        from nd2_to_tif import main as entry
    elif options.tool == "align-worker":
        from centroid_align_xy import main as entry
    else:
        entry = launcher_main
    return entry()


if __name__ == "__main__":
    # PyInstaller must dispatch multiprocessing workers before our argument parser.
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
