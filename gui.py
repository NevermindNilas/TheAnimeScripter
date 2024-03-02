import sys
import time
import os
import json

from pypresence import Presence
from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QTextEdit,
)

class StreamToTextEdit:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message):
        self.text_edit.append(message)

    def flush(self):
        pass


class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The Anime Scripter - 1.4.0")
        self.setFixedSize(1280, 720)

        self.client_id = "1213461768785891388"
        self.RPC = Presence(self.client_id)
        self.RPC.connect()

        self.start_time = int(time.time())
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePresence)
        self.timer.start(1000)

        self.setStyleSheet("""
            QWidget {
                background-color: #202020;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #404040;
                color: #FFFFFF;
            }
            QLineEdit {
                background-color: #404040;
                color: #FFFFFF;
            }
            QCheckBox {
                color: #FFFFFF;
            }
        """)

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        self.layout = QVBoxLayout()
        centralWidget.setLayout(self.layout)

        self.inputLayout = QHBoxLayout()
        self.inputLabel = QLabel("Input Path:")
        self.inputEntry = QLineEdit()
        self.inputButton = QPushButton("Browse")
        self.inputButton.clicked.connect(self.browseInput)
        self.inputLayout.addWidget(self.inputLabel)
        self.inputLayout.addWidget(self.inputEntry)
        self.inputLayout.addWidget(self.inputButton)

        self.outputLayout = QHBoxLayout()
        self.outputLabel = QLabel("Output Path:")
        self.outputEntry = QLineEdit()
        self.outputButton = QPushButton("Browse")
        self.outputButton.clicked.connect(self.browseOutput)
        self.outputLayout.addWidget(self.outputLabel)
        self.outputLayout.addWidget(self.outputEntry)
        self.outputLayout.addWidget(self.outputButton)

        self.checkboxLayout = QVBoxLayout()
        self.createCheckbox("Dedup")
        self.createCheckbox("Interpolate")
        self.createCheckbox("Upscale")
        self.createCheckbox("Half Precision Mode")
        self.createCheckbox("Segment")
        self.createCheckbox("Scene Change")
        self.createCheckbox("Depth")
        self.createCheckbox("Keep Audio")

        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(self.runCommand)

        self.outputWindow = QTextEdit()
        self.outputWindow.setReadOnly(True)

        sys.stdout = StreamToTextEdit(self.outputWindow)
        sys.stderr = StreamToTextEdit(self.outputWindow)

        self.layout.addLayout(self.inputLayout)
        self.layout.addLayout(self.outputLayout)
        self.layout.addLayout(self.checkboxLayout)
        self.layout.addWidget(self.runButton)
        self.layout.addWidget(self.outputWindow)

        self.settingsFile = "settings.json"
        self.loadSettings()

    def createCheckbox(self, text):
        checkbox = QCheckBox(text)
        self.checkboxLayout.addWidget(checkbox)

    def browseInput(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if filePath:
            self.inputEntry.setText(filePath)

    def browseOutput(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.outputEntry.setText(directory)

    def updatePresence(self):
        self.RPC.update(
            details="Idle",
            start=self.start_time,
            large_image="icon",
            small_image="icon",
            large_text="The Anime Scripter - 1.4.0",
            small_text="Idle",
        )

    def runCommand(self):
        self.RPC.update(
            details="Processing",
            start=self.start_time,
            large_image="icon",
            small_image="icon",
            large_text="The Anime Scripter - 1.4.0",
            small_text="Processing",
        )
        command = ["./main.exe"]

        if self.inputEntry.text():
            command.append("--input")
            command.append(self.inputEntry.text())
        else:
            self.outputWindow.append("Input file not selected")
            return

        if self.outputEntry.text():
            command.append("--output")
            command.append(self.outputEntry.text())

        for i in range(self.checkboxLayout.count()):
            checkbox = self.checkboxLayout.itemAt(i).widget()
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                command.append(f"--{checkbox.text().lower().replace(' ', '_')}")
            if checkbox.text() == "Half Precision Mode":
                command.append("--half 1")
        
        if not os.path.isfile("main.exe"):
            self.outputWindow.append("main.exe not found")
            return
        
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handleStdout)
        self.process.readyReadStandardError.connect(self.handleStderr)
        self.process.start(command[0], command[1:])

    def handleStdout(self):
        data = bytes(self.process.readAllStandardOutput()).decode()
        self.outputWindow.append(data)

    def handleStderr(self):
        data = bytes(self.process.readAllStandardError()).decode()
        self.outputWindow.append(data)

    def loadSettings(self):
        if os.path.exists(self.settingsFile):
            with open(self.settingsFile, "r") as file:
                settings = json.load(file)
            self.inputEntry.setText(settings.get("input_path", ""))
            self.outputEntry.setText(settings.get("output_path", ""))
            for i in range(self.checkboxLayout.count()):
                checkbox = self.checkboxLayout.itemAt(i).widget()
                if isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(settings.get(checkbox.text(), False))

    def saveSettings(self):
        settings = {
            "input_path": self.inputEntry.text(),
            "output_path": self.outputEntry.text()
        }
        for i in range(self.checkboxLayout.count()):
            checkbox = self.checkboxLayout.itemAt(i).widget()
            if isinstance(checkbox, QCheckBox):
                settings[checkbox.text()] = checkbox.isChecked()
        with open(self.settingsFile, "w") as file:
            json.dump(settings, file, indent=4)

    def closeEvent(self, event):
        self.saveSettings()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())
