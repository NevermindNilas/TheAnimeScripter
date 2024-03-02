import sys
import time
from PyQt5.QtCore import QProcess, QTimer
from PyQt5.QtWidgets import (
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
from pypresence import Presence


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
        self.timer.timeout.connect(self.update_presence)
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

        self.layout.addLayout(self.inputLayout)
        self.layout.addLayout(self.outputLayout)
        self.layout.addLayout(self.checkboxLayout)
        self.layout.addWidget(self.runButton)
        self.layout.addWidget(self.outputWindow)

    def createCheckbox(self, text):
        checkbox = QCheckBox(text)
        self.checkboxLayout.addWidget(checkbox)

    def browseInput(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if filePath:
            self.inputEntry.setText(filePath)

    def browseOutput(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Select Output File")
        if filePath:
            self.outputEntry.setText(filePath)

    def update_presence(self):
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

        if self.outputEntry.text():
            command.append("--output")
            command.append(self.outputEntry.text())

        for i in range(self.checkboxLayout.count()):
            checkbox = self.checkboxLayout.itemAt(i).widget()
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                command.append(f"--{checkbox.text().lower().replace(' ', '_')}")

        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handleStdout)
        self.process.readyReadStandardError.connect(self.handleStderr)
        self.process.start(command[0], command[1:])

    def handleStdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.outputWindow.append(data)

    def handleStderr(self):
        data = self.process.readAllStandardError().data().decode()
        self.outputWindow.append(data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec_())
