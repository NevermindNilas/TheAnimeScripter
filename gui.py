from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QTextEdit,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QStackedWidget,
)
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import QTimer, Qt
from src.uiLogic import uiStyleSheet, runCommand, StreamToTextEdit, loadSettings, saveSettings, updatePresence

import sys
import time
from pypresence import Presence
from main import scriptVersion

TITLE = f"The Anime Scripter - {scriptVersion} (Alpha)"

class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(1280, 720)

        self.client_id = "1213461768785891388"
        self.RPC = Presence(self.client_id)
        try:
            self.RPC.connect()
        except ConnectionRefusedError:
            print("Could not connect to Discord. Is Discord running?")

        self.start_time = int(time.time())
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePresence)
        self.timer.start(1000)

        self.setStyleSheet(uiStyleSheet())

        self.stackedWidget = QStackedWidget()
        self.centralWidget = QWidget()
        self.stackedWidget.addWidget(self.centralWidget)
        self.setCentralWidget(self.stackedWidget)

        self.createLayouts()
        self.createWidgets()

        self.settingsFile = "settings.json"
        loadSettings(self)

    def createLayouts(self):
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.pathLayout = QVBoxLayout()
        self.checkboxLayout = QVBoxLayout()
        self.outputLayout = QVBoxLayout()

    def createWidgets(self):
        inputFields = [
            ("Interpolate Factor:", 2, 100),
            ("Upscale Factor:", 2, 4),
            ("Resize Factor:", 1, 4)
        ]

        for label, defaultValue, maxValue in inputFields:
            layout, entry = self.createInputField(label, defaultValue, maxValue)
            self.checkboxLayout.addLayout(layout)

        self.pathGroup = self.createGroup("Paths", self.pathLayout)
        self.checkboxGroup = self.createGroup("Options", self.checkboxLayout)
        self.outputGroup = self.createGroup("Terminal", self.outputLayout)

        self.inputEntry = self.createPathWidgets("Input Path:", self.browseInput)
        self.outputEntry = self.createPathWidgets("Output Path:", self.browseOutput)

        for option in ["Resize", "Dedup", "Interpolate", "Upscale", "Segment", "Depth"]:
            self.createCheckbox(option)

        self.outputWindow = QTextEdit()
        self.outputWindow.setReadOnly(True)
        self.outputLayout.addWidget(self.outputWindow)

        sys.stdout = StreamToTextEdit(self.outputWindow)
        sys.stderr = StreamToTextEdit(self.outputWindow)

        self.runButton = self.createButton("Run", lambda: runCommand(self, TITLE))
        self.settingsButton = self.createButton("Settings", self.openSettings)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.runButton)
        self.buttonLayout.addWidget(self.settingsButton)

        self.addWidgetsToLayout(self.layout, [self.pathGroup, self.checkboxGroup, self.outputGroup], 5)
        self.layout.addLayout(self.buttonLayout)

    def createGroup(self, title, layout):
        group = QGroupBox(title)
        group.setLayout(layout)
        return group

    def createPathWidgets(self, label, slot):
        layout = QHBoxLayout()
        label = QLabel(label)
        entry = QLineEdit()
        entry.setFixedWidth(1050)
        button = self.createButton("Browse", slot)
        layout.addWidget(label)
        layout.addWidget(entry)
        layout.addWidget(button)
        self.pathLayout.addLayout(layout)
        return entry

    def createInputField(self, label, defaultValue, maxValue):
        layout = QHBoxLayout()
        layout.addStretch(1)
        label = QLabel(label)
        entry = QLineEdit()
        entry.setText(str(defaultValue))
        entry.setValidator(QIntValidator(0, maxValue))
        entry.setFixedWidth(30)
        entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(entry)
        return layout, entry

    def createCheckbox(self, text):
        checkbox = QCheckBox(text)
        self.checkboxLayout.addWidget(checkbox)

    def createButton(self, text, slot):
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    def addWidgetsToLayout(self, layout, widgets, spacing):
        for widget in widgets:
            layout.addWidget(widget)
            layout.addSpacing(spacing)

    def browseInput(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if filePath:
            self.inputEntry.setText(filePath)

    def browseOutput(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.outputEntry.setText(directory)

    def updatePresence(self):
        updatePresence(self.RPC, self.start_time, TITLE)

    def closeEvent(self, event):
        saveSettings(self)
        event.accept()

    def openSettings(self):
        self.settingsWidget = QWidget()
        settingsLayout = QVBoxLayout()
        settingsLabel = QLabel("Settings go here")
        settingsLayout.addWidget(settingsLabel)
        backButton = self.createButton("Back", self.goBack)
        settingsLayout.addWidget(backButton)
        self.settingsWidget.setLayout(settingsLayout)
        self.stackedWidget.addWidget(self.settingsWidget)
        self.stackedWidget.setCurrentWidget(self.settingsWidget)

    def goBack(self):
        self.stackedWidget.removeWidget(self.settingsWidget)
        self.stackedWidget.setCurrentWidget(self.centralWidget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())