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
    QMenuBar,
    QMenu,
    QVBoxLayout,
    QLabel,
    QGroupBox,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer
from src.uiLogic import uiStyleSheet, runCommand, StreamToTextEdit, loadSettings, saveSettings
import sys
import time
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
        self.timer.timeout.connect(self.updatePresence)
        self.timer.start(1000)

        self.setStyleSheet(uiStyleSheet())

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.createMenuBar()
        self.createLayouts()
        self.createWidgets()

        self.settingsFile = "settings.json"
        loadSettings(self)

    def createMenuBar(self):
        self.menuBar = QMenuBar()
        self.settingsMenu = QMenu("Settings", self)
        self.settingsAction = QAction("Open Settings", self)
        self.settingsAction.triggered.connect(self.openSettings)
        self.settingsMenu.addAction(self.settingsAction)
        self.menuBar.addMenu(self.settingsMenu)
        self.setMenuBar(self.menuBar)

    def createLayouts(self):
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.pathLayout = QVBoxLayout()
        self.checkboxLayout = QVBoxLayout()
        self.outputLayout = QVBoxLayout()

    def createWidgets(self):
        self.pathGroup = self.createGroup("Paths", self.pathLayout)
        self.checkboxGroup = self.createGroup("Options", self.checkboxLayout)
        self.outputGroup = self.createGroup("Terminal", self.outputLayout)

        self.inputEntry = self.createPathWidgets("Input Path:", self.browseInput)
        self.outputEntry = self.createPathWidgets("Output Path:", self.browseOutput)

        for option in ["Dedup", "Interpolate", "Upscale", "Segment", "Scene Change", "Depth"]:
            self.createCheckbox(option)

        self.outputWindow = QTextEdit()
        self.outputWindow.setReadOnly(True)
        self.outputLayout.addWidget(self.outputWindow)

        sys.stdout = StreamToTextEdit(self.outputWindow)
        sys.stderr = StreamToTextEdit(self.outputWindow)

        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(lambda: runCommand(self))

        self.layout.addWidget(self.pathGroup)
        self.layout.addSpacing(5)
        self.layout.addWidget(self.checkboxGroup)
        self.layout.addSpacing(5)
        self.layout.addWidget(self.outputGroup)
        self.layout.addSpacing(5)
        self.layout.addWidget(self.runButton)

    def createGroup(self, title, layout):
        group = QGroupBox(title)
        group.setLayout(layout)
        return group

    def createPathWidgets(self, label, slot):
        layout = QHBoxLayout()
        label = QLabel(label)
        entry = QLineEdit()
        entry.setFixedWidth(1050)
        button = QPushButton("Browse")
        button.clicked.connect(slot)
        layout.addWidget(label)
        layout.addWidget(entry)
        layout.addWidget(button)
        self.pathLayout.addLayout(layout)
        return entry

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

    def closeEvent(self, event):
        saveSettings(self)
        event.accept()

    def openSettings(self):
        self.settingsWidget = QWidget()
        settingsLayout = QVBoxLayout()
        settingsLabel = QLabel("Settings go here")
        settingsLayout.addWidget(settingsLabel)
        backButton = QPushButton("Back")
        backButton.clicked.connect(self.goBack)
        settingsLayout.addWidget(backButton)
        self.settingsWidget.setLayout(settingsLayout)
        self.setCentralWidget(self.settingsWidget)

    def goBack(self):
        self.setCentralWidget(self.centralWidget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())