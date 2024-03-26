# I hate making GUIs with a passion

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
    QComboBox,
)

from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import QTimer, Qt
from src.uiLogic import (
    darkUiStyleSheet,
    lightUiStyleSheet,
    runCommand,
    StreamToTextEdit,
    loadSettings,
    saveSettings,
    updatePresence,
    fadeIn,
)

import os
import sys
import time

# from pypresence import Presence
from main import scriptVersion

TITLE = f"The Anime Scripter - {scriptVersion} (Alpha)"


class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(1280, 720)

        """
        self.clientID = "1213461768785891388"
        self.RPC = Presence(self.clientID)
        try:
            self.RPC.connect()
        except ConnectionRefusedError:
            print("Could not connect to Discord. Is Discord running?")
        """

        self.start_time = int(time.time())
        self.timer = QTimer()
        # self.timer.timeout.connect(self.updatePresence)
        self.timer.start(1000)

        self.setStyleSheet(darkUiStyleSheet())

        self.stackedWidget = QStackedWidget()
        self.centralWidget = QWidget()
        self.stackedWidget.addWidget(self.centralWidget)
        self.setCentralWidget(self.stackedWidget)

        self.createLayouts()
        self.createWidgets()

        self.settingsFile = os.path.join(os.getcwd(), "settings.json")
        loadSettings(self)
        fadeIn(self, self.centralWidget, 500)

    def createLayouts(self):
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.pathLayout = QVBoxLayout()
        self.checkboxLayout = QVBoxLayout()
        self.outputLayout = QVBoxLayout()

    def createWidgets(self):
        self.checkboxInputLayout = QHBoxLayout()

        for option in [
            "Resize",
            "Dedup",
            "Denoise",
            "Upscale",
            "Interpolate",
            "Segment",
            "Depth",
            "Sharpen",
        ]:
            self.createCheckbox(option)

        inputFields = [
            ("Interpolate Factor:", 2, 100),
            ("Upscale Factor:", 2, 4),
            ("Resize Factor:", 1, 4),
            (
                "Number of Threads:",
                1,
                4,
            ),  # Experimental feature, needs more work but it's there
        ]

        self.inputFieldsLayout = QVBoxLayout()

        upscaleMethods = ["ShuffleCugan", "Cugan", "RealESRGAN", "Span", "OmniSR", "ShuffleCugan-NCNN", "Cugan-NCNN", "RealESRGAN-NCNN", "Span-NCNN"]  # Replace with actual methods
        interpolateMethods = ["Rife4.16-Lite", "Rife4.15", "Rife4.14", "Rife4.6", "Rife4.16-Lite-NCNN", "Rife4.15-NCNN", "Rife4.14-NCNN", "Rife4.6-NCNN", "GMFSS"]  # Replace with actual methods
        denoiseMethods = ["DPIR", "SCUNet", "NAFNet", "Span"]  # Replace with actual methods

        for label, defaultValue, maxValue in inputFields:
            layout, entry = self.createInputField(label, defaultValue, maxValue)
            self.inputFieldsLayout.addLayout(layout)
            if label == "Resize Factor:":
                self.resizeFactorEntry = entry
            elif label == "Interpolate Factor:":
                self.interpolateFactorEntry = entry
                dropdownLayout, dropdown = self.createLabeledDropdown("Interpolate Method:", interpolateMethods)
                self.inputFieldsLayout.addLayout(dropdownLayout)
                self.interpolateMethodDropdown = dropdown
            elif label == "Upscale Factor:":
                self.upscaleFactorEntry = entry
                dropdownLayout, dropdown = self.createLabeledDropdown("Upscale Method:", upscaleMethods)
                self.inputFieldsLayout.addLayout(dropdownLayout)
                self.upscaleMethodDropdown = dropdown
            elif label == "Number of Threads:":
                self.numThreadsEntry = entry
                self.inputFieldsLayout.addLayout(dropdownLayout)
                self.denoiseMethodDropdown = dropdown

        self.checkboxInputLayout.addLayout(self.checkboxLayout)
        self.checkboxInputLayout.addLayout(self.inputFieldsLayout)

        self.pathGroup = self.createGroup("Paths", self.pathLayout)
        self.checkboxGroup = self.createGroup("Options", self.checkboxInputLayout)
        self.outputGroup = self.createGroup("Terminal", self.outputLayout)

        self.inputEntry = self.createPathWidgets("Input Path:", self.browseInput)
        self.outputEntry = self.createPathWidgets("Output Path:", self.browseOutput)

        self.outputWindow = QTextEdit()
        self.outputWindow.setReadOnly(True)
        self.outputLayout.addWidget(self.outputWindow)

        sys.stdout = StreamToTextEdit(self.outputWindow)
        sys.stderr = StreamToTextEdit(self.outputWindow)

        self.runButton = self.createButton("Run", self.runButtonOnClick)
        self.settingsButton = self.createButton("Settings", self.openSettingsPanel)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.runButton)
        self.buttonLayout.addWidget(self.settingsButton)

        self.addWidgetsToLayout(
            self.layout, [self.pathGroup, self.checkboxGroup, self.outputGroup], 5
        )
        self.layout.addLayout(self.buttonLayout)
    
    def createLabeledDropdown(self, label, options):
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label = QLabel(label)
        dropdown = QComboBox()
        dropdown.addItems(options)
        layout.addWidget(label)
        layout.addWidget(dropdown)
        layout.addStretch(1)
        return layout, dropdown
    
    def createGroup(self, title, layout):
        group = QGroupBox(title)
        group.setLayout(layout)
        return group

    def runButtonOnClick(self):
        saveSettings(self)
        runCommand(self, TITLE)
    
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
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label = QLabel(label)
        entry = QLineEdit()
        entry.setText(str(defaultValue))
        entry.setValidator(QIntValidator(0, maxValue))
        entry.setFixedWidth(30)
        entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(entry)
        layout.addStretch(1)  
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
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi);;All Files (*)",
        )
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

    def toggleTheme(self):
        if self.styleSheet() == darkUiStyleSheet():
            self.setStyleSheet(lightUiStyleSheet())
        else:
            self.setStyleSheet(darkUiStyleSheet())
        fadeIn(self, self.centralWidget, 300)

    def openSettingsPanel(self):
        self.settingsWidget = QWidget()
        settingsLayout = QVBoxLayout()

        upscaleSettingsGroup = self.createGroup("Main Settings", QVBoxLayout())
        extraGroup = self.createGroup("Extra", QVBoxLayout())

        settingsLayout.addWidget(upscaleSettingsGroup)
        settingsLayout.addWidget(extraGroup)

        buttonsLayout = QHBoxLayout()

        themeButton = self.createButton("Toggle Day / Night Theme", self.toggleTheme)
        backButton = self.createButton("Back", self.goBack)

        buttonsLayout.addWidget(themeButton)
        buttonsLayout.addWidget(backButton)

        settingsLayout.addLayout(buttonsLayout)

        self.settingsWidget.setLayout(settingsLayout)
        self.stackedWidget.addWidget(self.settingsWidget)
        self.stackedWidget.setCurrentWidget(self.settingsWidget)

        fadeIn(self, self.settingsWidget, 300)

    def goBack(self):
        self.stackedWidget.removeWidget(self.settingsWidget)
        self.stackedWidget.setCurrentWidget(self.centralWidget)

        fadeIn(self, self.centralWidget, 300)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())
