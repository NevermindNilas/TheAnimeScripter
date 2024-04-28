import os
import sys

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
from PyQt6.QtCore import Qt
from src.uiLogic import (
    darkUiStyleSheet,
    lightUiStyleSheet,
    runCommand,
    StreamToTextEdit,
    loadSettings,
    saveSettings,
    fadeIn,
    dropdownsLabels,
)

import logging

logging.basicConfig(filename='gui.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

TITLE = "The Anime Scripter - 1.6.0 (Alpha)"

if getattr(sys, "frozen", False):
    mainPath = os.path.dirname(sys.executable)
else:
    mainPath = os.path.dirname(os.path.abspath(__file__))

"""
from pypresence import Presence
self.clientID = "1213461768785891388"
self.RPC = Presence(self.clientID)
try:
    self.RPC.connect()
except ConnectionRefusedError:
    print("Could not connect to Discord. Is Discord running?")
self.timer.timeout.connect(self.updatePresence)
"""

class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(1280, 720)
        

        self.setStyleSheet(darkUiStyleSheet())
        self.stackedWidget = QStackedWidget()
        self.centralWidget = QWidget()
        self.stackedWidget.addWidget(self.centralWidget)
        self.setCentralWidget(self.stackedWidget)

        self.createLayouts()
        self.createWidgets()

        self.settingsFile = os.path.join(os.getcwd(), "settings.json")
        loadSettings(self, self.settingsFile)
        fadeIn(self, self.centralWidget, 500)

        dropdowns = [
            ("Interpolate Method:", "Interpolation"),
            ("Upscale Method:", "Upscaling"),
            ("Denoise Method:", "Denoise"),
            ("Dedup Method:", "Dedup"),
            ("Depth Method:", "Depth"),
            ("Encode Method:", "Encode"),
        ]

        self.dropdowns = {}
        for label, method in dropdowns:
            dropdownLayout, dropdown = self.createLabeledDropdown(
                label, dropdownsLabels(method)
            )
            setattr(self, f"{method.lower()}MethodDropdown", dropdown)
            self.dropdowns[method.lower()] = dropdownLayout

        checkboxes = [
            ("Keep Audio", "Enable or disable audio in the output file."),
            (
                "Benchmark Mode",
                "Benchmark mode will disable encoding and only monitor the performance of the script without the creation of an output file.",
            ),
        ]

        self.checkboxes = {}
        for text, help_text in checkboxes:
            checkbox = QCheckBox(text)
            checkbox.setToolTip(help_text)
            setattr(self, f"{text.replace(' ', '').lower()}Checkbox", checkbox)
            self.checkboxes[text.replace(" ", "").lower()] = checkbox

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
            "Sharpen",
            "Segment",
            "Depth",
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

        for label, defaultValue, maxValue in inputFields:
            layout, entry = self.createInputField(label, defaultValue, maxValue)
            self.inputFieldsLayout.addLayout(layout)
            if label == "Resize Factor:":
                self.resizeFactorEntry = entry
            elif label == "Interpolate Factor:":
                self.interpolateFactorEntry = entry

            elif label == "Upscale Factor:":
                self.upscaleFactorEntry = entry

            elif label == "Number of Threads:":
                self.numThreadsEntry = entry

        self.checkboxInputLayout.addLayout(self.checkboxLayout)
        self.checkboxInputLayout.addLayout(self.inputFieldsLayout)

        self.pathGroup = self.createGroup("Paths", self.pathLayout)
        self.checkboxGroup = self.createGroup("Options", self.checkboxInputLayout)
        self.outputGroup = self.createGroup("Log", self.outputLayout)

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
        saveSettings(self, self.settingsFile)
        runCommand(self, mainPath)

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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        label = QLabel(label)
        entry = QLineEdit()
        entry.setText(str(defaultValue))
        entry.setValidator(QIntValidator(0, maxValue))
        entry.setFixedWidth(30)
        entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(entry)
        layout.addStretch(0)
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

    def closeEvent(self, event):
        saveSettings(self, self.settingsFile)
        event.accept()

    def toggleTheme(self):
        if self.styleSheet() == darkUiStyleSheet():
            self.setStyleSheet(lightUiStyleSheet())
        else:
            self.setStyleSheet(darkUiStyleSheet())

    def openSettingsPanel(self):
        loadSettings(self, self.settingsFile)
        self.settingsWidget = QWidget()
        settingsLayout = QVBoxLayout()

        mainSettings = QVBoxLayout()
        mainSettings.setContentsMargins(10, 10, 10, 10)
        mainSettings.setSpacing(10)
        mainSettings.setAlignment(Qt.AlignmentFlag.AlignTop)
        mainSettingsGroup = self.createGroup("Main Settings", mainSettings)

        dropdowns = [
            ("Interpolate Method:", "Interpolation"),
            ("Upscale Method:", "Upscaling"),
            ("Denoise Method:", "Denoise"),
            ("Dedup Method:", "Dedup"),
            ("Depth Method:", "Depth"),
            ("Encode Method:", "Encode"),
        ]

        for label, method in dropdowns:
            dropdownLayout, dropdown = self.createLabeledDropdown(
                label, dropdownsLabels(method)
            )
            setattr(self, f"{method.lower()}MethodDropdown", dropdown)
            mainSettings.addLayout(dropdownLayout)

        self.encodeParamsLabel = QLabel()
        mainSettings.addWidget(self.encodeParamsLabel)
        self.updateEncodeParamsLabel(self.encodeMethodDropdown.currentText())
        self.encodeMethodDropdown.currentTextChanged.connect(
            self.updateEncodeParamsLabel
        )

        noteLabel = QLabel("NOTE: If this page hasn't updated but you see a \"Settings saved successfully\" in the log panel then it's fine. The settings have been saved.")
        
        # Add the note label to the settings layout
        mainSettings.addWidget(noteLabel)
        settingsLayout.addWidget(mainSettingsGroup)

        extraSettings = QVBoxLayout()
        extraSettings.setContentsMargins(10, 10, 10, 10)
        extraSettings.setSpacing(10)
        extraSettings.setAlignment(Qt.AlignmentFlag.AlignTop)
        extraSettingsGroup = self.createGroup("Extra Settings", extraSettings)

        checkboxes = [
            ("Keep Audio", "Enable or disable audio in the output file."),
            (
                "Benchmark Mode",
                "Benchmark mode will disable encoding and only monitor the performance of the script without the creation of an output file.",
            ),
        ]

        for text, help_text in checkboxes:
            checkbox = QCheckBox(text)
            checkbox.setToolTip(help_text)
            setattr(self, f"{text.replace(' ', '').lower()}Checkbox", checkbox)
            extraSettings.addWidget(checkbox)

        settingsLayout.addWidget(extraSettingsGroup)

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

    def updateEncodeParamsLabel(self, text):
        encodeDict = dropdownsLabels("Encode")
        if text in encodeDict:
            _, params = encodeDict[text]
            self.encodeParamsLabel.setText(params)
        else:
            self.encodeParamsLabel.setText("")

    def goBack(self):
        saveSettings(self, self.settingsFile)
        self.stackedWidget.removeWidget(self.settingsWidget)
        self.stackedWidget.setCurrentWidget(self.centralWidget)

        fadeIn(self, self.centralWidget, 300)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())
