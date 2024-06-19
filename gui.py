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
    Style,
    runCommand,
    StreamToTextEdit,
    loadSettings,
    saveSettings,
    fadeIn,
    dropdownsLabels,
)
from BlurWindow.blurWindow import GlobalBlur

import logging

logging.basicConfig(
    filename="gui.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

TITLE = "The Anime Scripter - 1.8.6 (Alpha)"
W, H = 1280, 720

if getattr(sys, "frozen", False):
    mainPath = os.path.dirname(sys.executable)
else:
    mainPath = os.path.dirname(os.path.abspath(__file__))


class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(W, H)

        GlobalBlur(self.winId(), Acrylic=True)

        self.setStyleSheet(Style())
        self.centralWidget = QWidget()
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.centralWidget)
        self.setCentralWidget(self.stackedWidget)

        self.createLayouts()
        self.createWidgets()

        self.settingsFile = os.path.join(os.getcwd(), "settings.json")
        loadSettings(self, self.settingsFile)
        fadeIn(self, self.centralWidget, 500)

    def createLayouts(self):
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.pathLayout = QVBoxLayout()
        self.checkboxLayout = QVBoxLayout()
        self.outputLayout = QVBoxLayout()

    def createWidgets(self):
        self.inputEntry = self.createPathWidgets("Input Path:", self.browseInput)
        self.outputEntry = self.createPathWidgets("Output Path:", self.browseOutput)

        self.OptionLayout = QHBoxLayout()
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
        ]

        self.inputFieldsLayout = QVBoxLayout()
        self.inputFieldsLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for label, defaultValue, maxValue in inputFields:
            layout, entry = self.createInputField(label, defaultValue, maxValue)
            self.inputFieldsLayout.addLayout(layout)
            if label == "Resize Factor:":
                self.resizeFactorEntry = entry
            elif label == "Interpolate Factor:":
                self.interpolateFactorEntry = entry
            elif label == "Upscale Factor:":
                self.upscaleFactorEntry = entry

        self.mainSettings = QVBoxLayout()
        self.mainSettings.setSpacing(10)
        self.mainSettings.setAlignment(Qt.AlignmentFlag.AlignTop)

        dropdowns = [
            ("Resize Method:", "Resize"),
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
            self.mainSettings.addLayout(dropdownLayout)

        self.encodeParamsLabel = QLabel()
        self.mainSettings.addWidget(self.encodeParamsLabel)
        self.updateEncodeParamsLabel(self.encodeMethodDropdown.currentText())
        self.encodeMethodDropdown.currentTextChanged.connect(
            self.updateEncodeParamsLabel
        )

        self.extraSettings = QVBoxLayout()
        self.extraSettings.setAlignment(Qt.AlignmentFlag.AlignTop)

        checkboxes = [
            (
                "Benchmark Mode",
                "Benchmark mode will disable encoding and only monitor the performance of the script without the creation of an output file.",
            ),
            (
                "Scene Change Detection",
                "Enable or disable scene change detection, can impact performance but it will improve the quality of the output when interpolating.",
            ),
            (
                "Rife Ensemble",
                "Enable or disable RIFE ensemble, this can improve the quality of the interpolation at the cost of less performance.",
            ),
        ]

        for text, help_text in checkboxes:
            checkbox = QCheckBox(text)
            checkbox.setToolTip(help_text)
            setattr(self, f"{text.replace(' ', '').lower()}Checkbox", checkbox)
            self.extraSettings.addWidget(checkbox)

        self.OptionLayout.addLayout(self.checkboxLayout)
        self.OptionLayout.addLayout(self.inputFieldsLayout)
        self.OptionLayout.addLayout(self.mainSettings)
        self.OptionLayout.addLayout(self.extraSettings)
        self.OptionLayout.addStretch(1)
        self.OptionLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.pathGroup = self.createGroup("Paths", self.pathLayout, 100)
        self.checkboxGroup = self.createGroup("Options", self.OptionLayout)
        self.outputGroup = self.createGroup("Log", self.outputLayout, 250)

        self.outputWindow = QTextEdit()
        self.outputWindow.setReadOnly(True)
        self.outputLayout.addWidget(self.outputWindow)

        sys.stdout = StreamToTextEdit(self.outputWindow)
        sys.stderr = StreamToTextEdit(self.outputWindow)

        self.runButton = self.createButton("Run", self.runButtonOnClick)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.runButton)

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
        return layout, dropdown

    def createGroup(self, title, layout, maxHeight=None):
        group = QGroupBox(title)
        group.setLayout(layout)
        if maxHeight is not None:
            group.setMaximumHeight(maxHeight)
        return group

    def runButtonOnClick(self):
        saveSettings(self, self.settingsFile)
        runCommand(self, mainPath, self.settingsFile)

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
        entry.setFixedWidth(50)
        entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(entry)
        layout.addStretch(0)
        return layout, entry

    def createCheckbox(self, text):
        checkbox = QCheckBox(text)
        self.checkboxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
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
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*)",
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
