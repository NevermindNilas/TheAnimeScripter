from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtCore import QProcess

import os
import json

def uiStyleSheet() -> str:
    """
    Returns the stylesheet for the UI,
    These are Global preset styles meant to be used in the entire application
    """
    return """
        QMainWindow {
            background-color: #2D2D2D;
        }
        
        QWidget {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }

        QPushButton {
            background-color: #3F3F3F;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
        }

        QPushButton:hover {
            background-color: #4A90E2;
        }

        QLineEdit {
            background-color: #3F3F3F;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 5px;
        }

        QCheckBox {
            color: #FFFFFF;
            padding: 5px;
        }

        QTextEdit {
            background-color: #3F3F3F;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 5px;
        }

        QGroupBox {
            border: 1px solid #4A4A4A;
            border-radius: 5px;
            margin-top: 1ex;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: -7px 5px 0 5px;
        }
    """


def runCommand(self, TITLE) -> None:
    self.RPC.update(
        details="Processing",
        start=self.start_time,
        large_image="icon",
        small_image="icon",
        large_text=TITLE,
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
    else:
        self.outputWindow.append("Output directory was not selected, using default")
        
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


class StreamToTextEdit:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message):
        self.text_edit.append(message)

    def flush(self):
        pass


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
        "output_path": self.outputEntry.text(),
    }
    for i in range(self.checkboxLayout.count()):
        checkbox = self.checkboxLayout.itemAt(i).widget()
        if isinstance(checkbox, QCheckBox):
            settings[checkbox.text()] = checkbox.isChecked()
    with open(self.settingsFile, "w") as file:
        json.dump(settings, file, indent=4)

def updatePresence(RPC, start_time, TITLE):
    RPC.update(
        details="Idle",
        start=start_time,
        large_image="icon",
        small_image="icon",
        large_text=TITLE,
        small_text="Idle",
    )