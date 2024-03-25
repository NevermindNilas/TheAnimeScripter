import subprocess
import os
import json

from PyQt6.QtWidgets import QCheckBox


def darkUiStyleSheet() -> str:
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


def lightUiStyleSheet() -> str:
    """
    Returns the stylesheet for the UI,
    These are Global preset styles meant to be used in the entire application
    """
    return """
        QMainWindow {
            background-color: #E9ECEC;
        }
        
        QWidget {
            background-color: #E9ECEC;
            color: #000000;
        }

        QPushButton {
            background-color: #B0BEC5;
            color: #000000;
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
        }

        QPushButton:hover {
            background-color: #78909C;
        }

        QLineEdit {
            background-color: #CFD8DC;
            color: #000000;
            border: none;
            border-radius: 5px;
            padding: 5px;
        }

        QCheckBox {
            color: #000000;
            padding: 5px;
        }

        QTextEdit {
            background-color: #CFD8DC;
            color: #000000;
            border: none;
            border-radius: 5px;
            padding: 5px;
        }

        QGroupBox {
            border: 1px solid #B0BEC5;
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
    """
    self.RPC.update(
        details="Processing",
        start=self.start_time,
        large_image="icon",
        small_image="icon",
        large_text=TITLE,
        small_text="Processing",
    )
    """
    mainExePath = os.path.join(os.path.dirname(__file__), "main.exe")
    if not os.path.isfile(mainExePath):
        try:
            mainExePath = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "main.py"
            )
            command = ["python", mainExePath]
        except FileNotFoundError:
            self.outputWindow.append("main.exe nor main.py not found")
            return
    else:
        command = [mainExePath]

    loadSettingsFile = json.load(open(self.settingsFile))

    for option in loadSettingsFile:
        loweredOption = option.lower()
        loweredOptionValue = str(loadSettingsFile[option]).lower()

        if loweredOption == "input_path":
            loweredOption = "input"

        if loweredOption == "output_path":
            loweredOption = "output"

        if loweredOptionValue == "true":
            loweredOptionValue = "1"
        elif loweredOptionValue == "false":
            loweredOptionValue = "0"

        command.append(f"--{loweredOption} {loweredOptionValue}")

    command = " ".join(command)
    print(command)

    subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # self.process.readyReadStandardOutput.connect(self.handleStdout)
    # self.process.readyReadStandardError.connect(self.handleStderr)
    # self.process.start(command[0], command[1:])
    # print(mainExePath)
    # self.process = QProcess()
    # self.process.readyReadStandardOutput.connect(self.handleStdout)
    # self.process.readyReadStandardError.connect(self.handleStderr)
    # self.process.start(command[0], command[1:])


class StreamToTextEdit:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message):
        self.text_edit.append(message)

    def flush(self):
        pass


def loadSettings(self):
    if os.path.exists(self.settingsFile):
        try:
            with open(self.settingsFile, "r") as file:
                settings = json.load(file)
            self.inputEntry.setText(settings.get("input_path", ""))
            self.outputEntry.setText(settings.get("output_path", ""))
            for i in range(self.checkboxLayout.count()):
                checkbox = self.checkboxLayout.itemAt(i).widget()
                if isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(settings.get(checkbox.text(), False))

        except Exception as e:
            print(
                self.outputWindow.append(
                    f"An error occurred while loading settings, {e}"
                )
            )


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
