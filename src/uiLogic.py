import os
import json
import threading
import logging

from PyQt6.QtWidgets import QCheckBox, QGraphicsOpacityEffect
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve


def Style() -> str:
    """
    Returns the stylesheet for the UI,
    These are Global preset styles meant to be used in the entire application
    """
    return """
        * {
            font-family: Segoe UI;
            font-size: 13px;
            color: #FFFFFF;
        
        }
        QMainWindow {
            background-color: rgba(0, 0, 0, 0);
            border-radius: 5px;
        }
        
        QWidget {
            background-color: rgba(0, 0, 0, 0);
        }

        QPushButton {
            background-color: rgba(60, 60, 60, 0.5);
            color: #FFFFFF;
            border-radius: 5px;
            min-height: 25px;
        }

        QPushButton:hover {
            background-color: rgba(60, 60, 60, 0.7);
        }

        QLineEdit {
            background-color: rgba(60, 60, 60, 0.5);
            color: #FFFFFF;
            border-radius: 5px;
            min-height: 25px;
        }

        QComboBox {
            min-width: 100px;
            background-color: rgba(60, 60, 60, 0.5);
            color: #FFFFFF;
            border-radius: 5px;
            min-height: 25px;
        }

        QComboBox QAbstractItemView {
            border: 2px solid darkgray; /* Border for the dropdown list */
            selection-background-color: #606060; /* Background color of hovered item */
            color: #FFFFFF; /* Text color of items */
            background-color: rgba(60, 60, 60, 0.8); /* Background of the dropdown */
            border-radius: 5px; /* Rounded corners for the dropdown list */
        }


        QCheckBox {
            color: #FFFFFF;
        }

        QTextEdit {
            background-color: rgba(60, 60, 60, 0.5);
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 5px;
        }

        QGroupBox {
            border: 1px solid #4A4A4A;
            border-radius: 5px;
            margin-top: 2ex;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: -7px 5px 0 5px;
        }
    """


def runCommand(self, mainPath, settingsFile, run=False) -> None:
    try:
        mainExePath = os.path.join(mainPath, "main.exe")
        if not os.path.isfile(mainExePath):
            try:
                mainExePath = os.path.join(mainPath, "main.py")
                command = ["python", mainExePath]
            except FileNotFoundError:
                self.outputWindow.append("main.exe nor main.py not found")
                return
        else:
            command = [mainExePath]

        loadSettingsFile = json.load(open(settingsFile))
        DontcloseTerminal = loadSettingsFile.get("close_terminal")

        for option in loadSettingsFile:
            loweredOption = option.lower()
            loweredOptionValue = str(loadSettingsFile[option])

            if loweredOption == "output":
                if loweredOptionValue == "":
                    continue
            elif loweredOption == "input":
                if loweredOptionValue == "":
                    print("Input path is empty")
                    return
            else:
                loweredOptionValue = loweredOptionValue.lower()

            if loweredOptionValue == "false":
                continue

            if loweredOption == "close_terminal":
                continue

            if loweredOption in ["input", "output"]:
                command.append(f'--{loweredOption} "{loweredOptionValue}"')
            else:
                if loweredOptionValue in ["true", "True"]:
                    command.append(f"--{loweredOption}")
                else:
                    command.append(f"--{loweredOption} {loweredOptionValue}")

        command = " ".join(command)
        print(command)
        if run:

            def runCommandInTerminal(command):
                # Check if Windows Terminal (wt) is available
                WindowsTerminal = os.system("where wt >nul 2>&1")
                if DontcloseTerminal:
                    terminalCommand = (
                        f"start wt cmd /k {command}"
                        if WindowsTerminal == 0
                        else f"start cmd /k {command}"
                    )
                else:
                    terminalCommand = (
                        f"start wt cmd /c {command}"
                        if WindowsTerminal == 0
                        else f"start cmd /c {command}"
                    )
                try:
                    os.system(terminalCommand)
                except Exception as e:
                    print(f"An error occurred while running the command: {e}")

            threading.Thread(
                target=runCommandInTerminal, args=(command,), daemon=True
            ).start()

    except Exception as e:
        print(f"An error occurred while running the command, {e}")
        logging.error(f"An error occurred while running, {e}")


class StreamToTextEdit:
    def __init__(self, textEdit):
        self.textEdit = textEdit

    def write(self, text):
        self.textEdit.append(text)

    def flush(self):
        pass


def loadSettings(self, settingsFile):
    if os.path.exists(settingsFile):
        try:
            with open(settingsFile, "r") as file:
                settings = json.load(file)
            self.inputEntry.setText(settings.get("input", ""))
            self.outputEntry.setText(settings.get("output", ""))
            self.resizeFactorEntry.setText(settings.get("resize_factor", ""))
            self.interpolateFactorEntry.setText(settings.get("interpolate_factor", ""))
            self.upscaleFactorEntry.setText(settings.get("upscale_factor", ""))

            # Dropdowns
            self.interpolationMethodDropdown.setCurrentIndex(
                self.interpolationMethodDropdown.findText(
                    settings.get("interpolate_method", "")
                )
            )
            self.upscalingMethodDropdown.setCurrentIndex(
                self.upscalingMethodDropdown.findText(
                    settings.get("upscale_method", "")
                )
            )
            self.denoiseMethodDropdown.setCurrentIndex(
                self.denoiseMethodDropdown.findText(settings.get("denoise_method", ""))
            )
            self.dedupMethodDropdown.setCurrentIndex(
                self.dedupMethodDropdown.findText(settings.get("dedup_method", ""))
            )
            self.depthMethodDropdown.setCurrentIndex(
                self.depthMethodDropdown.findText(settings.get("depth_method", ""))
            )
            self.encodeMethodDropdown.setCurrentIndex(
                self.encodeMethodDropdown.findText(settings.get("encode_method", ""))
            )
            self.resizeMethodDropdown.setCurrentIndex(
                self.resizeMethodDropdown.findText(settings.get("resize_method", ""))
            )

            # Checkboxes
            for i in range(self.checkboxLayout.count()):
                checkbox = self.checkboxLayout.itemAt(i).widget()
                if isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(settings.get(checkbox.text(), False))

            # Additional stuff
            self.benchmarkmodeCheckbox.setChecked(settings.get("benchmark", False))
            self.scenechangedetectionCheckbox.setChecked(
                settings.get("scenechange", False)
            )
            self.rifeensembleCheckbox.setChecked(settings.get("ensemble", False))
            self.donotcloseterminalonfinishCheckbox.setChecked(
                settings.get("close_terminal", False)
            )

        except Exception as e:
            self.outputWindow.append(f"An error occurred while loading settings, {e}")
            logging.error(f"An error occurred while loading settings, {e}")


def saveSettings(self, settingsFile, printSave=True):
    try:
        settings = {
            "input": self.inputEntry.text(),
            "output": self.outputEntry.text(),
            "resize_factor": self.resizeFactorEntry.text(),
            "interpolate_factor": self.interpolateFactorEntry.text(),
            "upscale_factor": self.upscaleFactorEntry.text(),
            "interpolate_method": self.interpolationMethodDropdown.currentText(),
            "upscale_method": self.upscalingMethodDropdown.currentText(),
            "denoise_method": self.denoiseMethodDropdown.currentText(),
            "dedup_method": self.dedupMethodDropdown.currentText(),
            "depth_method": self.depthMethodDropdown.currentText(),
            "encode_method": self.encodeMethodDropdown.currentText(),
            "resize_method": self.resizeMethodDropdown.currentText(),
            "benchmark": self.benchmarkmodeCheckbox.isChecked(),
            "scenechange": self.scenechangedetectionCheckbox.isChecked(),
            "ensemble": self.rifeensembleCheckbox.isChecked(),
            "close_terminal": self.donotcloseterminalonfinishCheckbox.isChecked(),
            #"save_shortcut": self.answer if hasattr(self, "answer") else None,
        }
        for i in range(self.checkboxLayout.count()):
            checkbox = self.checkboxLayout.itemAt(i).widget()
            if isinstance(checkbox, QCheckBox):
                settings[checkbox.text()] = checkbox.isChecked()
        with open(settingsFile, "w") as file:
            json.dump(settings, file, indent=4)

        if printSave:
            self.outputWindow.append("Settings saved successfully")

    except Exception as e:
        self.outputWindow.append(f"An error occurred while saving settings, {e}")


def fadeIn(self, widget, duration=500):
    opacity_effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(opacity_effect)
    self.animation = QPropertyAnimation(opacity_effect, b"opacity")
    self.animation.setDuration(duration)
    self.animation.setStartValue(0)
    self.animation.setEndValue(1)
    self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
    self.animation.start()


def dropdownsLabels(method):
    match method:
        case "Upscaling":
            return [
                "ShuffleCugan",
                "Cugan",
                "Span",
                "Compact",
                "UltraCompact",
                "SuperUltraCompact",
                "ShuffleCugan-TensorRT",
                "Span-TensorRT",
                "Compact-TensorRT",
                "UltraCompact-TensorRT",
                "SuperUltraCompact-TensorRT",
                "Cugan-DirectML",
                "Span-DirectML",
                "Compact-DirectML",
                "UltraCompact-DirectML",
                "SuperUltraCompact-DirectML",
                "ShuffleCugan-NCNN",
                "Span-NCNN",
            ]
        case "Interpolation":
            return [
                "Rife4.18",
                "Rife4.17",
                "Rife4.16-Lite",
                "Rife4.15-Lite",
                "Rife4.15",
                "Rife4.6",
                "Rife4.18-TensorRT",
                "Rife4.17-TensorRT",
                "Rife4.15-TensorRT",
                "Rife4.15-Lite-TensorRT",
                "Rife4.6-TensorRT",
                "Rife4.17-NCNN",
                "Rife4.17-NCNN",
                "Rife4.16-Lite-NCNN",
                "Rife4.15-NCNN",
                "Rife4.15-Lite-NCNN",
                "Rife4.6-NCNN",
                "GMFSS",
            ]
        case "Denoise":
            return ["DPIR", "SCUNet", "NAFNet"]
        case "Dedup":
            return ["SSIM", "MSE", "SSIM-CUDA"]
        case "Depth":
            return [
                "Small_v2",
                "Base_v2",
                "Large_v2",
                "Small_v2-TensorRT",
                "Base_v2-TensorRT",
                "Large_v2-TensorRT",
            ]
        case "Encode":
            return {
                "x264": ("x264", "-c:v libx264 -preset fast -crf 15"),
                "x264_animation": (
                    "x264_animation",
                    "-c:v libx264 -preset fast -tune animation -crf 15",
                ),
                "x264_10bit": (
                    "x264_10bit",
                    "-c:v libx264 -preset fast -profile:v high10 -crf 15",
                ),
                "x265": ("x265", "-c:v libx265 -preset fast -crf 15"),
                "x265_10bit": (
                    "x265_10bit",
                    "-c:v libx265 -preset fast -profile:v main10 -crf 15",
                ),
                "nvenc_h264": ("nvenc_h264", "-c:v h264_nvenc -preset p1 -cq 15"),
                "nvenc_h265": ("nvenc_h265", "-c:v hevc_nvenc -preset p1 -cq 15"),
                "nvenc_h265_10bit": (
                    "nvenc_h265_10bit",
                    "-c:v hevc_nvenc -preset p1 -profile:v main10 -cq 15",
                ),
                "qsv_h264": (
                    "qsv_h264",
                    "-c:v h264_qsv -preset veryfast -global_quality 15",
                ),
                "qsv_h265": (
                    "qsv_h265",
                    "-c:v hevc_qsv -preset veryfast -global_quality 15",
                ),
                "qsv_h265_10bit": (
                    "qsv_h265_10bit",
                    "-c:v hevc_qsv -preset veryfast -profile:v main10 -global_quality 15",
                ),
                "nvenc_av1": ("nvenc_av1", "-c:v av1_nvenc -preset p1 -cq 15"),
                "av1": ("av1", "-c:v libsvtav1 -preset 8 -crf 15"),
                "h264_amf": ("h264_amf", "-c:v h264_amf -quality speed -rc cqp -qp 15"),
                "hevc_amf": ("hevc_amf", "-c:v hevc_amf -quality speed -rc cqp -qp 15"),
                "hevc_amf_10bit": (
                    "hevc_amf_10bit",
                    "-c:v hevc_amf -quality speed -rc cqp -qp 15 -profile:v main10",
                ),
                "prores": ("prores", "-c:v prores_ks -profile:v 4 -qscale:v 15"),
            }
        case "Resize":
            return [
                "Bilinear",
                "Bicubic",
                "Lanczos",
                "Nearest",
                "Spline",
                "Spline16",
                "Spline36",
            ]
