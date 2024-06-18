import subprocess
import os
import json
import threading
import logging
import time

from PyQt6.QtWidgets import QCheckBox, QGraphicsOpacityEffect
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve

def Style() -> str:
    """
    Returns the stylesheet for the UI,
    These are Global preset styles meant to be used in the entire application
    """
    return """
        * {
            font-family: Tahoma;
            font-size: 12px;
            color: #FFFFFF;
        
        }
        QMainWindow {
            background-color: rgba(0, 0, 0, 0);
            border-radius: 10px;
        }
        
        QWidget {
            background-color: rgba(0, 0, 0, 0);
        }

        QPushButton {
            background-color: rgba(60, 60, 60, 0.5);
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
        }

        QPushButton:hover {
            background-color: rgba(60, 60, 60, 0.7);
        }

        QLineEdit {
            background-color: rgba(60, 60, 60, 0.5);
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
            background-color: rgba(60, 60, 60, 0.5);
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

def runCommand(self, mainPath, settingsFile) -> None:
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

        for option in loadSettingsFile:
            loweredOption = option.lower()

            loweredOptionValue = str(loadSettingsFile[option])
            if "http" not in loweredOptionValue:
                loweredOptionValue = loweredOptionValue.lower()

            if loweredOptionValue == "false":
                continue

            if loweredOption == "output" and loweredOptionValue == "":
                continue

            if loweredOption in ["input", "output"]:
                command.append(f"--{loweredOption} \"{loweredOptionValue}\"")
            else:
                if loweredOptionValue in ["true", "True"]:
                    command.append(f"--{loweredOption}")
                else:
                    command.append(f"--{loweredOption} {loweredOptionValue}")

        command = " ".join(command)
        print(command)
        def runCommandInTerminal(command):
            os.system(f'start cmd /c "{command} & exit"')

        threading.Thread(target=runCommandInTerminal, args=(command,), daemon=True).start()


    except Exception as e:
        self.outputWindow.append(f"An error occurred while running the command, {e}")
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
            logging.error(f"An error occurred while loading settings, {e}")


def saveSettings(self, settingsFile):
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
        }
        for i in range(self.checkboxLayout.count()):
            checkbox = self.checkboxLayout.itemAt(i).widget()
            if isinstance(checkbox, QCheckBox):
                settings[checkbox.text()] = checkbox.isChecked()
        with open(settingsFile, "w") as file:
            json.dump(settings, file, indent=4)
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
                "Rife4.17",
                "Rife4.17-Lite",
                "Rife4.16-Lite",
                "Rife4.15-Lite",
                "Rife4.15",
                "Rife4.6",
                "Rife4.17-TensorRT",
                "Rife4.15-TensorRT",
                "Rife4.15-Lite-TensorRT",
                "Rife4.6-TensorRT",
                "Rife4.16-Lite-NCNN",
                "Rife4.15-NCNN",
                "Rife4.15-Lite-NCNN",
                "Rife4.6-NCNN",
                "GMFSS",
            ]
        case "Denoise":
            return ["DPIR", "SCUNet", "NAFNet", "Span"]
        case "Dedup":
            return ["SSIM", "MSE", "SSIM-CUDA"]
        case "Depth":
            return ["Small", "Base", "Large", "Small-TensorRT", "Base-TensorRT", "Large-TensorRT", "Small-DirectML", "Base-DirectML", "Large-DirectML"]
        case "Encode":
            return {
                "x264": ("x264", "-c:v libx264 -preset fast -crf 17"),
                "x264_animation": (
                    "x264_animation",
                    "-c:v libx264 -preset fast -tune animation -crf 17",
                ),
                "x264_10bit": ("x264_10bit", "-c:v libx264 -preset fast -profile:v high10 -crf 17"),
                "x265": ("x265", "-c:v libx265 -preset fast -crf 17"),
                "x265_10bit": ("x265_10bit", "-c:v libx265 -preset fast -profile:v main10 -crf 17"),
                "nvenc_h264": ("nvenc_h264", "-c:v h264_nvenc -preset p1 -cq 17"),
                "nvenc_h265": ("nvenc_h265", "-c:v hevc_nvenc -preset p1 -cq 17"),
                "nvenc_h265_10bit": ("nvenc_h265_10bit", "-c:v hevc_nvenc -preset p1 -profile:v main10 -cq 17"),
                "qsv_h264": (
                    "qsv_h264",
                    "-c:v h264_qsv -preset veryfast -global_quality 17",
                ),
                "qsv_h265": (
                    "qsv_h265",
                    "-c:v hevc_qsv -preset veryfast -global_quality 17",
                ),
                "qsv_h265_10bit": (
                    "qsv_h265_10bit",
                    "-c:v hevc_qsv -preset veryfast -profile:v main10 -global_quality 17",
                ),
                "nvenc_av1": ("nvenc_av1", "-c:v av1_nvenc -preset p1 -cq 17"),
                "av1": ("av1", "-c:v libsvtav1 -preset 8 -crf 17"),
                "h264_amf": ("h264_amf", "-c:v h264_amf -quality speed -rc cqp -qp 17"),
                "hevc_amf": ("hevc_amf", "-c:v hevc_amf -quality speed -rc cqp -qp 17"),
                "hevc_amf_10bit": ("hevc_amf_10bit", "-c:v hevc_amf -quality speed -rc cqp -qp 17 -profile:v main10"),
                "prores": ("prores", "-c:v prores_ks -profile:v 4 -qscale:v 17"),
            }
        case "Resize":
            return ["Bilinear", "Bicubic", "Lanczos", "Nearest", "Spline", "Spline16", "Spline36"]
