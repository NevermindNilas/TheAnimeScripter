import subprocess
import os
import json

from PyQt6.QtWidgets import QCheckBox, QGraphicsOpacityEffect
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve


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

        if loweredOptionValue == "true":
            loweredOptionValue = "1"
        elif loweredOptionValue == "false":
            loweredOptionValue = "0"
        
        if loweredOption == "output" and loweredOptionValue == "":
            continue

        command.append(f"--{loweredOption} {loweredOptionValue}")

    command = " ".join(command)
    print(command)
    try:
        subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception as e:
        self.outputWindow.append(f"An error occurred while running the command, {e}")


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


def saveSettings(self):
    settings = {
        "input": self.inputEntry.text(),
        "output": self.outputEntry.text(),
        "resize_factor": self.resizeFactorEntry.text(),
        "interpolate_factor": self.interpolateFactorEntry.text(),
        "nt": self.numThreadsEntry.text(),
        "upscale_factor": self.upscaleFactorEntry.text(),
        "interpolate_method": self.interpolateMethodDropdown.currentText(),
        "upscale_method": self.upscaleMethodDropdown.currentText(),
        "denoise_method": self.denoiseMethodDropdown.currentText(),
        "dedup_method": self.dedupMethodDropdown.currentText(),
        "depth_method": self.depthMethodDropdown.currentText(),
        "encode_method": self.encodeMethodDropdown.currentText(),
        "audio": self.keepAudioCheckbox.isChecked(),
        "benchmark": self.benchmarkCheckbox.isChecked(),
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
                "RealESRGAN",
                "Span",
                "OmniSR",
                "ShuffleCugan-NCNN",
                "Cugan-NCNN",
                "RealESRGAN-NCNN",
                "Span-NCNN",
            ]
        case "Interpolation":
            return [
                "Rife4.16-Lite",
                "Rife4.15",
                "Rife4.14",
                "Rife4.6",
                "Rife4.16-Lite-NCNN",
                "Rife4.15-NCNN",
                "Rife4.14-NCNN",
                "Rife4.6-NCNN",
                "GMFSS",
            ]
        case "Denoise":
            return ["DPIR", "SCUNet", "NAFNet", "Span"]
        case "Dedup":
            return ["SSIM", "FFMPEG", "MSE"]
        case "Depth":
            return ["Small", "Base", "Large"]
        case "Encode":
            return {
                "x264": ("x264", "-c:v libx264 -preset fast -crf 16"),
                "x264_animation": ("x264_animation", "-c:v libx264 -preset fast -tune animation -crf 16"),
                "x265": ("x265", "-c:v libx265 -preset fast -crf 16"),
                "nvenc_h264": ("nvenc_h264", "-c:v h264_nvenc -preset p1 -cq 16"),
                "nvenc_h265": ("nvenc_h265", "-c:v hevc_nvenc -preset p1 -cq 16"),
                "qsv_h264": ("qsv_h264", "-c:v h264_qsv -preset veryfast -global_quality 16"),
                "qsv_h265": ("qsv_h265", "-c:v hevc_qsv -preset veryfast -global_quality 16"),
                "nvenc_av1": ("nvenc_av1", "-c:v av1_nvenc -preset p1 -cq 16"),
                "av1": ("av1", "-c:v libsvtav1 -preset 8 -crf 16"),
                "h264_amf": ("h264_amf", "-c:v h264_amf -quality speed -rc cqp -qp 16"),
                "hevc_amf": ("hevc_amf", "-c:v hevc_amf -quality speed -rc cqp -qp 16"),
                "vp9": ("vp9", "-c:v libvpx-vp9 -crf 16"),
                "qsv_vp9": ("qsv_vp9", "-c:v vp9_qsv -preset veryfast"),
                "prores": ("prores", "-c:v prores_ks -profile:v 4 -qscale:v 16"),
            }