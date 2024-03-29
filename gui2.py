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

from PyQt6.QtGui import QIntValidator, QIcon
from PyQt6.QtCore import Qt

from BlurWindow.blurWindow import GlobalBlur

from src.assets.guiPresets import (
    makePrimaryWidget,
    stylePrimaryWidget,
    styleBackgroundColor,
    styleButtonWidget,
    makeButtonWidget,
    iconPaths,
)
import time
# python -m pip install BlurWindow

TITLE = "The Anime Scripter - 1.6.0 (Alpha)"
ICONPATH = os.path.join(os.path.dirname(__file__), "src", "assets", "icon.ico")
WIDTH, HEIGHT = 1280, 720


class mainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(WIDTH, HEIGHT)

        self.setWindowIcon(QIcon(ICONPATH))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        GlobalBlur(self.winId(), Dark=True)
        self.setStyleSheet(styleBackgroundColor())

        # Layouts are a bit better for organizing widgets
        # I will use them in the future, I promise
        """
        self.sidePanel = makePrimaryWidget(
            style=stylePrimaryWidget(borderRadius=0),
            size=(WIDTH // 6, HEIGHT),
            pos=(-4, 0), # eyeballing it a bit, will fix later
            parent=self,
        )
        """

        self.pathWidgets = makePrimaryWidget(
            style=stylePrimaryWidget(),
            size=(WIDTH - 30, HEIGHT // 4),
            pos=(15, 15),
            parent=self,
        )

        self.scriptWidgets = makePrimaryWidget(
            style=stylePrimaryWidget(),
            size=(WIDTH - 30, HEIGHT - 300),
            pos=(15, 210),
            parent=self,
        )

        self.runButton = makeButtonWidget(
            style=styleButtonWidget(borderRadius=25),
            size=(50, 50),
            pos=(WIDTH // 2 - 25, HEIGHT - 70),
            parent=self,
            icon=iconPaths("play"),
        )

        self.settingsButton = makeButtonWidget(
            style=styleButtonWidget(borderRadius=25),
            size=(50, 50),
            pos=(WIDTH // 2 + 40, HEIGHT - 70),
            parent=self,
            icon=iconPaths("settings"),
        )


        self.aboutButton = makeButtonWidget(
            style=styleButtonWidget(borderRadius=25),
            size=(50, 50),
            pos=(WIDTH // 2 - 90, HEIGHT - 70),
            parent=self,
            icon=iconPaths("about"),
        )

        self.inputButton = makeButtonWidget(
            style=styleButtonWidget(borderRadius=25),
            size=(50, 50),
            pos=(WIDTH // 2 - 155, HEIGHT - 70),
            parent=self,
            #icon=iconPaths("input"),
            opacity=0.2,
        )

        self.outputButton = makeButtonWidget(
            style=styleButtonWidget(borderRadius=25),
            size=(50, 50),
            pos=(WIDTH // 2 + 105, HEIGHT - 70),
            parent=self,
            #icon=iconPaths("output"),
            opacity=0.2,
        )


    # avoiding possible lag in the window when moving the window arround
    # def moveEvent(self, event) -> None:
    #    time.sleep(0.02)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainApp()
    window.show()
    sys.exit(app.exec())
