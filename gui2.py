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
    makeTextWidget,
    makePrimaryWidget,
    makeButtonWidget,
    iconPaths,
    styleTextWidget,
    styleButtonWidget,
    styleBackgroundColor,
    stylePrimaryWidget,
)
import time
# python -m pip install BlurWindow

TITLE = "The Anime Scripter - 1.6.0 (Alpha)"
ICONPATH = os.path.join(os.path.dirname(__file__), "src", "assets", "icon.ico")
WIDTH, HEIGHT = 1280, 720



def create_widget(widget_type, style, size, pos, parent, **kwargs):
    if widget_type == 'button':
        return makeButtonWidget(style=style, size=size, pos=pos, parent=parent, **kwargs)
    elif widget_type == 'text':
        return makeTextWidget(style=style, size=size, pos=pos, parent=parent, **kwargs)
    elif widget_type == 'primary':
        return makePrimaryWidget(style=style, size=size, pos=pos, parent=parent, **kwargs)

class mainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(WIDTH, HEIGHT)

        self.setWindowIcon(QIcon(ICONPATH))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        GlobalBlur(self.winId(), Dark=True)
        self.setStyleSheet(styleBackgroundColor())

        self.pathWidgets = create_widget('primary', stylePrimaryWidget(), (WIDTH - 30, HEIGHT // 4 - 15), (15, 15), self)
        self.previewWidget = create_widget('primary', stylePrimaryWidget(), (WIDTH // 2 - 205, HEIGHT // 2 + 90), (15, 195), self)
        self.scriptsWidget = create_widget('primary', stylePrimaryWidget(), (WIDTH // 2 + 160, HEIGHT // 2 + 90), (WIDTH // 2 - 175, 195), self)

        self.runButton = create_widget('button', styleButtonWidget(borderRadius=25), (50, 50), (WIDTH // 2 - 25, HEIGHT - 60), self, icon=iconPaths("play"))
        self.settingsButton = create_widget('button', styleButtonWidget(borderRadius=25), (50, 50), (WIDTH // 2 + 40, HEIGHT - 60), self, icon=iconPaths("settings"))
        self.aboutButton = create_widget('button', styleButtonWidget(borderRadius=25), (50, 50), (WIDTH // 2 - 90, HEIGHT - 60), self, icon=iconPaths("about"))

        """
        self.inputButton = create_widget('button', styleButtonWidget(chanels=(0, 0, 0, 0.2), borderRadius=10), (150, 40), (190, 50), self, addText="Input")
        self.inputTextWidget = create_widget('text', styleTextWidget(chanels=(20, 20, 20, 0.5), borderRadius=10), (750, 40), (350, 50), self)

        self.outputButton = create_widget('button', styleButtonWidget(chanels=(0, 0, 0, 0.2), borderRadius=10), (150, 40), (190, 120), self, addText="Output")
        self.outputTextWidget = create_widget('text', styleTextWidget(chanels=(20, 20, 20, 0.5), borderRadius=10), (750, 40), (350, 120), self)
        """

    # avoiding possible lag in the window when moving the window arround
    # def moveEvent(self, event) -> None:
    #    time.sleep(0.02)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainApp()
    window.show()
    sys.exit(app.exec())
