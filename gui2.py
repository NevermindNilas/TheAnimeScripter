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



def createWidget(widget_type, style, size, pos, parent, **kwargs):
    if widget_type == 'button':
        return makeButtonWidget(style=style, size=size, pos=pos, parent=parent, **kwargs)
    elif widget_type == 'textArea':
        return makeTextWidget(style=style, size=size, pos=pos, parent=parent, **kwargs)
    elif widget_type == 'primary':
        return makePrimaryWidget(mainStyle=style, size=size, pos=pos, parent=parent, **kwargs)

class mainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(TITLE)
        self.setFixedSize(WIDTH, HEIGHT)

        self.setWindowIcon(QIcon(ICONPATH))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        GlobalBlur(self.winId(), Dark=True)
        self.setStyleSheet(styleBackgroundColor())

        self.handleUI() # create the widgets
        self.buttonLogic() # connect the buttons to the functions
    
    def handleUI(self):
        self.pathWidgets = createWidget('primary', stylePrimaryWidget(), (WIDTH // 2 + 160, HEIGHT // 4 - 15), (WIDTH // 2 - 175, 15), self, labelText = "Paths")
        self.optionsWidget = createWidget('primary', stylePrimaryWidget(), (WIDTH // 2 - 205, HEIGHT // 2 + 270), (15, 15), self, labelText = "Options")
        self.terminalWidget = createWidget('primary', stylePrimaryWidget(), (WIDTH // 2 + 160, HEIGHT // 2 + 90), (WIDTH // 2 - 175, 195), self, labelText = "Terminal")
        self.dockWidget = createWidget('primary', stylePrimaryWidget(), (250, 50), (WIDTH // 2 - 125, HEIGHT - 62), self)
        #self.shadowOptionWidget = createWidget('primary', stylePrimaryWidget(chanels=(0,0,0,0.2)), (WIDTH // 2 - 235, HEIGHT // 2 - 60), (30, 60), self, addShadow=False)
        #self.shadowTerminalWidget = createWidget('primary', stylePrimaryWidget(chanels=(0,0,0,0.2)), (WIDTH // 2 - 235, HEIGHT // 2 - 60), (30, 60), self)

        self.runButton = createWidget('button', styleButtonWidget(chanels=(255, 255, 255, 0), borderRadius=25), (50, 50), (WIDTH // 2 - 25, HEIGHT - 62), self, icon=iconPaths("play"))
        self.settingsButton = createWidget('button', styleButtonWidget(chanels=(255, 255, 255, 0), borderRadius=25), (50, 50), (WIDTH // 2 + 40, HEIGHT - 62), self, icon=iconPaths("settings"))
        self.aboutButton = createWidget('button', styleButtonWidget(chanels=(255, 255, 255, 0), borderRadius=25), (50, 50), (WIDTH // 2 - 90, HEIGHT - 62), self, icon=iconPaths("about"), opacity = 0.1)

        self.inputButton = createWidget('button', styleButtonWidget(chanels=(0, 0, 0, 0.2), borderRadius=10), size = (100, 40), pos = (480, 70), parent= self, addText="Input")
        self.outputButton = createWidget('button', styleButtonWidget(chanels=(0, 0, 0, 0.2), borderRadius=10), size= (100, 40), pos= (480, 120), parent = self, addText="Output")

        self.inputTextWidget = createWidget('textArea', styleTextWidget(chanels=(20, 20, 20, 0.5), borderRadius=10), (650, 40), (600, 70), self)
        self.outputTextWidget = createWidget('textArea', styleTextWidget(chanels=(20, 20, 20, 0.5), borderRadius=10), (650, 40), pos = (600, 120), parent= self)

        self.resizeButton = createWidget('button', styleButtonWidget(chanels=(0, 0, 0, 0), borderRadius=10), size = (100, 40), pos = (15, 170), parent= self, addText="Resize")
        self.deduplicateButton = createWidget('button', styleButtonWidget(chanels=(0, 0, 0, 0), borderRadius=10), size = (100, 40), pos = (15, 220), parent= self, addText="Deduplicate")

    def buttonLogic(self):
        self.runButton.clicked.connect(self.runScript)
        self.settingsButton.clicked.connect(self.openSettings)
        self.aboutButton.clicked.connect(self.openAbout)

        self.inputButton.clicked.connect(self.openInput)
        self.outputButton.clicked.connect(self.openOutput)

    def runScript(self):
        pass

    def openSettings(self):
        pass

    def openAbout(self):
        pass

    def openInput(self):
        pass

    def openOutput(self):
        pass

        
    # avoiding possible lag in the window when moving the window arround
    # def moveEvent(self, event) -> None:
    #    time.sleep(0.02)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainApp()
    window.show()
    sys.exit(app.exec())
