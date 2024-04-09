import os

from PyQt6.QtWidgets import (
    QWidget,
    QPushButton,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGraphicsBlurEffect,
    QLabel,
)

from PyQt6.QtGui import QColor, QIcon

DEFAULTFONT = "Segoe UI"

def stylePrimaryWidget(
    chanels: tuple = (255.0, 255.0, 255.0, 0.15), borderRadius: int = 15
) -> str:
    """
    Returns the stylesheet for the primary widget
    """
    return f"""
        QWidget {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                                        stop: 0 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.05}), 
                                        stop: 0.5 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]}), 
                                        stop: 1 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] - 0.05}));
            border-radius: {borderRadius}px;
            border: 1px solid rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]});
        }}
    """


def styleBackgroundColor(chanels: tuple = (255.0, 255.0, 255.0, 0.1)) -> str:
    """
    Returns the stylesheet for the background color
    """
    return f"""
        background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]});
    """


def styleButtonWidget(
    chanels: tuple = (255.0, 255.0, 255.0, 0.20),
    borderRadius: int = 15,
    textColor: tuple[int, int, int] = (255, 255, 255),
    textSize: int = 15,
) -> str:
    """
    Returns the stylesheet for the primary widget
    """
    """
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                stop: 0 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.1}), 
                stop: 1 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]}));
    """
    return f"""
        QPushButton {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]});
            border-radius: {borderRadius}px;
            font-weight: bold;
            font-size: {textSize}px;
            font-family: '{DEFAULTFONT}';
            color: rgb({textColor[0]}, {textColor[1]}, {textColor[2]});
            border: none;
            padding: 5px;
        }}
        QPushButton:hover {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.1});
        }}
        QPushButton:pressed {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.2});
        }}
    """


def styleTextWidget(
    chanels: tuple = (255.0, 255.0, 255.0, 0.20),
    borderRadius: int = 15,
    textColor: tuple[int, int, int] = (255, 255, 255),
    textSize: int = 15,
) -> str:
    """
    Returns the stylesheet for the text widget
    """
    return f"""
        QLabel {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]});
            border-radius: {borderRadius}px;
            font-weight: bold;
            font-size: {textSize}px;
            font-family: '{DEFAULTFONT}';
            color: rgb({textColor[0]}, {textColor[1]}, {textColor[2]});
            border: 1px solid rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.2});
            padding: 5px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                                        stop: 0 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.1}), 
                                        stop: 1 rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]}));
        }}
    """

def addShadowEffect(widget, blurRadius: int = 50):
    """
    Adds a shadow effect to the widget
    """
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blurRadius)
    shadow.setXOffset(0)
    shadow.setYOffset(0)
    shadow.setColor(QColor(0, 0, 0, 255))
    widget.setGraphicsEffect(shadow)

def addBlurEffect(widget, blurRadius: int = 10):
    """
    Adds a blur effect to the widget
    """
    blur = QGraphicsBlurEffect()
    blur.setBlurRadius(blurRadius)
    widget.setGraphicsEffect(blur)

def makePrimaryWidget(
    size: tuple[int, int],
    pos: tuple[int, int],
    parent: QWidget,
    mainStyle: str = "",
    addShadow: bool = True,
    addShadowBlurRadius: int = 50,
    labelText: str = "",
    labelOffset: tuple[int, int] = (15, 15),
    labelSize: int = 15,
) -> QWidget:
    widget = QWidget(parent)
    widget.setGeometry(*pos, *size)
    widget.setStyleSheet(mainStyle)

    if labelText:
        label = QLabel(labelText, widget)
        label.move(*labelOffset)
        label.setAutoFillBackground(True)
        label.setStyleSheet(f"""
            QLabel {{
                background-color: none;
                color: white;
                font-weight: bold;
                font-size: {labelSize}px;
                font-family: '{DEFAULTFONT}';
                border: none;
            }}
        """)

    if addShadow:
        addShadowEffect(widget, blurRadius=addShadowBlurRadius)

    return widget

def makeButtonWidget(
    size: tuple[int, int],
    pos: tuple[int, int],
    parent: QWidget,
    style: str,
    addText: str = "",
    icon: str = "",
    opacity: float = 1.0,
) -> QPushButton:
    widget = QPushButton(parent)
    widget.setGeometry(*pos, *size)
    widget.setStyleSheet(style)

    if addText:
        widget.setText(addText)

    if icon:
        widget.setIcon(QIcon(icon))


    if opacity < 1.0:
        opacityEffect = QGraphicsOpacityEffect()
        opacityEffect.setOpacity(opacity)
        widget.setGraphicsEffect(opacityEffect)

    return widget

def makeTextWidget(
    size: tuple[int, int],
    pos: tuple[int, int],
    parent: QWidget,
    style: str,
    addText: str = "",
    opacity: float = 1.0,
) -> QLabel:
    widget = QLabel(parent)
    widget.setGeometry(*pos, *size)
    widget.setStyleSheet(style)

    if addText:
        widget.setText(addText)

    if opacity < 1.0:
        opacityEffect = QGraphicsOpacityEffect()
        opacityEffect.setOpacity(opacity)
        widget.setGraphicsEffect(opacityEffect)

    return widget

def iconPaths(iconName: str) -> str:
    """
    Returns the path for the icon
    """
    for extension in ["ico", "png"]:
        try:
            path = os.path.join(
                os.path.dirname(__file__), f"{iconName.lower()}.{extension}"
            )
            if os.path.exists(path):
                return path
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Icon {iconName} not found in assets folder")

def replaceForwardWithBackwardSlashes(path: str) -> str:
    """
    Replaces forward slashes with backward slashes
    Just to avoid any issues with Windows paths
    """
    return path.replace("/", "\\")