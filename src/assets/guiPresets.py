import os

from PyQt6.QtWidgets import (
    QWidget,
    QPushButton,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect
)

from PyQt6.QtGui import QColor, QIcon

def stylePrimaryWidget(
    chanels: tuple = (255.0, 255.0, 255.0, 0.15), borderRadius: int = 15
) -> str:
    """
    Returns the stylesheet for the primary widget
    """
    return f"""
        QWidget {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]});
            border-radius: {borderRadius}px;
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
    chanels: tuple = (255.0, 255.0, 255.0, 0.20), borderRadius: int = 15, textColor: tuple[int, int, int] = (255, 255, 255), textSize: int = 12
) -> str:
    """
    Returns the stylesheet for the primary widget
    """
    return f"""
        QPushButton {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3]});
            border-radius: {borderRadius}px;
            font-weight: bold;
            font-size: {textSize}px;
            font-family: Roboto;
            color: rgb({textColor[0]}, {textColor[1]}, {textColor[2]});
        }}
        QPushButton:hover {{
            background-color: rgba({chanels[0]}, {chanels[1]}, {chanels[2]}, {chanels[3] + 0.1});
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


def makePrimaryWidget(
    size: tuple[int, int],
    pos: tuple[int, int],
    parent: QWidget,
    style: str,
    addShadow: bool = True,
    addShadowBlurRadius: int = 50,
) -> QWidget:
    widget = QWidget(parent)
    widget.setGeometry(*pos, *size)
    widget.setStyleSheet(style)

    if addShadow:
        addShadowEffect(widget, blurRadius=addShadowBlurRadius)

    return widget


def makeButtonWidget(
    size: tuple[int, int],
    pos: tuple[int, int],
    parent: QWidget,
    style: str,
    addShadow: bool = True,
    addShadowBlurRadius: int = 50,
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

    if addShadow:
        addShadowEffect(widget, blurRadius=addShadowBlurRadius)

    # Set the opacity of the button
    opacityEffect = QGraphicsOpacityEffect()
    opacityEffect.setOpacity(opacity)
    widget.setGraphicsEffect(opacityEffect)

    return widget

def iconPaths(iconName: str) -> str:
    """
    Returns the path for the icon
    """
    match iconName.lower():
        case "logo":
            pass

        case "play":
            try:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.ico")
            
            except FileNotFoundError:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.png")
            
        case "playoutline":
            try:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.ico")
            except FileNotFoundError:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.png")
            
        case "settings":
            try:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.ico")
            except FileNotFoundError:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.png")
            
        case "about":
            try:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.ico")
            except FileNotFoundError:
                return os.path.join(os.path.dirname(__file__), f"{iconName}.png")
            
        case _:
            raise FileNotFoundError(f"Icon {iconName} not found in assets folder")