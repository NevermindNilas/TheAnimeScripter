"""
Logging and Print Utilities

Provides unified functions for both console output and file logging.
Ensures consistent message formatting and logging throughout the application.
"""

import logging
from src.utils.coloredPrints import cyan, red, yellow, green


def logAndPrint(message: str, colorFunc: str = "cyan") -> None:
    """
    Print a colored message to console and log it to file.
    
    Args:
        message (str): Message to display and log
        colorFunc (str): Color function name ('cyan', 'red', 'yellow', 'green')
    """
    colorFunctions = {
        "cyan": cyan,
        "red": red,
        "yellow": yellow,
        "green": green
    }
    
    if colorFunc in colorFunctions:
        print(colorFunctions[colorFunc](message))
    else:
        print(message)
    
    logging.info(message)


def coloredPrint(message: str, colorFunc: str = "cyan") -> None:
    """
    Print a colored message to console only (no logging).
    
    Args:
        message (str): Message to display
        colorFunc (str): Color function name ('cyan', 'red', 'yellow', 'green')
    """
    colorFunctions = {
        "cyan": cyan,
        "red": red,
        "yellow": yellow,
        "green": green
    }
    
    if colorFunc in colorFunctions:
        print(colorFunctions[colorFunc](message))
    else:
        print(message)
