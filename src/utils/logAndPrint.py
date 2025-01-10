import logging
from src.utils.coloredPrints import *  # noqa
from src.utils.coloredPrints import cyan, red, yellow, green


def logAndPrint(message: str, colorFunc: str = "cyan") -> None:
    if colorFunc == "cyan":
        print(cyan(message))
    elif colorFunc == "red":
        print(red(message))
    elif colorFunc == "yellow":
        print(yellow(message))
    elif colorFunc == "green":
        print(green(message))

    logging.info(message)
