import logging
from src.utils.coloredPrints import *  # noqa


def logAndPrint(message: str, colorFunc):
    print(colorFunc(message))
    logging.info(message)
