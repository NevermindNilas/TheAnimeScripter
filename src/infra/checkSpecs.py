import logging
import os
import platform

import psutil

from src.constants import SYSTEM


def getWindowsInfo():
    osName = platform.system()
    osVersion = platform.release()

    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)

    # platform.processor() shells out to WMI/registry (~12-28ms) and returns
    # exactly PROCESSOR_IDENTIFIER on Windows; read the env directly when set.
    cpuInfo = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam:.2f} GB")


def getLinuxInfo():
    osName = platform.system()
    osVersion = platform.release()

    cpuInfo = platform.processor()
    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam} GB")


def checkSystem():
    logging.info("\n============== System Checker ==============")
    try:
        if SYSTEM == "Windows":
            getWindowsInfo()
        elif SYSTEM == "Linux":
            getLinuxInfo()
        else:
            logging.error("Unsupported OS")
    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")
    except ImportError as e:
        logging.error(f"Error importing the required modules: {e}")
