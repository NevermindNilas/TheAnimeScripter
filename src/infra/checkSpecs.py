import logging
import os
import platform

import psutil

import src.constants as cs


def _windowsCpuName():
    # Registry holds the marketing name ("13th Gen Intel Core i7-13700K"); it is
    # a sub-millisecond read vs platform.processor()'s WMI shell-out, which only
    # returns the cryptic PROCESSOR_IDENTIFIER string anyway.
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
    except OSError:
        return None


def getWindowsInfo():
    osName = platform.system()
    osVersion = platform.release()

    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)

    cpuInfo = (
        _windowsCpuName()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
    )

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam:.2f} GB")

    from src.infra.isCudaInit import listWindowsGpuNames

    gpus = listWindowsGpuNames()
    if gpus:
        logging.info(f"GPU(s): {', '.join(gpus)}")


def getLinuxInfo():
    osName = platform.system()
    osVersion = platform.release()

    cpuInfo = platform.processor()
    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam} GB")


def getMacosInfo():
    osName = platform.system()
    osVersion = platform.mac_ver()[0] or platform.release()

    cpuInfo = platform.processor() or platform.machine()
    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam} GB")


def checkSystem():
    logging.info("\n============== System Checker ==============")
    try:
        if cs.SYSTEM == "Windows":
            getWindowsInfo()
        elif cs.SYSTEM == "Linux":
            getLinuxInfo()
        elif cs.SYSTEM == "Darwin":
            getMacosInfo()
        else:
            logging.error("Unsupported OS")
    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")
    except ImportError as e:
        logging.error(f"Error importing the required modules: {e}")
