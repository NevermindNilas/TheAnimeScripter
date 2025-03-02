import logging
import psutil
import GPUtil
import platform


def getWindowsInfo():
    osName = platform.system()
    osVersion = platform.release()

    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)

    cpuInfo = platform.processor()
    gpus = GPUtil.getGPUs()

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam:.2f} GB")
    for i, gpu in enumerate(gpus):
        logging.info(f"Graphics Card {i}: {gpu.name}")


def getLinuxInfo():
    osName = platform.system()
    osVersion = platform.release()

    cpuInfo = platform.processor()
    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)
    gpus = GPUtil.getGPUs()

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam} GB")
    for i, gpu in enumerate(gpus):
        logging.info(f"Graphics Card {i}: {gpu.name}")


def checkSystem(sysUsed):
    logging.info("\n============== System Checker ==============")
    try:
        if sysUsed == "Windows":
            getWindowsInfo()
        elif sysUsed == "Linux":
            getLinuxInfo()
        else:
            logging.error("Unsupported operating system")
    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")
    except ImportError as e:
        logging.error(f"Error importing the required modules: {e}")
