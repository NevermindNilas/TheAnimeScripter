import logging
import os
import platform
import sys

import src.constants as cs


def _totalRamBytes():
    # Read from the OS directly: psutil costs ~9 ms to import, and this log
    # line is the only thing that used it.
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys
    else:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except ValueError, OSError, AttributeError:
            pass
    import psutil

    return psutil.virtual_memory().total


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

    totalRam = round(_totalRamBytes() / (1024.0**3), 2)

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
    totalRam = round(_totalRamBytes() / (1024.0**3), 2)

    logging.info(f"OS: {osName} {osVersion}")
    logging.info(f"CPU: {cpuInfo}")
    logging.info(f"Total RAM: {totalRam} GB")


def getMacosInfo():
    osName = platform.system()
    osVersion = platform.mac_ver()[0] or platform.release()

    cpuInfo = platform.processor() or platform.machine()
    totalRam = round(_totalRamBytes() / (1024.0**3), 2)

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
