import logging
import platform

try:
    import wmi
except ImportError:
    if platform.system() == "Windows":
        logging.error("No system checker available, please install wmi for Windows")
        raise SystemExit

try:
    import psutil
    import GPUtil
except ImportError:
    if platform.system() == "Linux":
        logging.error(
            "No system checker available, please install psutil and GPUtil for Linux"
        )
        raise SystemExit

def getWindowsInfo():
    computer = wmi.WMI()
    osInfo = computer.Win32_OperatingSystem()[0]
    procInfo = computer.Win32_Processor()[0]
    gpuInfo = computer.Win32_VideoController()
    osName = osInfo.Name.encode("utf-8").split(b"|")[0].decode("utf-8")
    systemRam = format(float(osInfo.TotalVisibleMemorySize) / 1048576, ".2f")  # Convert KB to GB
    availableRam = format(float(osInfo.FreePhysicalMemory) / 1048576, ".2f")  # Convert KB to GB

    logging.info(f"OS Name: {osName}")
    logging.info(f"CPU: {procInfo.Name}")
    logging.info(f"RAM: {systemRam} GB")
    logging.info(f"Available RAM: {availableRam} GB")
    for i in range(len(gpuInfo)):
        logging.info(f"Graphics Card {i}: {gpuInfo[i].Name}")

def getLinuxInfo():
    osName = platform.uname().system
    cpuCount = psutil.cpu_count()
    cpuFreq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"  # Some systems may not support cpu_freq
    ramInfo = psutil.virtual_memory()
    systemRam = round(ramInfo.total / (1024.0**3), 2)  # Convert Bytes to GB
    availableRam = round(ramInfo.available / (1024.0**3), 2)  # Convert Bytes to GB
    gpus = GPUtil.getGPUs()

    logging.info(f"OS Name: {osName}")
    logging.info(f"CPU: {cpuCount} cores, {cpuFreq} MHz")
    logging.info(f"RAM: {systemRam} GB")
    logging.info(f"Available RAM: {availableRam} GB")
    for i, gpu in enumerate(gpus):
        logging.info(f"Graphics Card {i}: {gpu.name}")

def checkSystem():
    try:
        logging.info("\n============== System Checker ==============")
        if platform.system() == "Windows":
            getWindowsInfo()
        elif platform.system() == "Linux":
            getLinuxInfo()
    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")