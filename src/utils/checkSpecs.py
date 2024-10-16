import logging


def getWindowsInfo():
    import wmi

    computer = wmi.WMI()
    procInfo = computer.Win32_Processor()[0]
    gpuInfo = computer.Win32_VideoController()
    osInfo = computer.Win32_OperatingSystem()[0]
    totalRam = float(osInfo.TotalVisibleMemorySize) / 1048576  # Convert KB to GB

    logging.info(f"CPU: {procInfo.Name}")
    logging.info(f"Total RAM: {totalRam:.2f} GB")
    for i, gpu in enumerate(gpuInfo):
        logging.info(f"Graphics Card {i}: {gpu.Name}")


def getLinuxInfo():
    import psutil
    import GPUtil

    cpuInfo = psutil.cpu_info()
    ramInfo = psutil.virtual_memory()
    totalRam = round(ramInfo.total / (1024.0**3), 2)  # Convert Bytes to GB
    gpus = GPUtil.getGPUs()

    logging.info(f"CPU: {cpuInfo.brand_raw}")
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
