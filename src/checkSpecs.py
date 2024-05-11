import logging
import wmi


def checkSystem():
    try:
        logging.info("\n============== System Checker ==============")
        computer = wmi.WMI()
        osInfo = computer.Win32_OperatingSystem()[0]
        procInfo = computer.Win32_Processor()[0]
        gpuInfo = computer.Win32_VideoController()
        osName = osInfo.Name.encode("utf-8").split(b"|")[0].decode("utf-8")
        systemRam = format(float(osInfo.TotalVisibleMemorySize) / 1048576, '.2f')  # in GB
        availableRam = format(float(osInfo.FreePhysicalMemory) / 1048576, '.2f')  # in GB

        logging.info(f"OS Name: {osName}")
        logging.info(f"CPU: {procInfo.Name}")
        logging.info(f"RAM: {systemRam} GB")
        logging.info(f"Available RAM: {availableRam} GB")
        for i in range(len(gpuInfo)):
            logging.info(f"Graphics Card {i}: {gpuInfo[i].Name}")

    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")