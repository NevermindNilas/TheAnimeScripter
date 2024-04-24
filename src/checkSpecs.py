import logging
import wmi


def checkSystem():
    try:
        logging.info("\n============== System Checker ==============")
        computer = wmi.WMI()
        os_info = computer.Win32_OperatingSystem()[0]
        proc_info = computer.Win32_Processor()[0]
        gpu_info = computer.Win32_VideoController()
        os_name = os_info.Name.encode("utf-8").split(b"|")[0].decode("utf-8")
        system_ram = format(float(os_info.TotalVisibleMemorySize) / 1048576, '.2f')  # in GB
        available_ram = format(float(os_info.FreePhysicalMemory) / 1048576, '.2f')  # in GB

        logging.info(f"OS Name: {os_name}")
        logging.info(f"CPU: {proc_info.Name}")
        logging.info(f"RAM: {system_ram} GB")
        logging.info(f"Available RAM: {available_ram} GB")
        for i in range(len(gpu_info)):
            logging.info(f"Graphics Card {i}: {gpu_info[i].Name}")

    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")