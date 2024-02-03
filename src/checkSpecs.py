import logging
import wmi

def checkSystem():
    try:
        logging.info("\n============== System Checker ==============")
        computer = wmi.WMI()
        os_info = computer.Win32_OperatingSystem()[0]
        proc_info = computer.Win32_Processor()[0]
        gpu_info = computer.Win32_VideoController()
        os_name = os_info.Name.encode('utf-8').split(b'|')[0]
        system_ram = float(os_info.TotalVisibleMemorySize) / 1048576  # KB to GB

        logging.info(f"OS Name: {os_name}")
        logging.info(f"CPU: {proc_info.Name}")
        logging.info(f"RAM: {system_ram} GB")
        for gpu in gpu_info:
            logging.info(f"Graphics Card: {gpu.Name}")
        
    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")
        
    return system_ram