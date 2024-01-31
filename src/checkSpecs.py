import logging
import GPUtil
import psutil

def checkSystem():
    try:
        # Log GPU info
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                logging.info(f"GPU: {gpu.name}")
        else:
            logging.info("No GPU detected")

        # Log CPU info
        # Will come back to this later

        # Log RAM info
        ram = psutil.virtual_memory()
        logging.info(f"Total RAM: {ram.total / 1024**3:.2f} GB")
        logging.info(f"Available RAM: {ram.available / 1024**3:.2f} GB")

    except Exception as e:
        logging.error(f"An error occurred while checking the system: {e}")