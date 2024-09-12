import logging
import os.path as osp
from termcolor import colored

def set_logging(name=None, verbose=True):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Sets level and returns logger
    # rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    fmt = (
        # colored("[%(name)s]", "magenta", attrs=["bold"])
        colored("[%(asctime)s]", "blue")
        + colored("%(levelname)s:", "green")
        + colored("%(message)s", "white")
    )
    logging.basicConfig(format=fmt, level=logging.INFO if verbose else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)

