import subprocess
import sys
import os
import logging
from typing import Tuple
import src.constants as cs


def getPythonExecutable() -> str:
    """
    Get the path to the current Python executable

    Returns:
        str: Path to the Python executable
    """
    return sys.executable


def installDependencies(extension: str = "") -> Tuple[bool, str]:
    """
    Install dependencies from extra-requirements-windows.txt if it exists

    Returns:
        Tuple[bool, str]: Success status and message
    """
    pythonPath = getPythonExecutable()
    if not pythonPath:
        return False, "Failed to detect Python executable path"

    requirementsPath = os.path.join(os.path.dirname(pythonPath), extension)
    if not os.path.exists(requirementsPath):
        requirementsPath = os.path.join(cs.WHEREAMIRUNFROM, extension)
        if not os.path.exists(requirementsPath):
            return False, f"Requirements file not found: {requirementsPath}"

    logMessage = f"Using Python executable: {pythonPath}"
    logging.info(logMessage)

    cmd = f'"{pythonPath}" -I -m pip install -U -r "{requirementsPath}" --no-warn-script-location'

    try:
        logMessage = f"Installing requirements from: {requirementsPath}"
        logging.info(logMessage)
        print(logMessage)

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            logging.info(line.strip())

        process.stdout.close()
        returnCode = process.wait()

        if returnCode != 0:
            errorMsg = f"Error installing requirements (exit code: {returnCode})"
            logging.error(errorMsg)
            print(errorMsg)
            return False, errorMsg

        return True, "Successfully installed dependencies from requirements file"

    except Exception as e:
        errorMsg = f"Error installing requirements: {str(e)}"
        logging.error(errorMsg)
        print(errorMsg)
        return False, errorMsg
