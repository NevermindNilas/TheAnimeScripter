import subprocess
import sys
import os
import logging
import src.constants as cs
import json
import time
import hashlib

from pathlib import Path
from typing import Tuple
from src.utils.logAndPrint import logAndPrint


def getPythonExecutable() -> str:
    """
    Get the path to the current Python executable

    Returns:
        str: Path to the Python executable
    """
    return sys.executable


def uninstallDependencies(extension: str = "") -> Tuple[bool, str]:
    """Uninstall dependencies from extra-requirements-windows.txt if it exists
    Args:
        extension (str): Optional extension to the requirements file name"""

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
    cmd = f'"{pythonPath}" -I -m pip uninstall -y -r "{requirementsPath}"'
    try:
        logMessage = f"Uninstalling requirements from: {requirementsPath}"
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
            errorMsg = f"Error uninstalling requirements (exit code: {returnCode})"
            logging.error(errorMsg)
            print(errorMsg)
            return False, errorMsg

        return True, "Successfully uninstalled dependencies from requirements file"
    except Exception as e:
        errorMsg = f"Error uninstalling requirements: {str(e)}"
        logging.error(errorMsg)
        print(errorMsg)
        return False, errorMsg


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


class DependencyChecker:
    def __init__(self):
        self.cachePath = Path(cs.MAINPATH) / ".dependencyCache.json"
        self.cacheExpiry = 2592000  # every 30 days or so

    def needsUpdate(self, requirementsPath):
        """Check if dependencies need updating"""
        cache = self._loadCache()

        if time.time() - cache.get("timestamp", 0) > self.cacheExpiry:
            return True

        currentHash = self._getFileHash(requirementsPath)
        if currentHash != cache.get("requirements_hash"):
            return True

        return self._criticalPackagesChanged(cache.get("versions", {}))

    def updateCache(self, requirementsPath):
        """Update cache after successful installation"""
        versions = self._getCurrentVersions()
        cache = {
            "timestamp": time.time(),
            "requirements_hash": self._getFileHash(requirementsPath),
            "versions": versions,
        }

        try:
            with open(self.cachePath, "w") as f:
                json.dump(cache, f)
        except:
            pass

    def forceFullDownload(self, requirementsFile=None):
        """
        Force a full download of all dependencies, bypassing cache checks.
        This merges the functionality from initializeAFullDownload.
        """
        if requirementsFile is None:
            from src.utils.isCudaInit import detectNVidiaGPU

            isNvidia = detectNVidiaGPU()
            requirementsFile = (
                "extra-requirements-windows.txt"
                if isNvidia
                else "extra-requirements-windows-lite.txt"
            )

        requirementsPath = os.path.join(cs.WHEREAMIRUNFROM, requirementsFile)

        logAndPrint("Forcing full dependency download...", "yellow")
        success, message = installDependencies(requirementsFile)

        if not success:
            logAndPrint(message, "red")
            raise RuntimeError(f"Failed to install dependencies: {message}")
        else:
            logAndPrint(message, "green")
            self.updateCache(requirementsPath)
            return True

    def _loadCache(self):
        """Load cache from file"""
        if not self.cachePath.exists():
            return {}
        try:
            with open(self.cachePath, "r") as f:
                return json.load(f)
        except:
            return {}

    def _getFileHash(self, filepath):
        """Get MD5 hash of file"""
        try:
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None

    def _getCurrentVersions(self):
        """Get versions of critical packages"""
        versions = {}
        criticalPackages = {
            "torch": lambda: __import__("torch").__version__,
            "torchvision": lambda: __import__("torchvision").__version__,
            "numpy": lambda: __import__("numpy").__version__,
            "basswood-av": lambda: __import__("bv").__version__,
        }

        for pkg, getter in criticalPackages.items():
            try:
                versions[pkg] = getter()
            except ImportError:
                versions[pkg] = None

        return versions

    def _criticalPackagesChanged(self, cachedVersions):
        """Check if critical packages have different versions"""
        currentVersions = self._getCurrentVersions()
        return currentVersions != cachedVersions
