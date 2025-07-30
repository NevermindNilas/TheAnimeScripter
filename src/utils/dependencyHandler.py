import subprocess
import sys
import os
import logging
import src.constants as cs
import json
import hashlib

from pathlib import Path
from typing import Tuple, Iterable
from src.utils.logAndPrint import logAndPrint


def getPythonExecutable() -> str:
    """
    Get the path to the current Python executable

    Returns:
        str: Path to the Python executable
    """
    return sys.executable


# https://github.com/pypa/hatch/blob/4ebce0e1fe8bf0fcdef587a704c207a063d72575/src/hatch/utils/platform.py#L176-L180
def streamProcessOutput(process) -> Iterable[str]:
    # To avoid blocking never use a pipe's file descriptor iterator. See https://bugs.python.org/issue3907
    for line in iter(process.stdout.readline, b""):
        yield line.decode("utf-8")


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
        )

        for line in streamProcessOutput(process):
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


def installDependencies(extension: str = "", isNvidia: bool = True) -> Tuple[bool, str]:
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

    cmd = f'"{pythonPath}" -I -m pip install -r "{requirementsPath}" --no-warn-script-location --no-cache'

    try:
        logMessage = f"Installing requirements from: {requirementsPath}"
        logging.info(logMessage)
        print(logMessage)

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for line in streamProcessOutput(process):
            print(line, end="")
            logging.info(line.strip())

        process.stdout.close()
        returnCode = process.wait()

        if returnCode != 0:
            errorMsg = f"Error installing requirements (exit code: {returnCode})"
            logging.error(errorMsg)
            print(errorMsg)
            return False, errorMsg

        """
        if isNvidia:
            success, message = _installTensorRTRTX()
            if not success:
                logAndPrint(message, "red")
                raise RuntimeError(f"Failed to install TensorRT/RTX: {message}")
        """

        return True, "Successfully installed dependencies from requirements file"

    except Exception as e:
        errorMsg = f"Error installing requirements: {str(e)}"
        logging.error(errorMsg)
        print(errorMsg)
        return False, errorMsg


class DependencyChecker:
    def __init__(self):
        self.cachePath = Path(cs.MAINPATH) / ".dependencyCache.json"
        self._cache = None
        self._requirementsHash = {}

    def needsUpdate(self, requirementsPath):
        """Check if dependencies need updating - hash + torch import check"""
        cache = self._loadCache()

        currentHash = self._getFileHashCached(requirementsPath)

        # If hash differs, definitely need update
        if currentHash != cache.get("requirements_hash"):
            return True

        # Hash matches, but verify torch actually works
        try:
            import torch

            return False  # Should imply that dependencies are up-to-date, if not broken
        except ImportError:
            logging.info("Hash matches but torch import failed, triggering pip install")
            return True  # Torch broken or not present, need to install dependencies

    def updateCache(self, requirementsPath):
        """Update cache after successful installation"""
        cache = {
            "requirements_hash": self._getFileHashCached(requirementsPath),
        }

        try:
            with open(self.cachePath, "w") as f:
                json.dump(cache, f)
            self._cache = cache
        except Exception as e:
            logging.warning(f"Failed to update dependency cache: {e}")

    def forceFullDownload(self, requirementsFile=None):
        """
        Force a full download of all dependencies, bypassing cache checks.
        This merges the functionality from initializeAFullDownload.
        """
        if requirementsFile is None:
            from src.utils.isCudaInit import detectNVidiaGPU

            isNvidia = detectNVidiaGPU()
            if cs.SYSTEM == "Windows":
                requirementsFile = (
                    "extra-requirements-windows.txt"
                    if isNvidia
                    else "extra-requirements-windows-lite.txt"
                )
            else:  # Linux and other systems
                requirementsFile = (
                    "extra-requirements-linux.txt"
                    if isNvidia
                    else "extra-requirements-linux-lite.txt"
                )

        requirementsPath = os.path.join(cs.WHEREAMIRUNFROM, requirementsFile)

        self.uninstallDeprecatedDependencies()

        logAndPrint("Forcing full dependency download...", "yellow")
        success, message = installDependencies(requirementsFile, isNvidia=isNvidia)

        if not success:
            logAndPrint(message, "red")
            raise RuntimeError(f"Failed to install dependencies: {message}")
        else:
            logAndPrint(message, "green")
            self.updateCache(requirementsPath)
            return True

    def _loadCache(self):
        """Load cache from file - with in-memory caching"""
        if self._cache is not None:
            return self._cache

        if not self.cachePath.exists():
            self._cache = {}
            return self._cache

        try:
            with open(self.cachePath, "r") as f:
                self._cache = json.load(f)
            return self._cache
        except Exception as e:
            logging.warning(f"Failed to load dependency cache: {e}")
            self._cache = {}
            return self._cache

    def _getFileHashCached(self, filepath):
        """Get MD5 hash of file with caching"""
        if filepath in self._requirementsHash:
            return self._requirementsHash[filepath]

        file_hash = self._getFileHash(filepath)
        self._requirementsHash[filepath] = file_hash
        return file_hash

    def _getFileHash(self, filepath):
        """Get MD5 hash of file"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logging.warning(f"Failed to hash file {filepath}: {e}")
            return None

    def uninstallDeprecatedDependencies(self):
        """
        Uninstall deprecated dependencies from deprecated-requirements.txt
        """
        deprecatedRequirementsFile = "deprecated-requirements.txt"
        deprecatedRequirementsPath = os.path.join(
            cs.WHEREAMIRUNFROM, deprecatedRequirementsFile
        )

        if not os.path.exists(deprecatedRequirementsPath):
            logAndPrint(
                f"Deprecated requirements file not found: {deprecatedRequirementsPath}",
                "yellow",
            )
            return True  # Not an error if file doesn't exist

        logAndPrint("Uninstalling deprecated dependencies...", "yellow")
        success, message = uninstallDependencies(deprecatedRequirementsFile)

        if not success:
            logAndPrint(
                f"Failed to uninstall deprecated dependencies: {message}", "red"
            )
            return False
        else:
            logAndPrint("Successfully uninstalled deprecated dependencies", "green")
            return True


def uninstallDeprecatedDependenciesStandalone() -> Tuple[bool, str]:
    """
    Standalone function to uninstall deprecated dependencies from deprecated-requirements.txt

    Returns:
        Tuple[bool, str]: Success status and message
    """
    checker = DependencyChecker()
    try:
        success = checker.uninstallDeprecatedDependencies()
        if success:
            return True, "Successfully uninstalled deprecated dependencies"
        else:
            return False, "Failed to uninstall deprecated dependencies"
    except Exception as e:
        errorMsg = f"Error during deprecated dependencies uninstall: {str(e)}"
        logging.error(errorMsg)
        return False, errorMsg


def _installTensorRTRTX() -> Tuple[bool, str]:
    """Install TensorRT and RTX dependencies if not already installed.
    Returns:
        Tuple[bool, str]: Success status and message
    """
    from src.utils.downloadModels import downloadTensorRTRTX

    logAndPrint("Installing TensorRT and RTX dependencies...", "yellow")
    try:
        success = downloadTensorRTRTX()
        if not success:
            logAndPrint("Failed to install TensorRT/RTX:", "red")
            return False, "Failed to install TensorRT/RTX"
        else:
            logAndPrint("Successfully installed TensorRT/RTX", "green")
            return True, "Successfully installed TensorRT/RTX"
    except Exception as e:
        errorMsg = f"Error installing TensorRT/RTX: {str(e)}"
        logging.error(errorMsg)
        logAndPrint(errorMsg, "red")
        return False, errorMsg
