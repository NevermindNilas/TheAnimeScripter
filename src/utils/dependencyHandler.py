import subprocess
import sys
import os
import logging
import src.constants as cs
import json
import hashlib
import re

from pathlib import Path
from typing import Tuple, Iterable
from src.utils.logAndPrint import logAndPrint
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec


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
    cmd = f'"{pythonPath}" -I -m pip uninstall -y -r "{requirementsPath}" --disable-pip-version-check'
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

    cmd = f'"{pythonPath}" -I -m pip install -r "{requirementsPath}" --no-warn-script-location --no-cache --disable-pip-version-check'

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
        self.knownAliases = {
            "opencv-python": "cv2",
            "Pillow": "PIL",
            "PyYAML": "yaml",
            "scikit-image": "skimage",
            "scikit-learn": "sklearn",
            "onnxruntime-gpu": "onnxruntime",
            "onnxruntime-directml": "onnxruntime",
            "imageio-ffmpeg": "imageio_ffmpeg",
            "tensorrt": "tensorrt",
            "torch": "torch",
            "torchvision": "torchvision",
            "torchaudio": "torchaudio",
        }

    def needsUpdate(self, requirementsPath):
        """Check if dependencies need updating using hash + no-import presence checks"""
        cache = self._loadCache()

        currentHash = self._getFileHashCached(requirementsPath)

        # If hash differs, definitely need update
        if currentHash != cache.get("requirements_hash"):
            return True

        # Hash matches: verify required distributions are present (and optionally importable)
        missing, wrongVersion, notImportable = self.checkRequirementsInstalled(
            requirementsPath,
            moduleAliases=self.knownAliases,
            enforceVersions=False,
        )

        if missing:
            logging.info(f"Missing distributions detected: {missing}")
            return True

        if wrongVersion:
            logging.info(f"Version mismatches detected: {wrongVersion}")
            return True

        if notImportable:
            logging.debug(
                f"Distributions present but modules not importable via expected names: {notImportable}"
            )

        return False

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
        # If requirementsFile is provided explicitly, avoid referencing undefined isNvidia
        try:
            success, message = (
                installDependencies(requirementsFile, isNvidia=isNvidia)
                if "isNvidia" in locals()
                else installDependencies(requirementsFile)
            )
        except Exception as e:
            logAndPrint(str(e), "red")
            raise

        if not success:
            logAndPrint(message, "red")
            raise RuntimeError(f"Failed to install dependencies: {message}")
        else:
            logAndPrint(message, "green")
            self.updateCache(requirementsPath)
            return True

    def iterRequirements(self, requirementsPath: str):
        """Yield (name, rawSpecifier) pairs from a requirements.txt, ignoring comments, includes, and URLs."""
        try:
            with open(requirementsPath, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(("-r ", "--", "-c ")):
                        continue
                    if line.startswith(("-e ", "git+", "http://", "https://", "file:")):
                        continue
                    if ";" in line:
                        line = line.split(";", 1)[0].strip()
                    m = re.match(r"^([A-Za-z0-9._-]+)", line)
                    if not m:
                        continue
                    name = m.group(1)
                    spec = line[len(name) :].strip()
                    yield name, spec
        except FileNotFoundError:
            logging.warning(
                f"Requirements file not found for check: {requirementsPath}"
            )
            return
        except Exception as e:
            logging.warning(f"Failed to parse requirements at {requirementsPath}: {e}")
            return

    def checkRequirementsInstalled(
        self,
        requirementsPath: str,
        moduleAliases: dict | None = None,
        enforceVersions: bool = False,
    ):
        """Return (missing, wrongVersion, notImportable) without importing packages.

        - missing: list[str] of distributions not installed
        - wrongVersion: list[tuple[name, installedVersion, requiredSpecifier]] when enforceVersions True
        - notImportable: list[str] of expected top-level modules not discoverable via find_spec
        """
        moduleAliases = moduleAliases or {}
        missing: list[str] = []
        wrongVersion: list[tuple[str, str, str]] = []
        notImportable: list[str] = []

        try:
            from packaging.requirements import Requirement

            havePackaging = True
        except Exception:
            havePackaging = False

        for name, rawSpec in self.iterRequirements(requirementsPath):
            try:
                installedVer = version(name)
            except PackageNotFoundError:
                missing.append(name)
                continue

            if enforceVersions and havePackaging:
                try:
                    req = Requirement(f"{name}{rawSpec}")
                    if req.specifier and not req.specifier.contains(
                        installedVer, prereleases=True
                    ):
                        wrongVersion.append((name, installedVer, str(req.specifier)))
                except Exception:
                    # Ignore unparsable spec lines gracefully
                    pass

            # Optional: module importability check without importing heavy libs
            modName = moduleAliases.get(name, name.replace("-", "_"))
            if modName and find_spec(modName) is None:
                notImportable.append(modName)

        return missing, wrongVersion, notImportable

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
