import subprocess
import sys
import os
import logging
import src.constants as cs
import json
import hashlib
import re

from pathlib import Path
from typing import Iterable, Tuple
from src.utils.logAndPrint import logAndPrint
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec


KNOWN_MODULE_ALIASES = {
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "onnxruntime-gpu": "onnxruntime",
    "onnxruntime-directml": "onnxruntime",
    "onnxruntime-openvino": "onnxruntime",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "tensorrt": "tensorrt",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "triton-windows": "triton",
}


def getPythonExecutable() -> str:
    """
    Get the path to the current Python executable

    Returns:
        str: Path to the Python executable
    """
    return sys.executable


# https://github.com/pypa/hatch/blob/4ebce0e1fe8bf0fcdef587a704c207a063d72575/src/hatch/utils/platform.py#L176-L180
def streamProcessOutput(process) -> Iterable[str]:
    for line in iter(process.stdout.readline, b""):
        yield line.decode("utf-8", errors="replace")


def _getRuntimeRoot() -> Path:
    if cs.WHEREAMIRUNFROM:
        return Path(cs.WHEREAMIRUNFROM)
    return Path(__file__).resolve().parents[2]


def _resolveRequirementsPath(extension: str) -> Tuple[bool, str]:
    """Resolve a requirements file path based on the Python executable and runtime root."""
    pythonPath = getPythonExecutable()
    if not pythonPath:
        return False, "Failed to detect Python executable path"

    if extension and os.path.isabs(extension) and os.path.exists(extension):
        return True, extension

    candidate = os.path.join(os.path.dirname(pythonPath), extension)
    if extension and os.path.exists(candidate):
        return True, candidate

    candidate = os.path.join(_getRuntimeRoot(), extension)
    if extension and os.path.exists(candidate):
        return True, candidate

    return False, f"Requirements file not found: {candidate}"


def _versionSatisfiesRequirement(specifier: str, installedVersion: str) -> bool:
    try:
        from packaging.requirements import Requirement

        requirement = Requirement(specifier)
        if not requirement.specifier:
            return True
        return requirement.specifier.contains(installedVersion, prereleases=True)
    except Exception:
        return True


def _runPipCommand(pythonPath: str, args: list[str], action: str) -> Tuple[bool, str]:
    """Run pip with streamed output and return (success, message)."""
    try:
        logging.info("Using Python executable: %s", pythonPath)
        logging.info("Running pip action: %s", action)

        process = subprocess.Popen(
            [pythonPath, "-I", "-m", "pip", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if process.stdout is not None:
            for line in streamProcessOutput(process):
                print(line, end="")
                logging.info(line.strip())
            process.stdout.close()

        returnCode = process.wait()
        if returnCode != 0:
            errorMsg = f"Error {action} requirements (exit code: {returnCode})"
            logging.error(errorMsg)
            print(errorMsg)
            return False, errorMsg

        return True, f"Successfully {action}ed dependencies from requirements file"
    except Exception as e:
        errorMsg = f"Error {action} requirements: {str(e)}"
        logging.error(errorMsg)
        print(errorMsg)
        return False, errorMsg


def uninstallDependencies(extension: str = "") -> Tuple[bool, str]:
    """Uninstall dependencies from the selected requirements file.

    Args:
        extension (str): Requirements file name or absolute path.
    """
    pythonPath = getPythonExecutable()
    ok, requirementsPath = _resolveRequirementsPath(extension)
    if not ok:
        return False, requirementsPath

    logMessage = f"Uninstalling requirements from: {requirementsPath}"
    logging.info(logMessage)
    print(logMessage)

    return _runPipCommand(
        pythonPath,
        [
            "uninstall",
            "-y",
            "-r",
            requirementsPath,
            "--disable-pip-version-check",
        ],
        "uninstall",
    )


def installDependencies(extension: str = "", isNvidia: bool = True) -> Tuple[bool, str]:
    """
    Install dependencies from the selected requirements file.

    Returns:
        Tuple[bool, str]: Success status and message
    """
    pythonPath = getPythonExecutable()
    ok, requirementsPath = _resolveRequirementsPath(extension)
    if not ok:
        return False, requirementsPath

    logMessage = f"Installing requirements from: {requirementsPath}"
    logging.info(logMessage)
    print(logMessage)

    success, message = _runPipCommand(
        pythonPath,
        [
            "install",
            "-r",
            requirementsPath,
            "--no-warn-script-location",
            "--no-cache-dir",
            "--disable-pip-version-check",
        ],
        "install",
    )

    """
    if success and isNvidia:
        success, message = _installTensorRTRTX()
        if not success:
            logAndPrint(message, "red")
            raise RuntimeError(f"Failed to install TensorRT/RTX: {message}")
    """

    return success, message


class DependencyChecker:
    def __init__(self):
        self.cachePath = _getRuntimeRoot() / ".dependencyCache.json"
        self._cache = None
        self._requirementsHash = {}
        self.knownAliases = dict(KNOWN_MODULE_ALIASES)

    def needsUpdate(self, requirementsPath):
        """Check if dependencies need updating using hash and installed package checks."""
        cache = self._loadCache()
        currentHash = self._getFileHashCached(requirementsPath)

        if currentHash is None:
            logging.warning(
                f"Failed to hash requirements file; forcing update: {requirementsPath}"
            )
            return True

        if currentHash != cache.get("requirements_hash"):
            return True

        if cache.get("python_executable") != sys.executable:
            return True

        if cache.get("python_version") != sys.version:
            return True

        missing, wrongVersion, notImportable = self.checkRequirementsInstalled(
            requirementsPath,
            moduleAliases=self.knownAliases,
            enforceVersions=True,
        )

        if missing:
            logging.info(f"Missing distributions detected: {missing}")
            return True

        if wrongVersion:
            logging.info(f"Version mismatches detected: {wrongVersion}")
            return True

        if notImportable:
            logging.info(
                f"Distributions present but modules not importable via expected names: {notImportable}"
            )
            return True

        return False

    def updateCache(self, requirementsPath):
        """Update cache after successful installation."""
        cache = {
            "requirements_hash": self._getFileHashCached(requirementsPath),
            "python_executable": sys.executable,
            "python_version": sys.version,
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
        """
        if requirementsFile is None:
            from src.utils.isCudaInit import detectNVidiaGPU, detectGPUArchitecture

            isNvidia = detectNVidiaGPU()
            supportsCuda = False
            if isNvidia:
                supportsCuda, _, _ = detectGPUArchitecture()
            if cs.SYSTEM == "Windows":
                requirementsFile = (
                    "extra-requirements-windows.txt"
                    if supportsCuda
                    else "extra-requirements-windows-lite.txt"
                )
            else:
                requirementsFile = (
                    "extra-requirements-linux.txt"
                    if supportsCuda
                    else "extra-requirements-linux-lite.txt"
                )

        requirementsPath = os.path.join(_getRuntimeRoot(), requirementsFile)

        self.uninstallDeprecatedDependencies()

        logAndPrint("Forcing full dependency download...", "yellow")
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

        logAndPrint(message, "green")
        self.updateCache(requirementsPath)
        return True

    def iterRequirements(self, requirementsPath: str):
        """Yield (name, rawSpecifier) pairs from a requirements.txt file."""
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
                    match = re.match(r"^([A-Za-z0-9._-]+)", line)
                    if not match:
                        continue
                    name = match.group(1)
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
        """Return (missing, wrongVersion, notImportable) without importing packages."""
        moduleAliases = moduleAliases or {}
        missing: list[str] = []
        wrongVersion: list[tuple[str, str, str]] = []
        notImportable: list[str] = []

        for name, rawSpec in self.iterRequirements(requirementsPath):
            try:
                installedVer = version(name)
            except PackageNotFoundError:
                missing.append(name)
                continue

            if enforceVersions and rawSpec:
                fullSpecifier = f"{name}{rawSpec}"
                if not _versionSatisfiesRequirement(fullSpecifier, installedVer):
                    wrongVersion.append((name, installedVer, rawSpec))

            modName = moduleAliases.get(name, name.replace("-", "_"))
            if modName and find_spec(modName) is None:
                notImportable.append(modName)

        return missing, wrongVersion, notImportable

    def _loadCache(self):
        """Load cache from file with in-memory caching."""
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
        """Get MD5 hash of a file with caching."""
        if filepath in self._requirementsHash:
            return self._requirementsHash[filepath]

        fileHash = self._getFileHash(filepath)
        if fileHash is not None:
            self._requirementsHash[filepath] = fileHash
        return fileHash

    def _getFileHash(self, filepath):
        """Get MD5 hash of a file."""
        try:
            hashMd5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hashMd5.update(chunk)
            return hashMd5.hexdigest()
        except Exception as e:
            logging.warning(f"Failed to hash file {filepath}: {e}")
            return None

    def uninstallDeprecatedDependencies(self):
        """Uninstall dependencies listed in deprecated-requirements.txt when present."""
        deprecatedRequirementsFile = "deprecated-requirements.txt"
        deprecatedRequirementsPath = os.path.join(
            _getRuntimeRoot(), deprecatedRequirementsFile
        )

        if not os.path.exists(deprecatedRequirementsPath):
            logAndPrint(
                f"Deprecated requirements file not found: {deprecatedRequirementsPath}",
                "yellow",
            )
            return True

        logAndPrint("Uninstalling deprecated dependencies...", "yellow")
        success, message = uninstallDependencies(deprecatedRequirementsFile)

        if not success:
            logAndPrint(
                f"Failed to uninstall deprecated dependencies: {message}", "red"
            )
            return False

        logAndPrint("Successfully uninstalled deprecated dependencies", "green")
        return True


def uninstallDeprecatedDependenciesStandalone() -> Tuple[bool, str]:
    """
    Standalone function to uninstall dependencies from deprecated-requirements.txt.

    Returns:
        Tuple[bool, str]: Success status and message
    """
    checker = DependencyChecker()
    try:
        success = checker.uninstallDeprecatedDependencies()
        if success:
            return True, "Successfully uninstalled deprecated dependencies"
        return False, "Failed to uninstall deprecated dependencies"
    except Exception as e:
        errorMsg = f"Error during deprecated dependencies uninstall: {str(e)}"
        logging.error(errorMsg)
        return False, errorMsg


def _installTensorRTRTX() -> Tuple[bool, str]:
                    return self._requirementsHash[filepath]
