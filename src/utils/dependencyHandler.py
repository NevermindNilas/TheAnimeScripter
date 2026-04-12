import subprocess
import sys
import os
import logging
import src.constants as cs
import json
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

DEPENDENCY_PROFILE_REQUIREMENTS = {
    "windows-cuda": "extra-requirements-windows.txt",
    "windows-lite": "extra-requirements-windows-lite.txt",
    "linux-cuda": "extra-requirements-linux.txt",
    "linux-lite": "extra-requirements-linux-lite.txt",
}


def getDependencyProfile(systemName: str, supportsCuda: bool) -> str:
    normalizedSystem = "windows" if systemName.lower() == "windows" else "linux"
    return f"{normalizedSystem}-cuda" if supportsCuda else f"{normalizedSystem}-lite"


def getRequirementsFileForProfile(profile: str) -> str:
    normalizedProfile = profile.strip().lower()
    requirementsFile = DEPENDENCY_PROFILE_REQUIREMENTS.get(normalizedProfile)
    if requirementsFile is None:
        validProfiles = ", ".join(DEPENDENCY_PROFILE_REQUIREMENTS)
        raise ValueError(
            f"Unsupported dependency profile '{profile}'. Expected one of: {validProfiles}"
        )
    return requirementsFile


def getPythonExecutable() -> str:
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
    """Uninstall dependencies from the selected requirements file."""
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
    """Install dependencies from the selected requirements file."""
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

    return success, message


class DependencyChecker:
    def __init__(self):
        self.cachePath = _getRuntimeRoot() / ".dependencyCache.json"
        self._cache = None
        self.knownAliases = dict(KNOWN_MODULE_ALIASES)

    def loadStoredProfile(self) -> str | None:
        """Return the profile the user previously installed, or None."""
        cache = self._loadCache()
        return cache.get("profile")

    def storeProfile(self, profile: str):
        """Persist the user's chosen profile."""
        try:
            with open(self.cachePath, "w") as f:
                json.dump({"profile": profile}, f)
            self._cache = {"profile": profile}
        except Exception as e:
            logging.warning(f"Failed to store dependency profile: {e}")

    def clearCache(self):
        """Delete the cache file (used by --cleanup)."""
        try:
            if self.cachePath.exists():
                self.cachePath.unlink()
            self._cache = None
        except Exception as e:
            logging.warning(f"Failed to clear dependency cache: {e}")

    def ensureDependencies(self) -> bool:
        """Check that the stored profile's dependencies are installed.

        - No stored profile → prompt + install + store.
        - Stored profile with missing packages → prompt + install + store.
        - Everything present → return True immediately.

        Returns True if dependencies are satisfied after this call.
        """
        storedProfile = self.loadStoredProfile()

        if storedProfile is None:
            return self._promptInstallAndStore()

        try:
            requirementsFile = getRequirementsFileForProfile(storedProfile)
        except ValueError:
            logging.warning(f"Stored profile '{storedProfile}' is invalid, re-prompting")
            self.clearCache()
            return self._promptInstallAndStore()

        requirementsPath = os.path.join(_getRuntimeRoot(), requirementsFile)
        if not os.path.exists(requirementsPath):
            logging.warning(f"Requirements file missing: {requirementsPath}")
            return False

        missing, _, notImportable = self.checkRequirementsInstalled(
            requirementsPath,
            moduleAliases=self.knownAliases,
        )

        if not missing and not notImportable:
            return True

        logAndPrint(
            "Missing dependencies detected: "
            + ", ".join(missing + notImportable),
            "yellow",
        )
        return self._promptInstallAndStore()

    def installProfile(self, profile: str) -> bool:
        """Install a specific profile's requirements and store the choice."""
        requirementsFile = getRequirementsFileForProfile(profile)

        success, message = installDependencies(
            requirementsFile,
            isNvidia=profile.endswith("-cuda"),
        )

        if not success:
            logAndPrint(message, "red")
            return False

        self.storeProfile(profile)

        import importlib
        importlib.invalidate_caches()

        logAndPrint(
            "Dependencies installed successfully, continuing execution.",
            "green",
        )
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

    def _promptInstallAndStore(self) -> bool:
        """Show the inquirer prompt, install, and store the chosen profile."""
        from src.utils.argumentsChecker import _promptDownloadRequirementsSelection

        try:
            selectedProfile = _promptDownloadRequirementsSelection()
        except Exception:
            logAndPrint("Could not launch dependency installer.", "red")
            return False

        return self.installProfile(selectedProfile)

    def _loadCache(self):
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
