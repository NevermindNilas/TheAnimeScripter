import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from importlib import invalidate_caches
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

import src.constants as cs
from src.infra.logAndPrint import logAndPrint

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
    "nvidia-vfx": "nvvfx",
}

DEPENDENCY_PROFILE_REQUIREMENTS = {
    "windows-cuda": "extra-requirements-windows.txt",
    "windows-lite": "extra-requirements-windows-lite.txt",
    "linux-cuda": "extra-requirements-linux.txt",
    "linux-lite": "extra-requirements-linux-lite.txt",
    "linux-rocm": "extra-requirements-linux-rocm.txt",
    "macos-lite": "extra-requirements-macos-lite.txt",
    "macos-mps": "extra-requirements-macos.txt",
}

# The AE/Premiere panels and the Standalone app spawn TAS with no terminal. On
# Windows the dependency picker reads keys from the (hidden) console and so used
# to wait forever there; on macOS/Linux it failed at once and the run went on.
# A run that cannot work without an install and has no terminal to ask in now
# prints this marker and exits with this code; the hosts match the marker and
# open their own installer.
DEPENDENCIES_MISSING_MARKER = "TAS_DEPENDENCIES_MISSING"
DEPENDENCIES_MISSING_EXIT_CODE = 78  # EX_CONFIG

# Packages no TAS run can work without. A missing optional package (yt-dlp,
# vmaf-torch, ...) only breaks the runs that use it, so a run with no terminal
# to reinstall from warns and carries on, as it always did on macOS and Linux.
CORE_PACKAGES = frozenset({"torch", "torchvision"})


class DependenciesMissingError(RuntimeError):
    """The runtime needs a dependency install and nobody can be asked which."""


def canPromptInteractively() -> bool:
    """Whether a person at a terminal can answer the dependency picker.

    isatty() alone is not enough on Windows, where the picker reads its keys
    from the console and would wait forever on one nobody can see or type in:
    - NUL is a character device, so stdin ignored by the host (Node's stdio
      "ignore", subprocess.DEVNULL, `< NUL`) reads as a terminal;
    - a wrapper that captures stdout leaves the picker drawn into a pipe;
    - a console created with CREATE_NO_WINDOW (Task Scheduler, launchers) has
      no window at all.
    So stdin and stdout must both be the console, and the console must have a
    window (ConPTY / Windows Terminal report a pseudo-window, which counts). A
    window that exists but is hidden still counts: that cannot be told apart
    from a terminal the user minimised. A terminal run whose stdout is piped
    (`| tee`) no longer gets the picker and is told to use
    --download_requirements instead.
    """
    stdin = sys.stdin
    try:
        if stdin is None or not stdin.isatty():
            return False
        if sys.platform != "win32":
            return True

        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetConsoleMode.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.GetConsoleWindow.argtypes = []
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        mode = wintypes.DWORD()
        for stream in (stdin, sys.stdout):
            if stream is None:
                return False
            handle = msvcrt.get_osfhandle(stream.fileno())
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
        return bool(kernel32.GetConsoleWindow())
    except AttributeError, OSError, ValueError:
        return False


def _missingCorePackages() -> list[str]:
    """CORE_PACKAGES that are not installed or not importable."""
    missing = []
    for name in sorted(CORE_PACKAGES):
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if find_spec(name) is None:
            missing.append(name)
    return missing


def getDependencyProfile(
    systemName: str, supportsCuda: bool, isRocm: bool = False
) -> str:
    normalized = systemName.lower()
    if normalized == "windows":
        return "windows-cuda" if supportsCuda else "windows-lite"
    if normalized == "darwin":
        return "macos-mps"
    if isRocm:
        return "linux-rocm"
    return "linux-cuda" if supportsCuda else "linux-lite"


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


def _resolveRequirementsPath(extension: str) -> tuple[bool, str]:
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


def _runPipCommand(pythonPath: str, args: list[str], action: str) -> tuple[bool, str]:
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


def uninstallDependencies(extension: str = "") -> tuple[bool, str]:
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


def installDependencies(extension: str = "") -> tuple[bool, str]:
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

    def storeProfile(self, profile: str) -> bool:
        """Persist the user's chosen profile. Returns False if it could not."""
        try:
            with open(self.cachePath, "w") as f:
                json.dump({"profile": profile}, f)
            self._cache = {"profile": profile}
            return True
        except Exception as e:
            logging.warning(f"Failed to store dependency profile: {e}")
            return False

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

        Returns True if dependencies are satisfied after this call. With no
        terminal to prompt in, raises DependenciesMissingError when a
        CORE_PACKAGES package is missing or (on Windows) no usable profile is
        recorded; anything less is warned about and returns False so the run
        carries on, as it always did on macOS/Linux.
        """
        storedProfile = self.loadStoredProfile()

        if storedProfile is None:
            return self._recordMissing("No dependency profile is installed")

        try:
            requirementsFile = getRequirementsFileForProfile(storedProfile)
        except ValueError:
            logging.warning(
                f"Stored profile '{storedProfile}' is invalid, re-prompting"
            )
            return self._recordMissing(
                f"The installed dependency profile '{storedProfile}' is not recognised",
                clearInvalid=True,
            )

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

        missingList = ", ".join(missing + notImportable)
        logAndPrint(f"Missing dependencies detected: {missingList}", "yellow")
        if not canPromptInteractively() and CORE_PACKAGES.isdisjoint(
            missing + notImportable
        ):
            return False
        return self._promptInstallAndStore(f"Missing dependencies: {missingList}")

    def installProfile(self, profile: str) -> bool:
        """Install a specific profile's requirements and store the choice."""
        requirementsFile = getRequirementsFileForProfile(profile)

        success, message = installDependencies(requirementsFile)

        if not success:
            logAndPrint(message, "red")
            return False

        # The hosts read this record as "the install finished"; without it every
        # later run would ask for a profile again.
        if not self.storeProfile(profile):
            logAndPrint(
                f"Dependencies installed, but the profile record could not be "
                f"written to {self.cachePath}.",
                "red",
            )
            return False

        invalidate_caches()

        logAndPrint(
            "Dependencies installed successfully, continuing execution.",
            "green",
        )
        return True

    def iterRequirements(self, requirementsPath: str):
        """Yield (name, rawSpecifier) pairs from a requirements.txt file."""
        try:
            with open(requirementsPath, encoding="utf-8") as f:
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

    def _recordMissing(self, reason: str, clearInvalid: bool = False) -> bool:
        """No usable profile record: prompt at a terminal. Without one, stop.

        Except on macOS/Linux with the core packages importable: a from-source
        install there never writes the record and such runs always carried on.
        On Windows the record is written by every host install, so its absence
        means the install never finished, even when pip got as far as torch.
        """
        if canPromptInteractively():
            if clearInvalid:
                self.clearCache()
            return self._promptInstallAndStore(reason)

        missingCore = _missingCorePackages()
        if missingCore:
            raise DependenciesMissingError(
                f"{reason}, and core packages are missing: {', '.join(missingCore)}"
            )
        if sys.platform == "win32":
            raise DependenciesMissingError(reason)
        logAndPrint(
            f"{reason}; continuing with the packages already installed.", "yellow"
        )
        return False

    def _promptInstallAndStore(self, reason: str) -> bool:
        """Show the inquirer prompt, install, and store the chosen profile.

        Raises DependenciesMissingError, carrying `reason`, when there is no
        terminal to show the prompt in.
        """
        if not canPromptInteractively():
            raise DependenciesMissingError(reason)

        from src.cli.startup import _promptDownloadRequirementsSelection

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
            with open(self.cachePath) as f:
                self._cache = json.load(f)
            return self._cache
        except Exception as e:
            logging.warning(f"Failed to load dependency cache: {e}")
            self._cache = {}
            return self._cache
