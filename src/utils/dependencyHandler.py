import subprocess
import sys
import os
import logging
import src.constants as cs
import json
import hashlib
import re
import shutil
import tempfile

from pathlib import Path
from typing import Tuple, Iterable
from src.utils.logAndPrint import logAndPrint
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec


UV_EXTRA_BY_REQUIREMENTS = {
    "extra-requirements-windows.txt": "runtime-windows-cuda",
    "extra-requirements-windows-lite.txt": "runtime-windows-lite",
    "extra-requirements-linux.txt": "runtime-linux-cuda",
    "extra-requirements-linux-lite.txt": "runtime-linux-lite",
}

UV_EXTRAS = set(UV_EXTRA_BY_REQUIREMENTS.values())


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


def _resolveRequirementsPath(extension: str) -> Tuple[bool, str]:
    """Resolve a requirements file path based on the Python executable and repo root."""
    pythonPath = getPythonExecutable()
    if not pythonPath:
        return False, "Failed to detect Python executable path"

    candidate = os.path.join(os.path.dirname(pythonPath), extension)
    if os.path.exists(candidate):
        return True, candidate

    candidate = os.path.join(cs.WHEREAMIRUNFROM, extension)
    if os.path.exists(candidate):
        return True, candidate

    return False, f"Requirements file not found: {candidate}"


def _resolveUvProfile(profileSpecifier: str) -> str | None:
    if not profileSpecifier:
        return None

    directMatch = profileSpecifier.strip()
    if directMatch in UV_EXTRAS:
        return directMatch

    baseName = os.path.basename(directMatch)
    return UV_EXTRA_BY_REQUIREMENTS.get(baseName)


def _getUvLockPath() -> Path:
    return Path(cs.WHEREAMIRUNFROM) / "uv.lock"


def _getPyprojectPath() -> Path:
    return Path(cs.WHEREAMIRUNFROM) / "pyproject.toml"


def _useUvManagedDependencies(profileSpecifier: str) -> bool:
    return (
        _resolveUvProfile(profileSpecifier) is not None
        and _getUvLockPath().exists()
        and _getPyprojectPath().exists()
    )


def _resolveUvExecutable(pythonPath: str) -> str:
    candidates = []
    pythonDir = Path(pythonPath).resolve().parent

    if os.name == "nt":
        candidates.extend([pythonDir / "uv.exe", Path(cs.WHEREAMIRUNFROM) / "uv.exe"])
    else:
        candidates.extend([pythonDir / "uv", Path(cs.WHEREAMIRUNFROM) / "uv"])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    uvExecutable = shutil.which("uv")
    if uvExecutable:
        return uvExecutable

    raise FileNotFoundError(
        "uv executable not found. Expected a bundled uv binary next to the TAS runtime or a system uv on PATH."
    )


def _getUvCommandEnv() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_PROJECT_ENVIRONMENT", os.path.join(cs.WHEREAMIRUNFROM, ".uv-runtime-env"))
    return env


def _resolveUvExtra(extension: str) -> str | None:
    return _resolveUvProfile(extension)


def _getUvProfileHash(profile: str) -> str:
    hashMd5 = hashlib.md5()
    for path in (_getPyprojectPath(), _getUvLockPath()):
        with open(path, "rb") as fileHandle:
            for chunk in iter(lambda: fileHandle.read(4096), b""):
                hashMd5.update(chunk)
    hashMd5.update(profile.encode("utf-8"))
    return hashMd5.hexdigest()


def _getExportedRequirementsPath(profileSpecifier: str) -> Tuple[str | None, str | None]:
    uvProfile = _resolveUvProfile(profileSpecifier)
    if not _useUvManagedDependencies(profileSpecifier) or uvProfile is None:
        ok, requirementsPath = _resolveRequirementsPath(profileSpecifier)
        return (requirementsPath, None) if ok else (None, requirementsPath)

    uvExecutable = _resolveUvExecutable(getPythonExecutable())
    exportedRequirements = _exportLockedRequirements(uvExecutable, uvProfile)
    return exportedRequirements, None


def _exportLockedRequirements(uvExecutable: str, extra: str | None) -> str:
    exportCommand = [
        uvExecutable,
        "export",
        "--directory",
        cs.WHEREAMIRUNFROM,
        "--locked",
        "--no-emit-project",
        "--format",
        "requirements.txt",
    ]

    if extra is not None:
        exportCommand.extend(["--extra", extra])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as exportedRequirements:
        subprocess.run(
            exportCommand,
            check=True,
            stdout=exportedRequirements,
            env=_getUvCommandEnv(),
        )
        return exportedRequirements.name


def _runUvSync(pythonPath: str, extra: str | None, action: str) -> Tuple[bool, str]:
    """Run uv pip sync with streamed log output and return (success, message)."""
    exportedRequirements = None
    try:
        uvExecutable = _resolveUvExecutable(pythonPath)
        exportedRequirements = _exportLockedRequirements(uvExecutable, extra)

        logging.info("Using Python executable: %s", pythonPath)
        logging.info("Using uv executable: %s", uvExecutable)
        logging.info("Running uv sync action: %s", action)

        syncCommand = [
            uvExecutable,
            "pip",
            "sync",
            exportedRequirements,
            "--python",
            pythonPath,
        ]

        process = subprocess.Popen(
            syncCommand,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_getUvCommandEnv(),
        )

        if process.stdout is not None:
            for line in streamProcessOutput(process):
                print(line, end="")
                logging.info(line.strip())
            process.stdout.close()

        returnCode = process.wait()

        if returnCode != 0:
            errorMsg = f"Error {action}ing locked dependencies (exit code: {returnCode})"
            logging.error(errorMsg)
            print(errorMsg)
            return False, errorMsg

        targetDescription = extra or "base environment"
        return True, f"Successfully synced locked dependencies for {targetDescription}"
    except Exception as e:
        errorMsg = f"Error {action}ing locked dependencies: {str(e)}"
        logging.error(errorMsg)
        print(errorMsg)
        return False, errorMsg
    finally:
        if exportedRequirements and os.path.exists(exportedRequirements):
            try:
                os.remove(exportedRequirements)
            except OSError:
                logging.warning(
                    "Failed to remove temporary exported requirements file: %s",
                    exportedRequirements,
                )


def uninstallDependencies(extension: str = "") -> Tuple[bool, str]:
    """Reset the runtime environment back to the locked base dependency set.
    Args:
        extension (str): Legacy profile specifier retained for compatibility."""

    pythonPath = getPythonExecutable()

    logMessage = (
        "Resetting runtime environment to the locked base dependency set"
    )
    logging.info(logMessage)
    print(logMessage)

    return _runUvSync(pythonPath, None, "uninstall")


def installDependencies(extension: str = "", isNvidia: bool = True) -> Tuple[bool, str]:
    """
    Sync a locked runtime dependency profile into the current environment.

    Returns:
        Tuple[bool, str]: Success status and message
    """
    pythonPath = getPythonExecutable()

    uvExtra = _resolveUvExtra(extension)
    if uvExtra is None:
        return False, f"Unsupported dependency profile for uv sync: {extension}"

    logMessage = f"Syncing locked runtime profile {uvExtra}"
    logging.info(logMessage)
    print(logMessage)

    success, message = _runUvSync(pythonPath, uvExtra, "install")

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
        self.cachePath = Path(cs.WHEREAMIRUNFROM) / ".dependencyCache.json"
        self._cache = None
        self._requirementsHash = {}
        self.knownAliases = {
            "opencv-python": "cv2",
            "opencv-python-headless": "cv2",
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

        uvProfile = _resolveUvProfile(requirementsPath)
        exportedRequirements = None

        if _useUvManagedDependencies(requirementsPath) and uvProfile is not None:
            currentHash = _getUvProfileHash(uvProfile)

            if currentHash != cache.get("requirements_hash"):
                return True

            if cache.get("dependency_profile") != uvProfile:
                return True

            if cache.get("python_executable") != sys.executable:
                return True

            if cache.get("python_version") != sys.version:
                return True

            try:
                uvExecutable = _resolveUvExecutable(getPythonExecutable())
                exportedRequirements = _exportLockedRequirements(uvExecutable, uvProfile)
                missing, wrongVersion, notImportable = self.checkRequirementsInstalled(
                    exportedRequirements,
                    moduleAliases=self.knownAliases,
                    enforceVersions=False,
                )
            finally:
                if exportedRequirements and os.path.exists(exportedRequirements):
                    try:
                        os.remove(exportedRequirements)
                    except OSError:
                        logging.warning(
                            "Failed to remove temporary exported requirements file: %s",
                            exportedRequirements,
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

        currentHash = self._getFileHashCached(requirementsPath)

        if currentHash is None:
            logging.warning(
                f"Failed to hash requirements file; forcing update: {requirementsPath}"
            )
            return True

        # If hash differs, definitely need update
        if currentHash != cache.get("requirements_hash"):
            return True

        if cache.get("python_executable") != sys.executable:
            return True

        if cache.get("python_version") != sys.version:
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
        uvProfile = _resolveUvProfile(requirementsPath)
        requirementsHash = (
            _getUvProfileHash(uvProfile)
            if _useUvManagedDependencies(requirementsPath) and uvProfile is not None
            else self._getFileHashCached(requirementsPath)
        )

        cache = {
            "requirements_hash": requirementsHash,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "dependency_profile": uvProfile,
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

        requirementsPath = requirementsFile

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
        """Yield (name, rawSpecifier) pairs from an exported requirements file, ignoring comments, includes, and URLs."""
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
        if file_hash is not None:
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
        Reset the runtime environment before syncing the selected locked profile.
        """
        logAndPrint(
            "Resetting environment before syncing the selected runtime profile...",
            "yellow",
        )
        success, message = uninstallDependencies()

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
    Standalone helper to reset the runtime environment before a locked sync.

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
