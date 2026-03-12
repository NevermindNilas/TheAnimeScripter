import subprocess
import sys
import os
import logging
import src.constants as cs
import json
import hashlib
import re
import tomllib
import shutil

from pathlib import Path
from typing import Tuple, Iterable
from src.utils.logAndPrint import logAndPrint
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec


UV_RUNTIME_WINDOWS_CUDA = "runtime-windows-cuda"
UV_RUNTIME_WINDOWS_LITE = "runtime-windows-lite"
UV_RUNTIME_LINUX_CUDA = "runtime-linux-cuda"
UV_RUNTIME_LINUX_LITE = "runtime-linux-lite"

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

UV_EXTRA_BY_REQUIREMENTS = {
    "extra-requirements-windows.txt": UV_RUNTIME_WINDOWS_CUDA,
    "extra-requirements-windows-lite.txt": UV_RUNTIME_WINDOWS_LITE,
    "extra-requirements-linux.txt": UV_RUNTIME_LINUX_CUDA,
    "extra-requirements-linux-lite.txt": UV_RUNTIME_LINUX_LITE,
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


def _resolveUvProfile(profileSpecifier: str) -> str | None:
    if not profileSpecifier:
        return None

    directMatch = profileSpecifier.strip()
    if directMatch in UV_EXTRAS:
        return directMatch

    baseName = os.path.basename(directMatch)
    return UV_EXTRA_BY_REQUIREMENTS.get(baseName)


def _getUvLockPath() -> Path:
    return _getRuntimeRoot() / "uv.lock"


def _getPyprojectPath() -> Path:
    return _getRuntimeRoot() / "pyproject.toml"


def _getRuntimeRoot() -> Path:
    if cs.WHEREAMIRUNFROM:
        return Path(cs.WHEREAMIRUNFROM)
    return Path(__file__).resolve().parents[2]


def _resolveUvExtra(extension: str) -> str | None:
    return _resolveUvProfile(extension)


def _resolveUvExecutable(pythonPath: str) -> str:
    candidates = []
    pythonDir = Path(pythonPath).resolve().parent

    if os.name == "nt":
        candidates.extend(
            [
                pythonDir / "uv.exe",
                pythonDir / "uv.EXE",
                _getRuntimeRoot() / "uv.exe",
                _getRuntimeRoot() / "uv.EXE",
            ]
        )
    else:
        candidates.extend([pythonDir / "uv", _getRuntimeRoot() / "uv"])

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
    return os.environ.copy()


def _loadPyprojectMetadata() -> dict:
    with open(_getPyprojectPath(), "rb") as fileHandle:
        return tomllib.load(fileHandle)


def _requirementAppliesToCurrentEnvironment(specifier: str) -> bool:
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement

        requirement = Requirement(specifier)
        if requirement.marker is None:
            return True
        return requirement.marker.evaluate(default_environment())
    except Exception:
        if ";" not in specifier:
            return True

        marker = specifier.split(";", 1)[1].strip().lower()
        if "platform_system" not in marker:
            return True
        if "windows" in marker:
            return cs.SYSTEM == "Windows"
        if "linux" in marker:
            return cs.SYSTEM == "Linux"
        return True


def _getSelectedDependencySpecs(profileSpecifier: str) -> list[str]:
    metadata = _loadPyprojectMetadata()
    projectMetadata = metadata.get("project", {})

    dependencySpecs = list(projectMetadata.get("dependencies", []))
    profile = _resolveUvProfile(profileSpecifier)
    if profile is not None:
        dependencySpecs.extend(
            projectMetadata.get("optional-dependencies", {}).get(profile, [])
        )

    return [
        specifier
        for specifier in dependencySpecs
        if _requirementAppliesToCurrentEnvironment(specifier)
    ]


def _getDependencySelectionHash(profile: str | None) -> str:
    hashMd5 = hashlib.md5()
    with open(_getPyprojectPath(), "rb") as fileHandle:
        for chunk in iter(lambda: fileHandle.read(4096), b""):
            hashMd5.update(chunk)
    hashMd5.update((profile or "base").encode("utf-8"))
    return hashMd5.hexdigest()


def _parseRequirementName(specifier: str) -> str | None:
    try:
        from packaging.requirements import Requirement

        return Requirement(specifier).name
    except Exception:
        match = re.match(r"^([A-Za-z0-9._-]+)", specifier)
        return match.group(1) if match else None


def _versionSatisfiesRequirement(specifier: str, installedVersion: str) -> bool:
    try:
        from packaging.requirements import Requirement

        requirement = Requirement(specifier)
        if not requirement.specifier:
            return True
        return requirement.specifier.contains(installedVersion, prereleases=True)
    except Exception:
        return True


def _appendUnique(items: list[str], value: str):
    if value not in items:
        items.append(value)


def _stripRequirementMarker(specifier: str) -> str:
    return specifier.split(";", 1)[0].strip()


def _getUvExtraIndexUrls(
    profile: str | None, dependencySpecs: list[str]
) -> list[str]:
    if profile is None:
        return []

    metadata = _loadPyprojectMetadata()
    uvMetadata = metadata.get("tool", {}).get("uv", {})
    indexes = {
        indexEntry.get("name"): indexEntry.get("url")
        for indexEntry in uvMetadata.get("index", [])
        if indexEntry.get("name") and indexEntry.get("url")
    }

    requiredPackages = {
        _parseRequirementName(specifier)
        for specifier in dependencySpecs
        if _parseRequirementName(specifier)
    }

    extraIndexUrls: list[str] = []
    for packageName in requiredPackages:
        sourceEntries = uvMetadata.get("sources", {}).get(packageName, [])
        if isinstance(sourceEntries, dict):
            sourceEntries = [sourceEntries]

        for sourceEntry in sourceEntries:
            if sourceEntry.get("extra") != profile:
                continue

            indexUrl = indexes.get(sourceEntry.get("index"))
            if indexUrl:
                _appendUnique(extraIndexUrls, indexUrl)

    return extraIndexUrls


def _collectPendingDependencySpecs(
    dependencySpecs: list[str], moduleAliases: dict[str, str] | None = None
) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
    moduleAliases = moduleAliases or {}
    pendingSpecs: list[str] = []
    versionMismatches: list[tuple[str, str, str]] = []
    notImportable: list[str] = []

    for specifier in dependencySpecs:
        requirementName = _parseRequirementName(specifier)
        if not requirementName:
            continue

        try:
            installedVersion = version(requirementName)
        except PackageNotFoundError:
            _appendUnique(pendingSpecs, specifier)
            continue

        if not _versionSatisfiesRequirement(specifier, installedVersion):
            versionMismatches.append((requirementName, installedVersion, specifier))
            _appendUnique(pendingSpecs, specifier)
            continue

        moduleName = moduleAliases.get(requirementName, requirementName.replace("-", "_"))
        if moduleName and find_spec(moduleName) is None:
            notImportable.append(moduleName)
            _appendUnique(pendingSpecs, specifier)

    return pendingSpecs, versionMismatches, notImportable


def _runUvInstall(
    pythonPath: str, dependencySpecs: list[str], extra: str | None
) -> Tuple[bool, str]:
    try:
        uvExecutable = _resolveUvExecutable(pythonPath)
        installSpecs = [_stripRequirementMarker(specifier) for specifier in dependencySpecs]
        extraIndexUrls = _getUvExtraIndexUrls(extra, dependencySpecs)

        installCommand = [
            uvExecutable,
            "pip",
            "install",
            "--python",
            pythonPath,
            *installSpecs,
        ]

        for indexUrl in extraIndexUrls:
            installCommand.extend(["--extra-index-url", indexUrl])

        logging.info("Using Python executable: %s", pythonPath)
        logging.info("Using uv executable: %s", uvExecutable)
        logging.info(
            "Running uv pip install for %d dependency specifiers", len(dependencySpecs)
        )

        process = subprocess.Popen(
            installCommand,
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
            errorMsg = f"Error installing dependencies (exit code: {returnCode})"
            logging.error(errorMsg)
            print(errorMsg)
            return False, errorMsg

        return True, "Successfully installed runtime dependencies"
    except Exception as e:
        errorMsg = f"Error installing dependencies: {str(e)}"
        logging.error(errorMsg)
        print(errorMsg)
        return False, errorMsg


def uninstallDependencies(extension: str = "") -> Tuple[bool, str]:
    """Disable destructive runtime cleanup for shared Python environments.
    Args:
        extension (str): Legacy profile specifier retained for compatibility."""

    cachePath = _getRuntimeRoot() / ".dependencyCache.json"
    if cachePath.exists():
        try:
            cachePath.unlink()
        except OSError as e:
            return False, f"Failed to clear dependency cache: {e}"

    logMessage = "Dependency cleanup is disabled; cleared TAS dependency cache only"
    logging.info(logMessage)
    print(logMessage)
    return True, logMessage


def installDependencies(extension: str = "", isNvidia: bool = True) -> Tuple[bool, str]:
    """
    Ensure the selected runtime dependencies exist in the current environment.

    Returns:
        Tuple[bool, str]: Success status and message
    """
    pythonPath = getPythonExecutable()

    dependencySet = _resolveUvExtra(extension)
    if dependencySet is None:
        return False, f"Unsupported runtime dependency set: {extension}"

    dependencySpecs = _getSelectedDependencySpecs(extension)
    pendingSpecs, versionMismatches, notImportable = _collectPendingDependencySpecs(
        dependencySpecs,
        moduleAliases=KNOWN_MODULE_ALIASES,
    )

    if not pendingSpecs:
        return True, f"Runtime dependencies already satisfied for {dependencySet}"

    logMessage = (
        f"Installing {len(pendingSpecs)} runtime dependencies for {dependencySet}"
    )
    logging.info(logMessage)
    print(logMessage)

    if versionMismatches:
        logging.info("Version mismatches detected before install: %s", versionMismatches)
    if notImportable:
        logging.info("Broken imports detected before install: %s", notImportable)

    success, message = _runUvInstall(pythonPath, pendingSpecs, dependencySet)

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
        """Check if dependencies need updating using hash + no-import presence checks"""
        cache = self._loadCache()

        uvProfile = _resolveUvProfile(requirementsPath)

        if uvProfile is not None and _getPyprojectPath().exists():
            currentHash = _getDependencySelectionHash(uvProfile)

            if currentHash != cache.get("requirements_hash"):
                return True

            if cache.get("dependency_profile") != uvProfile:
                return True

            if cache.get("python_executable") != sys.executable:
                return True

            if cache.get("python_version") != sys.version:
                return True

            missing, wrongVersion, notImportable = self.checkRequirementsInstalled(
                dependencySpecs=_getSelectedDependencySpecs(requirementsPath),
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
        """Update cache after successful installation"""
        uvProfile = _resolveUvProfile(requirementsPath)
        requirementsHash = (
            _getDependencySelectionHash(uvProfile)
            if uvProfile is not None and _getPyprojectPath().exists()
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
                    UV_RUNTIME_WINDOWS_CUDA if supportsCuda else UV_RUNTIME_WINDOWS_LITE
                )
            else:
                requirementsFile = (
                    UV_RUNTIME_LINUX_CUDA if supportsCuda else UV_RUNTIME_LINUX_LITE
                )

        requirementsPath = requirementsFile

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

    def iterRequirements(
        self,
        requirementsPath: str | None = None,
        dependencySpecs: list[str] | None = None,
    ):
        """Yield (name, rawSpecifier) pairs from requirement specifiers or requirements files."""
        if dependencySpecs is not None:
            for specifier in dependencySpecs:
                name = _parseRequirementName(specifier)
                if not name:
                    continue

                try:
                    from packaging.requirements import Requirement

                    requirement = Requirement(specifier)
                    yield requirement.name, str(requirement.specifier)
                except Exception:
                    yield name, specifier[len(name) :].strip()
            return

        if requirementsPath is None:
            return

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
        requirementsPath: str | None = None,
        dependencySpecs: list[str] | None = None,
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

        for name, rawSpec in self.iterRequirements(
            requirementsPath, dependencySpecs=dependencySpecs
        ):
            try:
                installedVer = version(name)
            except PackageNotFoundError:
                missing.append(name)
                continue

            if enforceVersions and rawSpec:
                fullSpecifier = f"{name}{rawSpec}" if not dependencySpecs else next(
                    (
                        specifier
                        for specifier in dependencySpecs
                        if _parseRequirementName(specifier) == name
                    ),
                    f"{name}{rawSpec}",
                )
                if not _versionSatisfiesRequirement(fullSpecifier, installedVer):
                    wrongVersion.append((name, installedVer, rawSpec))

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
        Cleanup no longer prunes the shared Python environment.
        """
        logAndPrint(
            "Dependency cleanup is disabled for shared Python environments; clearing cache only...",
            "yellow",
        )
        success, message = uninstallDependencies()

        if not success:
            logAndPrint(
                f"Failed to clear dependency cache: {message}", "red"
            )
            return False
        else:
            logAndPrint("Successfully cleared dependency cache", "green")
            return True


def uninstallDeprecatedDependenciesStandalone() -> Tuple[bool, str]:
    """
    Standalone helper for non-destructive dependency cleanup.

    Returns:
        Tuple[bool, str]: Success status and message
    """
    checker = DependencyChecker()
    try:
        success = checker.uninstallDeprecatedDependencies()
        if success:
            return True, "Successfully cleared dependency cache"
        else:
            return False, "Failed to clear dependency cache"
    except Exception as e:
        errorMsg = f"Error while clearing dependency cache: {str(e)}"
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
