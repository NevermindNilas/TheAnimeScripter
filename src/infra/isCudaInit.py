import glob
import logging
import os
import platform
import shutil
import subprocess
from functools import lru_cache


class CudaChecker:
    def __init__(self):
        """
        Checks available torch accelerators (CUDA, then MPS on Apple Silicon)
        and exposes a unified .device property. Falls back to CPU otherwise.

        Note: This class checks accelerator availability in PyTorch, but does
        not validate CUDA GPU architecture compatibility. Use
        detectGPUArchitecture() to check for Pascal or older GPUs.
        """
        # Lazify the import
        import torch

        self.torch = torch
        self.logging = logging

        try:
            self._cuda_available = self.torch.cuda.is_available()
        except Exception as e:
            self._cuda_available = False
            self.logging.warning(f"CUDA is not available: {e}")

        self._mps_available = False
        if not self._cuda_available:
            try:
                mpsBackend = getattr(self.torch.backends, "mps", None)
                if mpsBackend is not None and mpsBackend.is_available():
                    self._mps_available = bool(mpsBackend.is_built())
            except Exception as e:
                self._mps_available = False
                self.logging.warning(f"MPS is not available: {e}")

        if self._cuda_available:
            self.enableCudaOptimizations()

    @property
    def cudaAvailable(self):
        return self._cuda_available

    def enableCudaOptimizations(self):
        self.torch.backends.cudnn.benchmark = False
        self.torch.backends.cudnn.enabled = True

    @property
    def device(self):
        if self._cuda_available:
            return self.torch.device("cuda")
        if self._mps_available:
            return self.torch.device("mps")
        return self.torch.device("cpu")

    @property
    def deviceCount(self):
        """Get the number of available accelerator devices."""
        if self._cuda_available:
            return self.torch.cuda.device_count()
        if self._mps_available:
            return 1
        return 0


def getNvsmipaths():
    paths = [shutil.which("nvidia-smi")]
    systemRoot = os.environ.get("SystemRoot", r"C:\\Windows")
    programFiles = os.environ.get("ProgramFiles", r"C:\\Program Files")
    programFilesX86 = os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")
    paths.append(os.path.join(systemRoot, "System32", "nvidia-smi.exe"))
    paths.append(
        os.path.join(programFiles, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
    )
    paths.append(
        os.path.join(programFilesX86, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
    )
    seen = set()
    ordered = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


@lru_cache(maxsize=1)
def queryNvidiaSmiGpus():
    """
    Run nvidia-smi once and cache the result for the whole process.

    A single query returns both presence and architecture info, so callers
    (detectNVidiaGPU + detectGPUArchitecture, possibly across multiple
    startup phases) share one subprocess spawn instead of each spawning their
    own.

    Returns:
        tuple[tuple[str, str], ...]: (name, compute_cap) per GPU, empty if
        nvidia-smi is absent or fails.
    """
    smiPath = next((p for p in getNvsmipaths() if p and os.path.exists(p)), None)
    if smiPath is None:
        return ()
    try:
        result = subprocess.run(
            [smiPath, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as _e:
        logging.info("nvidia-smi not found or failed to run")
        return ()
    if result.returncode != 0 or not result.stdout.strip():
        return ()
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            gpus.append((parts[0], parts[1]))
    return tuple(gpus)


def _loadNvml():
    """Load the NVIDIA Management Library shipped with the driver, or None."""
    import ctypes

    candidates = [
        "nvml.dll",
        os.path.join(
            os.environ.get("ProgramFiles", ""),
            "NVIDIA Corporation",
            "NVSMI",
            "nvml.dll",
        ),
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def queryNvmlGpus():
    """
    Query GPUs through NVML directly (no subprocess) for name + compute_cap.

    nvml.dll/libnvidia-ml ships with the NVIDIA driver, so this needs neither
    torch nor nvidia-smi on PATH and answers in ~one nvmlInit (much faster than
    spawning nvidia-smi, and pre-torch). Returns the same shape as
    queryNvidiaSmiGpus: (name, compute_cap) per GPU, empty on any failure (e.g.
    no NVIDIA driver -> nvml absent).
    """
    import ctypes

    lib = _loadNvml()
    if lib is None:
        return ()
    try:
        if lib.nvmlInit_v2() != 0:
            return ()
    except Exception:
        return ()
    try:
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return ()
        gpus = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) != 0:
                continue
            buf = ctypes.create_string_buffer(96)
            name = ""
            if lib.nvmlDeviceGetName(handle, buf, 96) == 0:
                name = buf.value.decode(errors="replace")
            major = ctypes.c_int()
            minor = ctypes.c_int()
            cap = "unknown"
            if (
                lib.nvmlDeviceGetCudaComputeCapability(
                    handle, ctypes.byref(major), ctypes.byref(minor)
                )
                == 0
            ):
                cap = f"{major.value}.{minor.value}"
            if name:
                gpus.append((name, cap))
        return tuple(gpus)
    except Exception:
        return ()
    finally:
        try:
            lib.nvmlShutdown()
        except Exception:
            pass


def queryGpus():
    """NVML first (no subprocess, pre-torch); nvidia-smi as fallback."""
    gpus = queryNvmlGpus()
    if gpus:
        return gpus
    return queryNvidiaSmiGpus()


def listWindowsGpuNames():
    """All display-adapter names from the registry (any vendor, no WMI)."""
    if platform.system().lower() != "windows":
        return []
    import winreg

    base = (
        r"SYSTEM\CurrentControlSet\Control\Class"
        r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
    )
    names = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as key:
                        desc, _ = winreg.QueryValueEx(key, "DriverDesc")
                        names.append(desc)
                except OSError:
                    continue
    except OSError:
        return []
    return names


def checkWindowsAdapters():
    """True if any display adapter is NVIDIA (registry scan, no WMI/subprocess)."""
    return any("nvidia" in name.lower() for name in listWindowsGpuNames())


def checkLinuxPci():
    vendorPaths = glob.glob("/sys/bus/pci/devices/*/vendor")
    for vendorPath in vendorPaths:
        try:
            with open(vendorPath, encoding="utf-8") as handle:
                if handle.read().strip().lower() == "0x10de":
                    return True
        except OSError:
            continue
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and "nvidia" in result.stdout.lower():
            return True
    except (subprocess.SubprocessError, FileNotFoundError) as _e:
        pass
    return False


def parseComputeCapability(value):
    try:
        parts = value.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor
    except (ValueError, AttributeError) as _e:
        return None, None


def isTuringOrNewer(major, minor):
    if major is None:
        return False
    if major > 7:
        return True
    if major == 7 and minor >= 5:
        return True
    return False


def detectNVidiaGPU():
    gpus = queryGpus()
    if gpus:
        logging.info(f"NVIDIA GPUs detected: {', '.join(name for name, _ in gpus)}")
        return True
    systemName = platform.system().lower()
    if systemName == "windows" and checkWindowsAdapters():
        logging.info("NVIDIA GPU detected via WMI")
        return True
    if systemName == "linux" and checkLinuxPci():
        logging.info("NVIDIA GPU detected via PCI scan")
        return True
    logging.info("No NVIDIA GPU detected")
    return False


def detectGPUArchitecture():
    """
    Returns:
        tuple: (isModernGPU: bool, gpuName: str, computeCapability: str)
               isModernGPU is True for Turing (compute 7.5) and newer architectures
    """
    gpus = queryGpus()
    if gpus:
        gpuName, computeCap = gpus[0]
        major, minor = parseComputeCapability(computeCap)
        isModern = isTuringOrNewer(major, minor)
        if not isModern:
            logging.warning(
                f"GPU {gpuName} has compute capability {computeCap} (Pascal or older). DirectML backend recommended."
            )
        else:
            logging.info(
                f"GPU {gpuName} has compute capability {computeCap} - modern CUDA support available"
            )
        return isModern, gpuName, computeCap

    # Neither NVML nor nvidia-smi could report compute_cap (driver absent or too
    # old). If an NVIDIA GPU is still detected by other means, assume a modern
    # (Turing+) arch rather than silently picking
    # the lite/DirectML dependency profile -- the removed `nvidia-smi -L`
    # heuristic likewise defaulted unrecognized NVIDIA GPUs to modern.
    if detectNVidiaGPU():
        logging.warning(
            "NVIDIA GPU detected but compute capability is unavailable; "
            "assuming a modern (Turing+) architecture."
        )
        return True, "unknown", "unknown"

    return False, "unknown", "unknown"
