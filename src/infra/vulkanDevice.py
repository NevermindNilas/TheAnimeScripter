"""The Vulkan device the NCNN backends run on.

Device 0 is whatever Vulkan lists first, which on a hybrid machine can be the
integrated GPU or a software rasteriser while a discrete GPU sits idle.
"""

from functools import lru_cache

from src.infra.logAndPrint import logAndPrint, logWarning

CPU_DEVICE = -1

# ncnn's GpuInfo.type(): 0 discrete, 1 integrated, 2 virtual, 3 CPU.
_SOFTWARE_RASTERISER = 3


@lru_cache(maxsize=1)
def _bestHardwareDevice():
    """(index, name) of the fastest Vulkan GPU, or None if there is none.

    A software rasteriser such as lavapipe does not count: ncnn's own CPU path
    is much faster than running its compute shaders on the CPU.
    """
    import ncnn  # ty: ignore[unresolved-import]

    if ncnn.get_gpu_count() == 0:
        return None
    index = ncnn.get_default_gpu_index()
    info = ncnn.get_gpu_info(index)
    if info.type() == _SOFTWARE_RASTERISER:
        return None
    return index, info.device_name()


def hasVulkanGpu() -> bool:
    """Whether an NCNN backend would reach a real GPU on this machine."""
    try:
        return _bestHardwareDevice() is not None
    except ImportError:
        # An install from before ncnn was a dependency: assume the common case.
        return True


@lru_cache(maxsize=1)
def bestVulkanDevice() -> int:
    """The index of the fastest Vulkan GPU, or ``CPU_DEVICE`` if there is none."""
    try:
        device = _bestHardwareDevice()
    except ImportError:
        return 0

    if device is None:
        logWarning(
            "NCNN found no Vulkan GPU, so this stage is running on the CPU and will "
            "be much slower. Installing your GPU's Vulkan driver enables acceleration."
        )
        return CPU_DEVICE

    index, name = device
    logAndPrint(f"NCNN is running on {name}", "green")
    return index
