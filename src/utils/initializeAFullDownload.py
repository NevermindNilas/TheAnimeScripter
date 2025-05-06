from src.utils.isCudaInit import detectNVidiaGPU
from src.utils.dependencyHandler import installDependencies


def initializeAFullDownload() -> None:
    """
    Initialize a full download by installing dependencies from the specified requirements file.
    """

    isNvidia = detectNVidiaGPU()
    extension = (
        "extra-requirements-windows.txt"
        if isNvidia
        else "extra-requirements-windows-lite.txt"
    )

    success, message = installDependencies(extension)
    if not success:
        print(f"Error: {message}")
        raise RuntimeError(f"Failed to install dependencies: {message}")
