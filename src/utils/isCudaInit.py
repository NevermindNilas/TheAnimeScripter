class CudaChecker:
    def __init__(self):
        """
        A dumb class to check if CUDA is available and to get the device name.
        Just to avoid writing the same code over and over again.

        Note: This class checks if CUDA is available in PyTorch, but does not
        validate if the GPU architecture is compatible with modern CUDA kernels.
        Use detectGPUArchitecture() to check for Pascal or older GPUs.
        """
        # Lazify the import
        import torch
        import logging

        global LOGGEDALREADY
        self.torch = torch
        self.logging = logging

        try:
            self._cuda_available = self.torch.cuda.is_available()

            if self._cuda_available:
                try:
                    test = self.torch.zeros(1, device="cuda")
                    _ = test + 1
                except RuntimeError as e:
                    if "no kernel image is available" in str(e).lower():
                        self.logging.warning(
                            f"CUDA is technically available but kernels are incompatible: {e}"
                        )
                        self.logging.warning(
                            "This usually indicates a Pascal (GTX 1000 series) or older GPU. "
                            "Consider using DirectML backend for better compatibility."
                        )
                del test
        except Exception as e:
            self._cuda_available = False
            self.logging.warning(f"CUDA is not available: {e}")

        if self._cuda_available:
            self.enableCudaOptimizations()

    @property
    def cudaAvailable(self):
        return self._cuda_available

    def enableCudaOptimizations(self):
        self.torch.backends.cudnn.benchmark = False
        self.torch.backends.cudnn.enabled = True

    def disableCudaOptimizations(self):
        self.torch.backends.cudnn.benchmark = False
        self.torch.backends.cudnn.enabled = False

    @property
    def device(self):
        return self.torch.device("cuda" if self.cudaAvailable else "cpu")

    @property
    def deviceName(self):
        if not self.cudaAvailable:
            return "cpu"
        try:
            return self.torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError) as e:
            self.logging.warning(f"Could not get CUDA device name: {e}")
            return "cuda_device_unknown"

    @property
    def deviceCount(self):
        """Get the number of available CUDA devices."""
        return self.torch.cuda.device_count() if self.cudaAvailable else 0

    @property
    def allDeviceNames(self):
        """Get the names of all available CUDA devices."""
        if not self.cudaAvailable:
            return ["cpu"]
        try:
            return [self.torch.cuda.get_device_name(i) for i in range(self.deviceCount)]
        except (RuntimeError, AssertionError) as e:
            self.logging.warning(f"Could not get all CUDA device names: {e}")
            return ["cuda_device_unknown"]


def detectNVidiaGPU():
    import subprocess
    import logging

    """
    Detects all NVIDIA GPUs present on the system.
    
    Returns:
        bool: True if NVIDIA GPU detected, False otherwise
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )

        if result.returncode == 0 and result.stdout:
            gpuLines = result.stdout.strip().split("\n")
            gpuNames = []

            for line in gpuLines:
                if ":" in line:
                    # Format is typically: "GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-...)"
                    gpu_name = line.split(":")[1].strip().split("(")[0].strip()
                    gpuNames.append(gpu_name)

            if gpuNames:
                logging.info(f"NVIDIA GPUs detected: {', '.join(gpuNames)}")
                return True
            else:
                logging.info("No NVIDIA GPU detected")
                return False
        else:
            logging.info("No NVIDIA GPU detected")
            return False

    except (subprocess.SubprocessError, FileNotFoundError):
        logging.info("nvidia-smi not found or failed to run")
        return False


def detectGPUArchitecture():
    """
    Returns:
        tuple: (isModernGPU: bool, gpuName: str, computeCapability: str)
               isModernGPU is True for Volta (compute 7.0) and newer architectures
    """
    import subprocess
    import logging

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(",")
                if len(parts) >= 2:
                    gpuName = parts[0].strip()
                    computeCap = parts[1].strip()

                    try:
                        majorVersion = int(float(computeCap))
                        isModern = majorVersion >= 7

                        if not isModern:
                            logging.warning(
                                f"GPU {gpuName} has compute capability {computeCap} (Pascal or older). "
                                f"Modern CUDA kernels may not be compatible. DirectML backend recommended."
                            )
                        else:
                            logging.info(
                                f"GPU {gpuName} has compute capability {computeCap} - modern CUDA support available"
                            )

                        return isModern, gpuName, computeCap
                    except (ValueError, TypeError):
                        logging.warning(
                            f"Could not parse compute capability: {computeCap}"
                        )

        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )

        if result.returncode == 0 and result.stdout:
            gpuName = result.stdout.strip().split("\n")[0]
            if ":" in gpuName:
                gpuName = gpuName.split(":")[1].strip().split("(")[0].strip()

            oldArchitectures = [
                "GTX 9",
                "GTX 10",
                "GT 7",
                "GT 8",
                "GT 9",
                "Quadro K",
                "Quadro M",
                "Quadro P",
                "Tesla K",
                "Tesla M",
                "Tesla P",
            ]

            isOld = any(arch in gpuName for arch in oldArchitectures)

            if isOld:
                logging.warning(
                    f"GPU {gpuName} appears to be Pascal generation or older. "
                    f"DirectML backend recommended for compatibility."
                )
                return False, gpuName, "unknown"

            return True, gpuName, "unknown"

    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logging.warning(f"Could not detect GPU architecture: {e}")

    return True, "unknown", "unknown"
