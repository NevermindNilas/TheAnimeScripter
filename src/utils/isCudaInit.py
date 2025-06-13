class CudaChecker:
    def __init__(self):
        """
        A dumb class to check if CUDA is available and to get the device name.
        Just to avoid writing the same code over and over again.
        """
        # Lazify the import
        import torch
        import logging

        global LOGGEDALREADY
        self.torch = torch
        self.logging = logging

        try:
            self._cuda_available = self.torch.cuda.is_available()
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
        Tuple[bool, list[str]]: (GPUs detected, List of GPU names if detected)
    
    Note: This function specifically detects NVIDIA GPUs using nvidia-smi.
    It will return False for AMD (ROCm) or Intel GPUs even if they support CUDA-like operations.
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
                return True, gpuNames
            else:
                logging.info("No NVIDIA GPU detected")
                return False, []
        else:
            logging.info("No NVIDIA GPU detected")
            return False, []

    except (subprocess.SubprocessError, FileNotFoundError):
        logging.info("nvidia-smi not found or failed to run")
        return False, []
