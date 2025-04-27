class CudaChecker:
    def __init__(self):
        """
        A dumb class to check if CUDA is available and to get the device name.
        Just to avoid writing the same code over and over again.
        """
        # Lazify the import
        import torch

        self.torch = torch

        try:
            self._cuda_available = self.torch.cuda.is_available()
        except Exception as e:
            self._cuda_available = False
            print(f"CUDA is not available: {e}")

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
        return self.torch.cuda.get_device_name(0) if self.cudaAvailable else "cpu"


def detectNVidiaGPU():
    import subprocess
    import platform
    import logging

    """
    Detects if an NVIDIA GPU is present on the system.
    
    Returns:
        Tuple[bool, Optional[str]]: (GPU detected, GPU name if detected)
    """
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.run(
                ["where", "nvidia-smi"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
            )
        else:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
            )

        if result.returncode == 0 and result.stdout:
            gpuInfo = result.stdout.strip().split("\n")[0]
            gpuName = gpuInfo.split(":")[1].strip() if ":" in gpuInfo else "NVIDIA GPU"
            logging.info(f"NVIDIA GPU detected: {gpuName}")
            return True
        else:
            logging.info("No NVIDIA GPU detected")
            return False

    except (subprocess.SubprocessError, FileNotFoundError):
        logging.info("nvidia-smi not found or failed to run")
        return False
