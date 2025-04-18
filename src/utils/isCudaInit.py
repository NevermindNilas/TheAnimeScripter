import torch


class CudaChecker:
    def __init__(self):
        """
        A dumb class to check if CUDA is available and to get the device name.
        Just to avoid writing the same code over and over again.
        """
        try:
            self._cuda_available = torch.cuda.is_available()
        except Exception as e:
            self._cuda_available = False
            print(f"CUDA is not available: {e}")

        if self._cuda_available:
            self.enableCudaOptimizations()

    @property
    def cudaAvailable(self):
        return self._cuda_available

    def enableCudaOptimizations(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

    def disableCudaOptimizations(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    @property
    def device(self):
        return torch.device("cuda" if self.cudaAvailable else "cpu")

    @property
    def deviceName(self):
        return torch.cuda.get_device_name(0) if self.cudaAvailable else "cpu"
