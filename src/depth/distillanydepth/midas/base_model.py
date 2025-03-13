import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


class BaseModel(torch.nn.Module, PyTorchModelHubMixin):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters, strict=False)
