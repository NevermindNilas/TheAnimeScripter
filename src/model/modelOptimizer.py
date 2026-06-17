import logging

import torch


class ModelOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype = torch.float32,
        memoryFormat: torch.memory_format = torch.contiguous_format,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.memoryFormat = memoryFormat

    def optimizeModel(self) -> torch.nn.Module:
        self.model.eval()

        if self.dtype == torch.float16:
            try:
                self.model = self.model.half()
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.dtype = torch.float32
        else:
            self.model = self.model.to(self.dtype)

        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("Model must be an instance of torch.nn.Module")

        return self.model.to(memory_format=self.memoryFormat)
