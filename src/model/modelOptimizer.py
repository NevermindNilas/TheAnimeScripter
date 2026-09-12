from __future__ import annotations

import logging

import torch


class ModelOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        # None means "torch default": the torch.* values below are resolved
        # inside optimizeModel so that importing this module never evaluates
        # torch attributes at def time (torch-less bare-venv loading, where
        # only a minimal torch stub exists).
        dtype: torch.dtype | None = None,
        memoryFormat: torch.memory_format | None = None,
    ) -> None:
        self.model = model
        self.dtype = dtype if dtype is not None else torch.float32
        self.memoryFormat = (
            memoryFormat if memoryFormat is not None else torch.contiguous_format
        )

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
