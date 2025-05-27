from torch.fx import GraphModule
import torch
import logging


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

        try:
            symbolicTraced: GraphModule = torch.fx.symbolic_trace(self.model)
        except Exception as e:
            logging.error(f"Error tracing model: {e}")
            return self.toMemoryFormat(self.model)

        # I only tested this so far, I will need to test other optimizations
        class ReplaceReLU(torch.fx.Transformer):
            def call_module(self, target, args, kwargs):
                if target == "relu":
                    return torch.nn.functional.leaky_relu(*args, **kwargs)
                return super().call_module(target, args, kwargs)

        try:
            transformer = ReplaceReLU(symbolicTraced)
            optimizedModel = transformer.transform()
        except Exception:
            optimizedModel = symbolicTraced

        optimizedModel = self.toMemoryFormat(optimizedModel)
        return optimizedModel

    def toMemoryFormat(self, model) -> torch.nn.Module:
        return model.to(memory_format=self.memoryFormat)
