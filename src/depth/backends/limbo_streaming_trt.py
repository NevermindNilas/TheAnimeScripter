"""TensorRT Limbo streaming with an actual dynamic view axis."""

import logging
from pathlib import Path

import numpy as np
import torch

from src.depth.backends._shared import limboResolution
from src.depth.backends.da3_streaming import DA3StreamingCuda
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap


class LimboStreamingTensorRT(DA3StreamingCuda):
    def handleModels(self):
        if self.depth_method not in ("video_limbo-tensorrt", "video_limbo_v2-tensorrt"):
            raise ValueError(f"Unsupported Limbo streaming model: {self.depth_method}")
        from src.depth.streaming_export import exportStreamingDepth

        base = self.depth_method.removeprefix("video_").removesuffix("-tensorrt")
        checkpoint = resolveWeightPath(
            base, modelsMap(base, modelType="pth"), modelType="pth", half=self.half
        )
        self.newHeight, self.newWidth = limboResolution(self.width, self.height)
        modelPath = exportStreamingDepth(
            checkpoint,
            Path(checkpoint).parent / "streaming",
            self.newHeight,
            self.newWidth,
            self.half,
        )
        self._loadEngine(modelPath)
        self._decodeWidth, self._decodeHeight = self.newWidth, self.newHeight
        self._decodeResize = True
        logging.info(
            "Limbo TensorRT streaming: %s, window=%d, input=%dx%d",
            self.depth_method,
            self.depthWindow,
            self.newWidth,
            self.newHeight,
        )

    def _loadEngine(self, modelPath):
        import tensorrt as trt

        from src.model.trtHandler import tensorRTEngineCreator, tensorRTEngineLoader

        enginePath = str(Path(modelPath).with_suffix(f".w{self.depthWindow}.engine"))
        loaded = tensorRTEngineLoader(enginePath)
        if loaded is None or loaded[0] is None:
            shape = [self.depthWindow, 3, self.newHeight, self.newWidth]
            loaded = tensorRTEngineCreator(
                modelPath=modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, self.newHeight, self.newWidth],
                inputsOpt=shape,
                inputsMax=shape,
                inputName=["image"],
            )
        if loaded is None or loaded[0] is None:
            raise RuntimeError("Failed to build Limbo streaming TensorRT engine")
        self.engine, self.context = loaded
        assert self.engine is not None and self.context is not None
        self.stream = torch.cuda.Stream()
        self.context.set_input_shape(
            "image", (self.depthWindow, 3, self.newHeight, self.newWidth)
        )
        self.bindings = {}
        for name in self.engine:
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in (trt.float16, trt.float32):
                raise RuntimeError(
                    f"Unsupported streaming tensor dtype: {name}: {dtype}"
                )
            shape = tuple(self.context.get_tensor_shape(name))
            self.bindings[name] = torch.empty(
                shape,
                dtype=torch.float16 if dtype == trt.float16 else torch.float32,
                device="cuda",
            )
            self.context.set_tensor_address(name, self.bindings[name].data_ptr())
        if set(self.bindings) != {"image", "depth"}:
            raise RuntimeError("Invalid Limbo streaming engine bindings")
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.stream.wait_stream(torch.cuda.current_stream())

    @torch.inference_mode()
    def _inferChunk(self, frames):
        count = len(frames)
        if not 1 <= count <= self.depthWindow:
            raise ValueError("Frame count is outside the streaming engine profile")
        if not self.context.set_input_shape(
            "image", (count, 3, self.newHeight, self.newWidth)
        ):
            raise RuntimeError("TensorRT rejected the streaming view count")
        # Every tail uses its real length. Duplicate padding changes attention.
        with torch.cuda.stream(self.stream):
            images = (
                torch.from_numpy(np.stack(frames))
                .to("cuda")
                .permute(0, 3, 1, 2)
                .float()
                / 255
            )
            self.bindings["image"][:count].copy_((images - self.mean) / self.std)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("Limbo streaming TensorRT inference failed")
            depth = self.bindings["depth"][:count].float().cpu()
        self.stream.synchronize()
        return depth.numpy()
