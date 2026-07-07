"""
Shared 6-channel scene-change scorer.

Scores a ``(prev, curr)`` RGB frame pair with an ONNX classifier that takes the
two frames concatenated along the channel axis (6-channel input, HxW resize) and
returns a scalar cut probability (softmax ``[0][0]``).

This is the pure scoring core lifted out of ``AutoClipMaxxvit`` so that BOTH the
whole-video autoclip prepass and the streaming interpolation-path detector share
one implementation. The prepass feeds already-decoded HWC uint8 frames
(``preprocessHWC``); the streaming detector feeds ``(1, 3, H, W)`` float [0,1]
tensors (``preprocessCHW``). ``score`` is identical for both.
"""

import logging
import os

import numpy as np
import torch

from src.infra.logAndPrint import logAndPrint
from src.model.download import downloadModels
from src.model.registry import modelsMap, weightsDir


class SceneChangeScorer6ch:
    """
    6-channel (prev+curr) ONNX scene-change classifier.

    Args:
        method: e.g. ``"maxxvit-tensorrt"`` or ``"maxxvit-directml"``. The
            ``-tensorrt`` / ``-directml`` suffix selects the backend.
        half: request fp16 (ignored for the TensorRT path — the model's softmax
            saturates in fp16 and TRT 10.x lacks an fp16 depthwise-conv kernel at
            this shape, so TRT is pinned to fp32; mirrors AutoClipMaxxvit).
        size: model spatial input (square). maxxvit == 224.
    """

    def __init__(self, method: str, half: bool, size: int = 224):
        self.method = method
        self.half = half
        self.H = self.W = size
        self.backend = "tensorrt" if method.endswith("-tensorrt") else "directml"
        self._loadModel()

    # ---- model loading (lifted from AutoClipMaxxvit) -----------------------

    def _resolveModelPath(self):
        # TensorRT path is pinned to fp32 weights + fp32 engine because TRT 10.x
        # lacks an fp16 kernel for the depthwise conv at this shape, and the
        # model's softmax is numerically unstable in fp16 (saturates).
        weightHalf = False if self.backend == "tensorrt" else self.half
        filename = modelsMap(self.method, half=weightHalf, modelType="onnx")
        folderName = self.method.replace("-tensorrt", "-onnx").replace(
            "-directml", "-onnx"
        )
        modelPath = os.path.join(weightsDir, folderName, filename)
        if not os.path.exists(modelPath):
            modelPath = downloadModels(
                model=self.method, half=weightHalf, modelType="onnx"
            )
        return modelPath

    def _loadModel(self):
        self.modelPath = self._resolveModelPath()
        if self.backend == "tensorrt":
            self._loadTensorRT()
        else:
            self._loadDirectML()

    def _loadDirectML(self):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            logging.info("Using DirectML for scene-change 6ch inference")
            self.session = ort.InferenceSession(
                self.modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logAndPrint(
                "DirectML provider not available, falling back to CPU. "
                "Performance will be significantly worse.",
                "yellow",
            )
            self.session = ort.InferenceSession(
                self.modelPath, providers=["CPUExecutionProvider"]
            )

    def _loadTensorRT(self):
        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )
        from src.utils.tensorrt_import import trt

        self.trt = trt

        if self.half:
            logging.warning(
                "scene-change 6ch tensorrt ignores --half: model softmax "
                "overflows in fp16. Building/using fp32 engine."
            )
        useFp16 = False

        enginePath = tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=useFp16,
            optInputShape=[0, 6, self.H, self.W],
        )

        engine, context = tensorRTEngineLoader(enginePath)
        if engine is None or context is None or not os.path.exists(enginePath):
            engine, context = tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=useFp16,
                inputsMin=[6, self.H, self.W],
                inputsOpt=[6, self.H, self.W],
                inputsMax=[6, self.H, self.W],
            )

        if engine is None or context is None:
            raise RuntimeError(
                f"Failed to build/load TensorRT engine for {self.modelPath}"
            )

        self.engine = engine
        self.context = context

        self.dType = torch.float32
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream()

        self.dummyInput = torch.zeros(
            (6, self.H, self.W), device=self.device, dtype=self.dType
        )
        self.dummyOutput = torch.zeros((1, 2), device=self.device, dtype=self.dType)

        bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]
        for i in range(self.engine.num_io_tensors):
            tensorName = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensorName, bindings[i])
            if self.engine.get_tensor_mode(tensorName) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensorName, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

        self.cudaGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    # ---- preprocessing -----------------------------------------------------

    def preprocessHWC(self, hwcTensor):
        """Prepass path: nelux HWC uint8 -> CHW [0,1] (numpy for DML, torch GPU
        for TRT)."""
        if self.backend == "tensorrt":
            with torch.cuda.stream(self.stream):
                t = (
                    hwcTensor.to(device=self.device, non_blocking=True)
                    .to(self.dType)
                    .div_(255.0)
                    .permute(2, 0, 1)
                )
            return t
        arr = hwcTensor.numpy()
        arr = arr.astype(np.float16 if self.half else np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    def preprocessCHW(self, frame):
        """Streaming path: a ``(1, 3, H, W)`` (or ``(3, H, W)``) float [0,1]
        tensor -> CHW [0,1] resized to the model input, matching the backend's
        tensor type."""
        import torch.nn.functional as F

        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        resized = F.interpolate(
            frame.float(),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (3, H, W) in [0,1]
        if self.backend == "tensorrt":
            return resized.to(device=self.device, dtype=self.dType)
        return np.ascontiguousarray(
            resized.detach()
            .cpu()
            .numpy()
            .astype(np.float16 if self.half else np.float32)
        )

    # ---- scoring (lifted from AutoClipMaxxvit) -----------------------------

    def scoreDirectML(self, prev, curr):
        inputs = np.concatenate((prev, curr), axis=0)
        result = self.session.run(None, {"input": inputs})[0]
        return float(result[0][0])

    def scoreTensorRT(self, prev, curr):
        with torch.cuda.stream(self.stream):
            self.dummyInput.copy_(torch.cat((prev, curr), dim=0), non_blocking=True)
            self.cudaGraph.replay()
            score = self.dummyOutput[0][0].item()
        self.stream.synchronize()
        return score

    def score(self, prev, curr):
        """Cut probability for a preprocessed pair (softmax ``[0][0]``)."""
        if self.backend == "tensorrt":
            return self.scoreTensorRT(prev, curr)
        return self.scoreDirectML(prev, curr)
