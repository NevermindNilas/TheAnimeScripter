"""MoGe-3 (ViT-L) depth on every backend.

CUDA, ROCm and MPS run the whole model in torch. TensorRT runs the fixed-shape
encoder/decoder as an engine and the voxel refiner in torch on the same GPU.
DirectML and OpenVINO run both halves through ONNX Runtime, with the voxel tables
built on the CPU each step. All of them share the post-process in
``src/depth/moge/model.py``.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.depth.backends.cuda import DepthCuda
from src.depth.backends.directml import DepthDirectMLV2
from src.depth.backends.mps import DepthMPS
from src.depth.backends.tensorrt import DepthTensorRTV2
from src.depth.moge.fast import prepareFastRefiner
from src.depth.moge.model import (
    OUTPUT_STRIDE,
    REFINE_STEPS,
    TOKEN_CHANNELS,
    TOKEN_PIXELS,
    coordsToGray,
    loadMoGe3,
    loadRefiner,
    refineCoords,
    tokenGrid,
    voxelBins,
)
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.infra.providerCheck import warnCpuFallback, warnIfProviderMissing
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()


def _checkpointPath(depthMethod: str, half: bool) -> str:
    # Every backend starts from the one checkpoint; the ONNX ones export from it.
    base = depthMethod.split("-")[0]
    return resolveWeightPath(
        base,
        modelsMap(model=base, modelType="pth", half=half),
        half=half,
        modelType="pth",
    )


def _announceLoad(depthMethod: str) -> None:
    if ADOBE:
        progressState.update({"status": f"Loading depth model: {depthMethod}..."})


def _setSizes(driver) -> None:
    rows, columns = tokenGrid(driver.width, driver.height, driver.depthQuality)
    driver.rows, driver.columns = rows, columns
    driver.newHeight, driver.newWidth = rows * TOKEN_PIXELS, columns * TOKEN_PIXELS
    driver.aspectRatio = driver.width / driver.height
    logging.info(
        f"MoGe-3 input {driver.newWidth}x{driver.newHeight}, "
        f"{rows * columns} tokens, {columns * OUTPUT_STRIDE}x{rows * OUTPUT_STRIDE} voxels"
    )


def _fitFrame(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The decoder scales to the model size; its cv2 fallback may not."""
    frame = frame.float()
    if frame.shape[-2:] != (height, width):
        frame = F.interpolate(
            frame, (height, width), mode="bilinear", align_corners=False
        )
    return frame


class MoGe3Cuda(DepthCuda):
    def handleModels(self):
        _announceLoad(self.depth_method)
        self.model = loadMoGe3(
            _checkpointPath(self.depth_method, self.half),
            checker.device,
            halfEncoder=self.half and torch.version.cuda is not None,
        )
        self.model.refiner = prepareFastRefiner(
            self.model.refiner, checker.device, self.half
        )
        _setSizes(self)
        if self.compileMode != "default":
            logging.info(f"compileMode {self.compileMode} ignored for MoGe-3")
            self.compileMode = "default"
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def processBatch(self, frames):
        for frame in frames:
            try:
                with torch.cuda.stream(self.stream):
                    tokens, coords, mask = self.model.encode(
                        _fitFrame(frame, self.newHeight, self.newWidth),
                        self.aspectRatio,
                        self.half,
                    )
                    refined = refineCoords(coords, tokens, self.model.refiner)
                    gray = coordsToGray(refined, mask, self.height, self.width)
                # The writer copies on its own stream and never waits on ours.
                self.stream.synchronize()
                self.writeBuffer.write(gray)
            except Exception as e:
                self.recordFailure(e)
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )


class MoGe3MPS(DepthMPS):
    def handleModels(self):
        _announceLoad(self.depth_method)
        self.model = loadMoGe3(_checkpointPath(self.depth_method, False), self.device)
        _setSizes(self)
        # fp32 whatever --half says, like the other DINOv2 MPS paths.
        self.dtype = torch.float32
        self.half = False
        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on MPS depth backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

    @torch.inference_mode()
    def processBatch(self, frames):
        for frame in frames:
            try:
                image = _fitFrame(frame, self.newHeight, self.newWidth).to(self.device)
                tokens, coords, mask = self.model.encode(image, self.aspectRatio)
                refined = refineCoords(
                    coords, tokens, self.model.refiner, tableDevice="cpu"
                )
                refined, mask = refined.cpu(), mask.cpu()
                torch.mps.synchronize()
                self.writeBuffer.write(
                    coordsToGray(refined, mask, self.height, self.width)
                )
            except Exception as e:
                self.recordFailure(e)
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )


class MoGe3TensorRT(DepthTensorRTV2):
    def handleModels(self):
        from src.depth.moge.export import exportBase

        _announceLoad(self.depth_method)
        _setSizes(self)
        checkpoint = _checkpointPath(self.depth_method, self.half)
        self.modelPath = exportBase(
            checkpoint,
            Path(checkpoint).parent / "onnx",
            self.rows,
            self.columns,
            self.aspectRatio,
            self.half,
        )
        self._batch = 1
        inputShape = [1, 3, self.newHeight, self.newWidth]
        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath, fp16=self.half, optInputShape=inputShape
        )
        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsOpt=inputShape,
                inputName=["image"],
                forceStatic=True,
            )
        if self.engine is None or self.context is None:
            raise RuntimeError(
                f"Failed to build or load a TensorRT engine for {self.modelPath}"
            )

        # The refiner is dynamic, so it stays in torch. Read only its tensors, so
        # the 1.3 GB that went into the engine never reach the GPU.
        self.refiner = prepareFastRefiner(
            loadRefiner(checkpoint, checker.device), checker.device, self.half
        )

        device = checker.device
        rowsOut = self.rows * OUTPUT_STRIDE
        columnsOut = self.columns * OUTPUT_STRIDE
        self.dummyInput = torch.zeros(inputShape, device=device, dtype=torch.float32)
        self.dummyTokens = torch.zeros(
            (1, TOKEN_CHANNELS, self.rows, self.columns),
            device=device,
            dtype=torch.float32,
        )
        self.dummyCoords = torch.zeros(
            (1, 3, rowsOut, columnsOut), device=device, dtype=torch.float32
        )
        self.dummyMask = torch.zeros(
            (1, 1, rowsOut, columnsOut), device=device, dtype=torch.float32
        )
        addresses = {
            "image": self.dummyInput.data_ptr(),
            "tokens": self.dummyTokens.data_ptr(),
            "coords": self.dummyCoords.data_ptr(),
            "mask": self.dummyMask.data_ptr(),
        }
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in addresses:
                raise RuntimeError(f"Unexpected MoGe-3 engine tensor: {name}")
            self.context.set_tensor_address(name, addresses[name])
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.dummyInput.shape)

        self._decodeWidth = self.newWidth
        self._decodeHeight = self.newHeight
        self._decodeResize = True

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def processBatch(self, frames):
        for frame in frames:
            try:
                with torch.cuda.stream(self.normStream):
                    self.dummyInput.copy_(
                        _fitFrame(frame, self.newHeight, self.newWidth),
                        non_blocking=True,
                    )
                self.normStream.synchronize()
                with torch.cuda.stream(self.stream):
                    self.cudaGraph.replay()
                    refined = refineCoords(
                        self.dummyCoords, self.dummyTokens, self.refiner
                    )
                    gray = coordsToGray(
                        refined, self.dummyMask, self.height, self.width
                    )
                self.stream.synchronize()
                self.writeBuffer.write(gray)
            except Exception as e:
                self.recordFailure(e)
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )


class MoGe3DirectML(DepthDirectMLV2):
    """MoGe-3 through ONNX Runtime; the same class serves ``-openvino``."""

    def handleModels(self):
        from src.depth.moge.export import exportBase, exportRefiner

        _announceLoad(self.depth_method)
        _setSizes(self)
        checkpoint = _checkpointPath(self.depth_method, self.half)
        directory = Path(checkpoint).parent / "onnx"
        self.basePath = exportBase(
            checkpoint,
            directory,
            self.rows,
            self.columns,
            self.aspectRatio,
            self.half,
        )
        self.refinerPath = exportRefiner(checkpoint, directory)

        available = self.ort.get_available_providers()
        if "openvino" in self.depth_method:
            provider, label = "OpenVINOExecutionProvider", "OpenVINO"
        else:
            provider, label = "DmlExecutionProvider", "DirectML"

        if provider in available:
            self.baseSession = self.ort.InferenceSession(
                self.basePath, providers=[provider]
            )
            self.refinerSession = self.ort.InferenceSession(
                self.refinerPath, providers=[provider]
            )
            warnIfProviderMissing(self.baseSession, provider, f"{label} depth")
            warnIfProviderMissing(self.refinerSession, provider, f"{label} depth")
        else:
            warnCpuFallback(
                f"{label} provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self._openCpuSessions()

        self.usingCpuFallback = provider not in available
        self._decodeWidth = self.newWidth
        self._decodeHeight = self.newHeight
        self._decodeResize = True

    def _openCpuSessions(self):
        providers = ["CPUExecutionProvider"]
        self.baseSession = self.ort.InferenceSession(self.basePath, providers=providers)
        self.refinerSession = self.ort.InferenceSession(
            self.refinerPath, providers=providers
        )

    def _fallbackToCpu(self):
        logAndPrint(
            "DirectML/OpenVINO could not run MoGe-3, falling back to CPU. Performance will be much slower.",
            "yellow",
        )
        self._openCpuSessions()
        self.usingCpuFallback = True

    @torch.inference_mode()
    def _infer(self, image: np.ndarray) -> torch.Tensor:
        from src.depth.moge.export import refinerFeed
        from src.depth.moge.sparse import buildLevels

        tokens, coords, mask = self.baseSession.run(None, {"image": image})
        current = torch.from_numpy(coords).permute(0, 2, 3, 1).float()
        height, width = current.shape[1:3]
        for _ in range(REFINE_STEPS):
            levels = buildLevels(voxelBins(current[0, :, :, 2]), 5)
            feed = refinerFeed(current.reshape(-1, 3).numpy(), tokens, levels)
            update = self.refinerSession.run(["update"], feed)[0]
            logDepth = current[..., 2] + torch.from_numpy(update).reshape(
                1, height, width
            )
            current = torch.cat([current[..., :2], logDepth[..., None]], dim=-1)
        return coordsToGray(current, torch.from_numpy(mask), self.height, self.width)

    @torch.inference_mode()
    def processFrame(self, frame):
        image = _fitFrame(frame.cpu(), self.newHeight, self.newWidth).numpy()
        try:
            try:
                gray = self._infer(image)
            except Exception as e:
                if self.usingCpuFallback:
                    raise
                logging.warning(f"MoGe-3 ONNX inference failed: {e}")
                self._fallbackToCpu()
                gray = self._infer(image)
            self.writeBuffer.write(gray)
        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Something went wrong while processing the frame, {e}")
