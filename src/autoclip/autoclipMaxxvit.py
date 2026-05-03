import os
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch
import nelux

import src.constants as cs
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.logAndPrint import logAndPrint
from src.utils.progressBarLogic import ProgressBarLogic


_SENTINEL = object()


class AutoClipMaxxvit:
    """
    Scene-change autoclip backed by the MaxxVit ONNX model.

    Decoding runs in a dedicated thread via a ``ThreadPoolExecutor``; nelux
    decodes + downscales to 224x224 (libswscale) and pushes torch tensors
    (HWC uint8) into a bounded queue. The main thread runs preprocess +
    inference. Output index ``[0][0]`` of the model's softmax is the cut
    probability; cut timestamps (seconds) are written to
    ``autoclipresults.txt``.
    """

    H = W = 224
    QUEUE_SIZE = 8

    def __init__(
        self,
        input,
        method,
        threshold,
        inPoint,
        outPoint,
        half,
    ):
        self.input = input
        self.method = method
        self.threshold = threshold
        self.inPoint = inPoint
        self.outPoint = outPoint
        self.half = half

        self.backend = "tensorrt" if method.endswith("-tensorrt") else "directml"
        self._loadModel()
        self._run()

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
            logging.info("Using DirectML for autoclip MaxxVit inference")
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
        import tensorrt as trt
        from src.utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt

        if self.half:
            logging.warning(
                "maxxvit-tensorrt ignores --half: model softmax overflows in fp16. "
                "Building/using fp32 engine."
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

    def _preprocessNumpy(self, hwcTensor):
        """DML: zero-copy torch CPU uint8 -> numpy CHW float [0,1]."""
        arr = hwcTensor.numpy()
        arr = arr.astype(np.float16 if self.half else np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    def _preprocessTorch(self, hwcTensor):
        """TRT: torch CPU uint8 -> CHW fp32 [0,1] on GPU."""
        with torch.cuda.stream(self.stream):
            t = (
                hwcTensor.to(device=self.device, non_blocking=True)
                .to(self.dType)
                .div_(255.0)
                .permute(2, 0, 1)
            )
        return t

    def _scoreDirectML(self, prev, curr):
        inputs = np.concatenate((prev, curr), axis=0)
        result = self.session.run(None, {"input": inputs})[0]
        return float(result[0][0])

    def _scoreTensorRT(self, prev, curr):
        with torch.cuda.stream(self.stream):
            self.dummyInput.copy_(torch.cat((prev, curr), dim=0), non_blocking=True)
            self.cudaGraph.replay()
            score = self.dummyOutput[0][0].item()
        self.stream.synchronize()
        return score

    def _openReader(self):
        # Linear iter is the fast path. Do NOT switch to ``reader.frame_at(i)``
        # in a loop: that path seeks to the nearest keyframe + decodes forward
        # per call, giving O(N^2)-ish behavior. Range slicing via
        # ``reader([startSec, endSec])`` reuses the persistent decoder.
        reader = nelux.VideoReader(
            self.input,
            backend="pytorch",
            decode_accelerator="cpu",
            resize=(self.W, self.H),
            num_threads=8,
        )
        fps = reader.fps
        if not fps or fps <= 0:
            raise RuntimeError(f"Invalid FPS reported for video: {self.input}")

        startFrame = int(round(float(self.inPoint) * fps)) if self.inPoint else 0
        endFrame = (
            int(round(float(self.outPoint) * fps))
            if self.outPoint
            else reader.frame_count
        )
        startSec = startFrame / fps
        totalFrames = max(1, endFrame - startFrame)

        if startFrame > 0 or endFrame < reader.frame_count:
            iterable = reader([startFrame / fps, endFrame / fps])
        else:
            iterable = reader

        try:
            reader.start_prefetch()
        except Exception:
            pass
        return iterable, fps, startSec, totalFrames

    def _decodeWorker(self, iterable, preprocess, queue):
        # nelux 0.9.2+ allocates a fresh tensor per iteration (no buffer
        # reuse), so no clone is required. Running preprocess() here also
        # parallelizes cast/transpose with the main thread's inference.
        try:
            for frame in iterable:
                queue.put(preprocess(frame))
        finally:
            queue.put(_SENTINEL)

    def _run(self):
        iterable, fps, startSec, totalFrames = self._openReader()

        if self.backend == "tensorrt":
            preprocess = self._preprocessTorch
            score = self._scoreTensorRT
        else:
            preprocess = self._preprocessNumpy
            score = self._scoreDirectML

        queue: Queue = Queue(maxsize=self.QUEUE_SIZE)
        cuts = []
        prev = None
        idx = -1

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="autoclip-decode") as pool:
            future = pool.submit(self._decodeWorker, iterable, preprocess, queue)
            try:
                with ProgressBarLogic(totalFrames, title=f"AutoClip ({self.method})") as pbar:
                    while True:
                        curr = queue.get()
                        if curr is _SENTINEL:
                            break
                        idx += 1
                        pbar.advance(1)
                        if prev is None:
                            prev = curr
                            continue
                        if score(prev, curr) > self.threshold:
                            cuts.append(startSec + idx / fps)
                        prev = curr
                future.result()
            except BaseException:
                # Drain queue so the decoder thread can finish promptly.
                while True:
                    item = queue.get()
                    if item is _SENTINEL:
                        break
                raise

        outPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
        with open(outPath, "w") as f:
            for i, t in enumerate(cuts):
                logging.info(f"Scene {i + 1}: cut at {t:.3f}s")
                f.write(f"{t}\n")
        logAndPrint(f"AutoClip wrote {len(cuts)} cuts to {outPath}", "green")
