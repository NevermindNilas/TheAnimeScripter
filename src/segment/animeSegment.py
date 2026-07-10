import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint, logWarning
from src.infra.progressBarLogic import ProgressBarLogic
from src.infra.providerCheck import warnIfProviderMissing
from src.io.ffmpegSettings import BuildBuffer, WriteBuffer
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()


def _readBatch(readBuffer, batchSize: int) -> list:
    """Pull up to batchSize frames off the decode buffer.

    A list shorter than batchSize means the decoder's single None sentinel has
    just been consumed, so the caller MUST stop; read() blocks forever once the
    sentinel is gone.
    """
    frames = []
    for _ in range(batchSize):
        frame = readBuffer.read()
        if frame is None:
            break
        frames.append(frame)
    return frames


class AnimeSegment:  # A bit ambiguous because of .train import AnimeSegmentation but it's fine
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        segment_batch: int = 1,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.segmentBatch = max(1, int(segment_batch))

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encode_method,
                custom_encoder=self.custom_encoder,
                grayscale=False,
                width=self.width,
                height=self.height,
                fps=self.fps,
                transparent=True,
                benchmark=self.benchmark,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):
        if ADOBE:
            progressState.update({"status": "Loading background removal model..."})

        filename = modelsMap("segment")
        modelPath = resolveWeightPath("segment", filename)

        from .train import AnimeSegmentation

        self.model = AnimeSegmentation.try_load(
            "isnet_is", modelPath, checker.device, img_size=1024
        )
        self.model.eval()
        # The GT encoder is a training-only distillation branch (6.88M params,
        # 26 MiB fp32). try_load builds it so the checkpoint's gt_encoder.* keys
        # load, but inference never calls it; drop it before moving to the GPU.
        self.model.gt_encoder = None
        self.model.to(checker.device)
        self.stream = torch.cuda.Stream()

        # The model always runs in a fixed 1024-box, so both the shape and the
        # batch are static and the whole forward can be captured once.
        s = 1024
        h, w = self.height, self.width
        self.boxHeight, self.boxWidth = (
            (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        )
        ph, pw = s - self.boxHeight, s - self.boxWidth
        self.padTop, self.padLeft = ph // 2, pw // 2
        self.pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)

        self.dummyInput = torch.zeros(
            (self.segmentBatch, 3, s, s),
            device=checker.device,
            dtype=torch.float32,
        )

        self.cudaGraph = None
        try:
            with torch.cuda.stream(self.stream):
                # 3 warmup iters before capture: cudnn.benchmark is False (no algo
                # autotune), so 3 is enough to JIT the cudnn kernels and allocate
                # their workspace before capture (PyTorch-documented minimum).
                with torch.inference_mode():
                    for _ in range(3):
                        self.model(self.dummyInput)
                        self.stream.synchronize()
            self.cudaGraph = torch.cuda.CUDAGraph()
            self.initTorchCudaGraph()
        except Exception as e:
            logging.warning(f"CUDA graph capture failed, falling back to eager: {e}")
            self.cudaGraph = None

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def getMask(self, frames: list) -> torch.Tensor:
        """frames: list of decoded [1, 3, H, W] tensors. Every frame is resized to
        the same 1024-box, and ISNet is fully convolutional, so the batch axis is
        frame-independent. Returns [B, 4, H, W] (RGB + alpha).

        The batch is concatenated *inside* self.stream. Building it on the default
        stream instead would let the forward read it before the copy landed."""
        with torch.cuda.stream(self.stream):
            input_img = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
            input_img = input_img.to(checker.device).float()
            h0, w0 = input_img.shape[2], input_img.shape[3]
            img_input = F.pad(
                F.interpolate(
                    input_img,
                    size=(self.boxHeight, self.boxWidth),
                    mode="bilinear",
                    align_corners=False,
                ),
                self.pad,
            )
            if self.cudaGraph is not None and img_input.shape == self.dummyInput.shape:
                self.dummyInput.copy_(img_input, non_blocking=True)
                self.cudaGraph.replay()
                pred = self.dummyOutput
            else:
                pred = self.model(img_input)
            # slices a view of the graph's static output; the interpolate below
            # materializes a fresh tensor, so nothing aliasing it reaches the
            # write queue and the next replay cannot overwrite a queued frame
            pred = pred[
                :,
                :,
                self.padTop : self.padTop + self.boxHeight,
                self.padLeft : self.padLeft + self.boxWidth,
            ]
            pred = F.interpolate(
                pred, size=(h0, w0), mode="bilinear", align_corners=False
            )
            pred = torch.cat((input_img, pred), dim=1)
        self.stream.synchronize()
        return pred

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            masks = self.getMask(frames)
            for i in range(masks.shape[0]):
                self.writeBuffer.write(masks[i : i + 1])

        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            while True:
                frames = _readBatch(self.readBuffer, self.segmentBatch)
                if frames:
                    self.processBatch(frames)
                    frameCount += len(frames)
                    bar(len(frames))
                if len(frames) < self.segmentBatch:
                    break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class AnimeSegmentTensorRT:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        segment_batch: int = 1,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.segmentBatch = max(1, int(segment_batch))

        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )
        from src.utils.tensorrt_import import trt

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encode_method,
                custom_encoder=self.custom_encoder,
                grayscale=False,
                width=self.width,
                height=self.height,
                fps=self.fps,
                transparent=True,
                benchmark=self.benchmark,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": "Loading TensorRT background removal model..."}
            )

        filename = modelsMap("segment-tensorrt")
        folderName = "segment-onnx"
        self.modelPath = resolveWeightPath(
            folderName, filename, downloadModel="segment-tensorrt"
        )

        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        B = self.segmentBatch
        paddedHeight = self.height + self.padHeight
        paddedWidth = self.width + self.padWidth

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=False,  # Setting this to false cuz fp16 results are really bad compared to fp32
            optInputShape=[B, 3, paddedHeight, paddedWidth],
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
                fp16=False,  # Setting this to false cuz fp16 results are really bad compared to fp32
                inputsMin=[1, 3, paddedHeight, paddedWidth],
                inputsOpt=[B, 3, paddedHeight, paddedWidth],
                inputsMax=[B, 3, paddedHeight, paddedWidth],
                inputName=["input"],
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (B, 3, paddedHeight, paddedWidth),
            device=checker.device,
            dtype=torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (B, 1, paddedHeight, paddedWidth),
            device=checker.device,
            dtype=torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frames):
        """frames: list of segmentBatch decoded tensors [1, 3, H, W]. Builds the
        [B, 3, paddedH, paddedW] engine input and returns the padded batch."""
        with torch.cuda.stream(self.normStream):
            batch = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
            batch = F.pad(
                batch.float(),
                (0, self.padWidth, 0, self.padHeight),
            )
            self.dummyInput.copy_(batch, non_blocking=True)
            self.normStream.synchronize()
            return batch

    @torch.inference_mode()
    def outputNorm(self, batch, i):
        with torch.cuda.stream(self.outputStream):
            frameWithMask = torch.cat(
                (batch[i : i + 1], self.dummyOutput[i : i + 1]), dim=1
            )
            frameWithMask = frameWithMask[
                :,
                :,
                : frameWithMask.shape[2] - self.padHeight,
                : frameWithMask.shape[3] - self.padWidth,
            ]
            self.outputStream.synchronize()
            return frameWithMask

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            real = len(frames)
            # the engine input shape is fixed at segmentBatch, so pad a short
            # final batch with the last frame and drop the padded outputs
            if real < self.segmentBatch:
                frames = frames + [frames[-1]] * (self.segmentBatch - real)
            batch = self.normFrame(frames)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
            for i in range(real):
                self.writeBuffer.write(self.outputNorm(batch, i))
        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            while True:
                frames = _readBatch(self.readBuffer, self.segmentBatch)
                if frames:
                    self.processBatch(frames)
                    frameCount += len(frames)
                    bar(len(frames))
                if len(frames) < self.segmentBatch:
                    break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class AnimeSegmentDirectML:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.ort = ort

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                grayscale=False,
                transparent=True,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.process)
                executor.submit(self.writeBuffer)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": "Loading DirectML background removal model..."}
            )

        self.filename = modelsMap("segment-directml")
        folderName = "segment-onnx"
        modelPath = resolveWeightPath(
            folderName, self.filename, downloadModel="segment-directml"
        )

        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        providers = self.ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            provider = "DmlExecutionProvider"
        else:
            logWarning(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            provider = "CPUExecutionProvider"

        self.model = self.ort.InferenceSession(modelPath, providers=[provider])
        warnIfProviderMissing(self.model, provider, "DirectML segment")
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)
        self.numpyDType = np.float32
        self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 1, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = False
        self.modelPath = modelPath

    def _fallbackToCpu(self):
        """Reinitialize model with CPU provider after DirectML failure."""
        logAndPrint(
            "DirectML encountered an error, falling back to CPU. Performance will be slower.",
            "yellow",
        )

        self.model = self.ort.InferenceSession(
            self.modelPath, providers=["CPUExecutionProvider"]
        )

        self.IoBinding = self.model.io_binding()
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = True

    def processFrame(self, frame: torch.tensor) -> torch.tensor:
        try:
            frame = frame.to(self.device).float()
            frame = F.pad(frame, (0, self.padWidth, 0, self.padHeight))
            self.dummyInput.copy_(frame)

            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            frameWithMask = torch.cat((frame, self.dummyOutput), dim=1)
            frameWithMask = frameWithMask[
                :,
                :,
                : frameWithMask.shape[2] - self.padHeight,
                : frameWithMask.shape[3] - self.padWidth,
            ]
            self.writeBuffer.write(frameWithMask)

        except UnicodeDecodeError as e:
            if not self.usingCpuFallback:
                logging.warning(f"DirectML UnicodeDecodeError: {e}")
                self._fallbackToCpu()
                self.processFrame(frame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )

        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            while (frame := self.readBuffer.read()) is not None:
                self.processFrame(frame)
                frameCount += 1
                bar(1)

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class AnimeSegmentOpenVino:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        logAndPrint(
            "OpenVINO backend is an experimental feature, please report any issues you encounter.",
            "yellow",
        )

        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.ort = ort

        try:
            import openvino  # noqa: F401
        except ImportError:
            logging.error(
                "OpenVINO is not installed. Please install it to use this backend."
            )
            raise

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                grayscale=False,
                transparent=True,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.process)
                executor.submit(self.writeBuffer)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": "Loading OpenVINO background removal model..."}
            )

        method = "segment-directml"
        self.filename = modelsMap(method)
        folderName = "segment-onnx"
        modelPath = resolveWeightPath(folderName, self.filename, downloadModel=method)

        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        providers = self.ort.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {providers}")

        if "OpenVINOExecutionProvider" in providers:
            logging.info("OpenVINO provider available. Defaulting to OpenVINO")
            provider = "OpenVINOExecutionProvider"
        else:
            logWarning(
                "OpenVINO provider not available, falling back to CPU, expect significantly worse performance"
            )
            provider = "CPUExecutionProvider"

        self.model = self.ort.InferenceSession(modelPath, providers=[provider])
        warnIfProviderMissing(self.model, provider, "OpenVINO segment")
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)
        self.numpyDType = np.float32
        self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 1, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = False
        self.modelPath = modelPath

    def _fallbackToCpu(self):
        """Reinitialize model with CPU provider after OpenVINO failure."""
        logAndPrint(
            "OpenVINO encountered an error, falling back to CPU. Performance will be slower.",
            "yellow",
        )

        self.model = self.ort.InferenceSession(
            self.modelPath, providers=["CPUExecutionProvider"]
        )

        self.IoBinding = self.model.io_binding()
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = True

    def processFrame(self, frame: torch.tensor) -> torch.tensor:
        try:
            frame = frame.to(self.device).float()
            frame = F.pad(frame, (0, self.padWidth, 0, self.padHeight))
            self.dummyInput.copy_(frame)

            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            frameWithMask = torch.cat((frame, self.dummyOutput), dim=1)
            frameWithMask = frameWithMask[
                :,
                :,
                : frameWithMask.shape[2] - self.padHeight,
                : frameWithMask.shape[3] - self.padWidth,
            ]
            self.writeBuffer.write(frameWithMask)

        except UnicodeDecodeError as e:
            if not self.usingCpuFallback:
                logging.warning(f"OpenVINO UnicodeDecodeError: {e}")
                self._fallbackToCpu()
                self.processFrame(frame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )

        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
