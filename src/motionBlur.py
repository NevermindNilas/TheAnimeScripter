import logging
import math
from time import time
from concurrent.futures import ThreadPoolExecutor

import torch

from src.constants import ADOBE
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.progressBarLogic import ProgressBarLogic

if ADOBE:
    from src.utils.aeComms import progressState


def generateWeights(numFrames, scheme="equal"):
    if numFrames <= 0:
        return []

    if scheme == "equal":
        weights = [1.0] * numFrames
    elif scheme == "gaussian_sym":
        radius = numFrames // 2
        sigma = max(numFrames / 6.0, 0.5)
        weights = [
            math.exp(-((i - radius) ** 2) / (2 * sigma * sigma))
            for i in range(numFrames)
        ]
    elif scheme == "pyramid":
        half = numFrames // 2
        weights = [
            float(i + 1) if i <= half else float(numFrames - i)
            for i in range(numFrames)
        ]
    elif scheme == "ascending":
        weights = [float(i + 1) for i in range(numFrames)]
    elif scheme == "descending":
        weights = [float(numFrames - i) for i in range(numFrames)]
    else:
        weights = [1.0] * numFrames

    total = sum(weights)
    return [w / total for w in weights]


class FrameCollector:
    __slots__ = ("frames",)

    def __init__(self):
        self.frames = []

    def put(self, frame):
        self.frames.append(frame)

    def clear(self):
        self.frames.clear()


class FrameBlender:
    def __init__(self, numFrames, scheme, dtype, device):
        self.numFrames = numFrames
        self.scheme = scheme
        self.isEqual = scheme == "equal"
        if not self.isEqual:
            weights = generateWeights(numFrames, scheme)
            self.weightTensor = (
                torch.tensor(weights, dtype=dtype, device=device)
                .view(numFrames, 1, 1, 1)
            )

    @torch.inference_mode()
    def __call__(self, frames):
        stacked = torch.cat(frames, dim=0)
        if self.isEqual:
            return stacked.mean(dim=0, keepdim=True).clamp_(0.0, 1.0)
        return (stacked * self.weightTensor).sum(dim=0, keepdim=True).clamp_(0.0, 1.0)


class MotionBlurPipeline:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half=True,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth="8bit",
        interpolate_method="rife4.25",
        interpolate_factor=4,
        moblur_strength="equal",
        ensemble=False,
        dynamic_scale=False,
        static_step=False,
        compile_mode="default",
        decode_method="cpu",
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.interpolateMethod = interpolate_method
        self.interpolateFactor = interpolate_factor
        self.moblurStrength = moblur_strength
        self.ensemble = ensemble
        self.dynamicScale = dynamic_scale
        self.staticStep = static_step
        self.compileMode = compile_mode
        self.decodeMethod = decode_method
        self.useCuda = torch.cuda.is_available()

        self.interpolateProcess = None

        logging.info(
            f"Motion blur: factor={self.interpolateFactor}, "
            f"strength={self.moblurStrength}, method={self.interpolateMethod}"
        )

        try:
            self._initInterpolationModel()
            self._run()
        except Exception as e:
            logging.exception(f"Motion blur pipeline failed: {e}")

    def _initInterpolationModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Initializing interpolation model: {self.interpolateMethod}..."}
            )

        match self.interpolateMethod:
            case (
                "rife"
                | "rife4.6"
                | "rife4.15-lite"
                | "rife4.16-lite"
                | "rife4.17"
                | "rife4.18"
                | "rife4.20"
                | "rife4.21"
                | "rife4.22"
                | "rife4.22-lite"
                | "rife4.25"
                | "rife4.25-lite"
                | "rife_elexor"
                | "rife4.25-heavy"
            ):
                from src.unifiedInterpolate import RifeCuda

                self.interpolateProcess = RifeCuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    self.ensemble,
                    self.interpolateFactor,
                    self.dynamicScale,
                    self.staticStep,
                    compileMode=self.compileMode,
                )

            case (
                "rife-ncnn"
                | "rife4.6-ncnn"
                | "rife4.15-lite-ncnn"
                | "rife4.16-lite-ncnn"
                | "rife4.17-ncnn"
                | "rife4.18-ncnn"
                | "rife4.20-ncnn"
                | "rife4.21-ncnn"
                | "rife4.22-ncnn"
                | "rife4.22-lite-ncnn"
            ):
                from src.unifiedInterpolate import RifeNCNN

                self.interpolateProcess = RifeNCNN(
                    self.interpolateMethod,
                    self.ensemble,
                    self.width,
                    self.height,
                    self.half,
                    self.interpolateFactor,
                )

            case (
                "rife-tensorrt"
                | "rife4.6-tensorrt"
                | "rife4.15-tensorrt"
                | "rife4.15-lite-tensorrt"
                | "rife4.17-tensorrt"
                | "rife4.18-tensorrt"
                | "rife4.20-tensorrt"
                | "rife4.21-tensorrt"
                | "rife4.22-tensorrt"
                | "rife4.22-lite-tensorrt"
                | "rife4.25-tensorrt"
                | "rife4.25-lite-tensorrt"
                | "rife_elexor-tensorrt"
                | "rife4.25-heavy-tensorrt"
            ):
                from src.unifiedInterpolate import RifeTensorRT

                self.interpolateProcess = RifeTensorRT(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "gmfss":
                from src.gmfss.gmfss import GMFSS

                self.interpolateProcess = GMFSS(
                    int(self.interpolateFactor),
                    self.half,
                    self.width,
                    self.height,
                    self.ensemble,
                    compileMode=self.compileMode,
                )

            case (
                "rife4.6-directml"
                | "rife4.6-openvino"
                | "rife4.15-directml"
                | "rife4.17-directml"
                | "rife4.18-directml"
                | "rife4.20-directml"
                | "rife4.21-directml"
                | "rife4.22-directml"
                | "rife4.22-lite-directml"
                | "rife4.25-directml"
                | "rife4.25-lite-directml"
                | "rife4.25-heavy-directml"
                | "rife4.15-openvino"
                | "rife4.17-openvino"
                | "rife4.18-openvino"
                | "rife4.20-openvino"
                | "rife4.21-openvino"
                | "rife4.22-openvino"
                | "rife4.22-lite-openvino"
                | "rife4.25-openvino"
                | "rife4.25-lite-openvino"
                | "rife4.25-heavy-openvino"
            ):
                from src.unifiedInterpolate import RifeDirectML

                self.interpolateProcess = RifeDirectML(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

    def _run(self):
        if ADOBE:
            progressState.update({"status": "Processing motion blur..."})

        startTime = time()

        self.readBuffer = BuildBuffer(
            videoInput=self.input,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            half=self.half,
            resize=False,
            width=self.width,
            height=self.height,
            bitDepth=self.bitDepth,
            decode_method=self.decodeMethod,
        )

        self.writeBuffer = WriteBuffer(
            self.input,
            self.output,
            self.encode_method,
            self.custom_encoder,
            self.width,
            self.height,
            self.fps,
            sharpen=False,
            sharpen_sens=0.0,
            grayscale=False,
            benchmark=self.benchmark,
            bitDepth=self.bitDepth,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.readBuffer)
            executor.submit(self.writeBuffer)
            executor.submit(self._processFrames)

        elapsed = time() - startTime
        fps = self.totalFrames / elapsed if elapsed > 0 else 0
        logging.info(f"Motion blur done: {elapsed:.2f}s, {fps:.2f} fps")

        if ADOBE:
            progressState.setCompleted(outputPath=self.output)

    def _processFrames(self):
        prevFrame = None
        framesToInsert = self.interpolateFactor - 1
        collector = FrameCollector()
        dtype = torch.float16 if self.half else torch.float32
        device = torch.device("cuda" if self.useCuda else "cpu")
        blender = FrameBlender(
            self.interpolateFactor, self.moblurStrength, dtype, device
        )

        with ProgressBarLogic(self.totalFrames, title=self.input) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break

                collector.clear()
                self.interpolateProcess(frame, collector, framesToInsert, None)

                if prevFrame is None:
                    prevFrame = frame
                    self.writeBuffer.write(frame)
                    bar(1)
                    continue

                collector.frames.insert(0, prevFrame)
                blended = blender(collector.frames)

                if self.useCuda:
                    torch.cuda.current_stream().synchronize()

                self.writeBuffer.write(blended)

                prevFrame = frame
                bar(1)

                if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                    break

        self.writeBuffer.close()
