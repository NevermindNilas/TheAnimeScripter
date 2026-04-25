import logging
import math
import os
from time import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch

from src.constants import ADOBE
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.progressBarLogic import ProgressBarLogic

if ADOBE:
    from src.utils.aeComms import progressState


def generateWeights(numSamples, scheme="gaussian_sym"):
    if numSamples <= 0:
        return []
    if numSamples == 1:
        return [1.0]

    if scheme == "equal":
        weights = [1.0] * numSamples
    elif scheme == "gaussian_sym":
        center = (numSamples - 1) / 2.0
        sigma = max(numSamples / 4.0, 0.5)
        weights = [
            math.exp(-((i - center) ** 2) / (2.0 * sigma * sigma))
            for i in range(numSamples)
        ]
    elif scheme == "pyramid":
        center = (numSamples - 1) / 2.0
        weights = [
            max(0.0, 1.0 - abs(i - center) / (center + 1.0))
            for i in range(numSamples)
        ]
    elif scheme == "ascending":
        weights = [float(i + 1) for i in range(numSamples)]
    elif scheme == "descending":
        weights = [float(numSamples - i) for i in range(numSamples)]
    else:
        weights = [1.0] * numSamples

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


class GammaCorrectBlender:
    """Weighted frame blend in linear-light space.

    Converts sRGB-encoded [0,1] input to linear via gamma 2.2, sums, then
    re-encodes. Gamma 2.2 is a close approximation to the sRGB piecewise EOTF
    and runs ~2x faster than the exact form. Blending in the encoded space
    (what the old pipeline did) crushes highlights and lifts shadows during
    averaging, producing the muddy, plastic look.
    """

    GAMMA = 2.2
    INV_GAMMA = 1.0 / 2.2

    def __init__(self, weights, dtype, device, gamma=True):
        self.weightTensor = (
            torch.tensor(weights, dtype=dtype, device=device)
            .view(-1, 1, 1, 1)
        )
        self.gamma = gamma
        self.isEqual = len(weights) > 0 and all(
            abs(w - weights[0]) < 1e-9 for w in weights
        )

    @torch.inference_mode()
    def __call__(self, frames):
        stacked = torch.cat(frames, dim=0)
        if self.gamma:
            stacked = stacked.clamp_(0.0, 1.0).pow_(self.GAMMA)

        if self.isEqual:
            blended = stacked.mean(dim=0, keepdim=True)
        else:
            blended = (stacked * self.weightTensor).sum(dim=0, keepdim=True)

        if self.gamma:
            blended = blended.clamp_(0.0, 1.0).pow_(self.INV_GAMMA)
        return blended.clamp_(0.0, 1.0)


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
        interpolate_factor=8,
        moblur_strength="gaussian_sym",
        moblur_shutter_angle=180.0,
        moblur_gamma=True,
        moblur_mask="",
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
        self.interpolateFactor = max(2, int(interpolate_factor))
        self.moblurStrength = moblur_strength
        self.shutterAngle = max(0.0, min(360.0, float(moblur_shutter_angle)))
        self.moblurGamma = bool(moblur_gamma)
        self.moblurMaskPath = moblur_mask or ""
        self.mask = None
        self.ensemble = ensemble
        self.dynamicScale = dynamic_scale
        self.staticStep = static_step
        self.compileMode = compile_mode
        self.decodeMethod = decode_method
        self.useCuda = torch.cuda.is_available()

        self.interpolateProcess = None

        logging.info(
            f"Motion blur: factor={self.interpolateFactor}, "
            f"shutter={self.shutterAngle}°, weights={self.moblurStrength}, "
            f"linear_blend={self.moblurGamma}, method={self.interpolateMethod}"
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

    def _loadMask(self, dtype, device):
        """Load protection mask as blur-weight tensor [1, 1, H, W] in [0, 1].

        Intended input: transparent PNG where the user paints protected regions
        with opaque dark pixels on a transparent background. Everything else
        (transparent or bright) blurs normally.

        Derivation of per-pixel protection:
            RGBA: protection = alpha * (1 - luma)
            RGB : protection = 1 - luma          (assume fully opaque)
            Gray: protection = 1 - value         (dark = protected)

        Returned tensor is blur-weight = 1 - protection, so downstream math is:
            out = blended * weight + currFrame * (1 - weight)
        """
        if not self.moblurMaskPath:
            return None

        if not os.path.isfile(self.moblurMaskPath):
            logging.warning(
                f"Motion blur mask not found: {self.moblurMaskPath}. Ignoring."
            )
            return None

        maskImg = cv2.imread(self.moblurMaskPath, cv2.IMREAD_UNCHANGED)
        if maskImg is None:
            logging.warning(
                f"Failed to read motion blur mask: {self.moblurMaskPath}. Ignoring."
            )
            return None

        if maskImg.shape[:2] != (self.height, self.width):
            logging.info(
                f"Resizing motion blur mask from {maskImg.shape[1]}x{maskImg.shape[0]} "
                f"to {self.width}x{self.height}"
            )
            maskImg = cv2.resize(
                maskImg, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )

        norm = 65535.0 if maskImg.dtype == "uint16" else 255.0

        if maskImg.ndim == 3 and maskImg.shape[2] == 4:
            bgr = maskImg[:, :, :3]
            alpha = maskImg[:, :, 3]
            luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lumaT = torch.from_numpy(luma).to(device=device, dtype=dtype).div_(norm)
            alphaT = torch.from_numpy(alpha).to(device=device, dtype=dtype).div_(norm)
            protection = alphaT * (1.0 - lumaT)
            channels = "RGBA"
        elif maskImg.ndim == 3:
            luma = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
            lumaT = torch.from_numpy(luma).to(device=device, dtype=dtype).div_(norm)
            protection = 1.0 - lumaT
            channels = "RGB (no alpha, assuming opaque)"
        else:
            gray = torch.from_numpy(maskImg).to(device=device, dtype=dtype).div_(norm)
            protection = 1.0 - gray
            channels = "grayscale"

        protection = protection.clamp_(0.0, 1.0)
        weightTensor = (1.0 - protection).view(1, 1, self.height, self.width)

        logging.info(
            f"Loaded motion blur mask: {self.moblurMaskPath} ({channels})"
        )
        return weightTensor

    def _computeWindow(self):
        """Resolve shutter window → sample counts from prev/next segments.

        Timeline around curr frame at offset 0:
            prevSeg samples at offset -1 + k/factor for k in [1..factor-1]
            curr at 0
            nextSeg samples at offset  k/factor for k in [1..factor-1]

        Shutter window [-halfWindow, +halfWindow] with halfWindow = angle/720.
        """
        factor = self.interpolateFactor
        framesToInsert = factor - 1
        halfWindow = self.shutterAngle / 720.0

        # prev segment: keep samples with k ≥ ceil(factor*(1-halfWindow))
        kStart = max(1, math.ceil(factor * (1.0 - halfWindow)))
        prevSegStart = min(framesToInsert, kStart - 1)
        leftCount = framesToInsert - prevSegStart

        # next segment: keep samples with k ≤ floor(factor*halfWindow)
        nextSegCount = min(framesToInsert, math.floor(factor * halfWindow))

        totalSamples = leftCount + 1 + nextSegCount
        noBlur = leftCount == 0 and nextSegCount == 0
        return framesToInsert, prevSegStart, leftCount, nextSegCount, totalSamples, noBlur

    def _processFrames(self):
        (
            framesToInsert,
            prevSegStart,
            leftCount,
            nextSegCount,
            totalSamples,
            noBlur,
        ) = self._computeWindow()

        dtype = torch.float16 if self.half else torch.float32
        device = torch.device("cuda" if self.useCuda else "cpu")

        weights = generateWeights(totalSamples, self.moblurStrength)
        blender = GammaCorrectBlender(weights, dtype, device, gamma=self.moblurGamma)

        self.mask = self._loadMask(dtype, device)

        logging.info(
            f"Motion blur window: {totalSamples} samples "
            f"({leftCount} prev + 1 curr + {nextSegCount} next)"
            + (" [no-blur pass-through]" if noBlur else "")
            + (" [masked]" if self.mask is not None else "")
        )

        collector = FrameCollector()

        prevFrame = None
        currFrame = None
        prevSeg = None

        with ProgressBarLogic(self.totalFrames, title=self.input) as bar:
            for _ in range(self.totalFrames):
                nextFrame = self.readBuffer.read()
                if nextFrame is None:
                    break

                if prevFrame is None:
                    # Prime interp state (firstRun stores I0, returns nothing).
                    collector.clear()
                    self.interpolateProcess(nextFrame, collector, framesToInsert, None)
                    prevFrame = nextFrame
                    self.writeBuffer.write(nextFrame)
                    bar(1)
                    continue

                collector.clear()
                self.interpolateProcess(nextFrame, collector, framesToInsert, None)
                newSeg = list(collector.frames)

                if currFrame is None:
                    # Bootstrap: need both neighboring segments before first blur.
                    prevSeg = newSeg
                    currFrame = nextFrame
                    continue

                if noBlur:
                    self.writeBuffer.write(currFrame)
                else:
                    window = []
                    if leftCount > 0 and prevSeg is not None:
                        window.extend(prevSeg[prevSegStart:])
                    window.append(currFrame)
                    if nextSegCount > 0:
                        window.extend(newSeg[:nextSegCount])

                    blended = blender(window)

                    if self.mask is not None:
                        blended = blended * self.mask + currFrame * (1.0 - self.mask)

                    if self.useCuda:
                        torch.cuda.current_stream().synchronize()

                    self.writeBuffer.write(blended)

                bar(1)

                prevFrame = currFrame
                currFrame = nextFrame
                prevSeg = newSeg

                if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                    break

            # Final frame: no nextSeg available, emit pristine.
            if currFrame is not None:
                self.writeBuffer.write(currFrame)
                bar(1)

        self.writeBuffer.close()
