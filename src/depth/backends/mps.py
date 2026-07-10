import os
import sys

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.depth.backends._shared import SlidingWindowNormalizer, calculateAspectRatio
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import BuildBuffer, WriteBuffer
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState


class DepthMPS:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small_v2-mps",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depthNorm: bool = False,
        depth_batch: int = 1,
    ):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS depth backend requested, but torch.mps is unavailable"
            )

        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.baseMethod = depth_method.replace("-mps", "")
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.compileMode = compileMode
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None
        self.depthBatch = max(1, int(depth_batch))
        self.device = torch.device("mps")

        self.handleModels()

        self.meanTensor = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
            1, 3, 1, 1
        )
        self.stdTensor = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
            1, 3, 1, 1
        )
        if self.dtype == torch.float16:
            self.meanTensor = self.meanTensor.half()
            self.stdTensor = self.stdTensor.half()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                half=self.half,
                resize=True,
                width=self.newWidth,
                height=self.newHeight,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                grayscale=True,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def _finalizeModelPrecision(self):
        self.model = self.model.eval()
        if self.half:
            try:
                self.model = self.model.half()
                self.dtype = torch.float16
            except Exception as e:
                logging.warning(f"Falling back to fp32 for MPS depth model: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32
        else:
            self.model = self.model.float()
            self.dtype = torch.float32

        self.model = self.model.to(self.device)

        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on MPS depth backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

    def _warmup(self):
        with torch.inference_mode():
            dummy = torch.zeros(
                (1, 3, self.newHeight, self.newWidth),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(2):
                _ = self.model(dummy)
            torch.mps.synchronize()

    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS depth model: {self.depth_method}..."}
            )

        from ..dpt_v2 import DepthAnythingV2

        match self.baseMethod:
            case "small_v2" | "distill_small_v2":
                method = "vits"
            case "base_v2" | "distill_base_v2":
                method = "vitb"
            case "large_v2" | "distill_large_v2":
                method = "vitl"
            case "giant_v2":
                method = "vitg"
            case _:
                raise ValueError(f"Unsupported MPS depth method: {self.depth_method}")

        self.filename = modelsMap(
            model=self.baseMethod, modelType="pth", half=self.half
        )
        modelPath = resolveWeightPath(
            self.baseMethod,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        modelConfigs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            }
            if "distill" not in self.baseMethod
            else {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "max_depth": 150.0,
                "mode": "disparity",
                "pretrain_type": "dinov2",
                "del_mask_token": False,
            },
        }

        if "distill" in self.baseMethod and "large" in self.baseMethod:
            from src.depth.distillanydepth.modeling.archs.dam.dam import DepthAnything

            self.model = DepthAnything(**modelConfigs[method])
        else:
            self.model = DepthAnythingV2(**modelConfigs[method])

        if "distill" in self.baseMethod:
            from safetensors.torch import load_file

            modelWeights = load_file(modelPath)
            self.model.load_state_dict(modelWeights)
            del modelWeights
        else:
            stateDict = torch.load(modelPath, map_location="cpu")
            self.model.load_state_dict(stateDict)
            del stateDict

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        self._finalizeModelPrecision()
        self._warmup()

    @torch.inference_mode()
    def normFrame(self, frame):
        frame = frame.to(device=self.device, dtype=self.dtype)
        return (frame - self.meanTensor) / self.stdTensor

    @torch.inference_mode()
    def _normalizeDepth(self, depth):
        if self.normalizer is not None:
            return self.normalizer.normalize(depth)
        return (depth - depth.min()) / (depth.max() - depth.min())

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            batch = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
            batch = self.normFrame(batch)
            depth = self.model(batch)
            depth = F.interpolate(
                depth,
                (self.height, self.width),
                mode="bilinear",
                align_corners=True,
            ).cpu()
            torch.mps.synchronize()
            for i in range(depth.shape[0]):
                self.writeBuffer.write(self._normalizeDepth(depth[i : i + 1]))
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        with ProgressBarLogic(self.totalFrames) as bar:
            while True:
                frames = []
                for _ in range(self.depthBatch):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if not frames:
                    break
                self.processBatch(frames)
                frameCount += len(frames)
                bar(len(frames))
                if len(frames) < self.depthBatch:
                    break

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()


class OGDepthV2MPS:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="og_small_v2-mps",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depthNorm: bool = False,
        depth_batch: int = 1,
    ):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS depth backend requested, but torch.mps is unavailable"
            )

        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.baseMethod = depth_method.replace("-mps", "")
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.compileMode = compileMode
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None
        self.depthBatch = max(1, int(depth_batch))
        self.encodeBuffer = Queue(maxsize=10)
        self.device = torch.device("mps")

        self.handleModels()

        if not hasattr(self, "newHeight") or not hasattr(self, "newWidth"):
            self.newHeight, self.newWidth = calculateAspectRatio(
                self.width, self.height, self.depthQuality
            )
        decodeW = getattr(self, "_decodeWidth", self.width)
        decodeH = getattr(self, "_decodeHeight", self.height)
        decodeResize = getattr(self, "_decodeResize", False)
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=decodeW,
                height=decodeH,
                half=self.half,
                resize=decodeResize,
                toTorch=False,
            )

            self.output = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.encodeThread)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def _finalizeModelPrecision(self):
        self.model = self.model.eval()
        if self.half:
            try:
                self.model = self.model.half()
                self.dtype = torch.float16
            except Exception as e:
                logging.warning(f"Falling back to fp32 for MPS depth model: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32
        else:
            self.model = self.model.float()
            self.dtype = torch.float32

        self.model = self.model.to(self.device)

        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on MPS depth backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

    def handleModels(self):
        from ..og_dpt_v2 import DepthAnythingV2

        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS depth model: {self.depth_method}..."}
            )

        match self.baseMethod:
            case "og_small_v2" | "og_distill_small_v2":
                method = "vits"
                toDownload = "small_v2"
            case "og_base_v2" | "og_distill_base_v2":
                method = "vitb"
                toDownload = "base_v2"
            case "og_large_v2" | "og_distill_large_v2":
                method = "vitl"
                toDownload = "large_v2"
            case "og_giant_v2":
                method = "vitg"
                toDownload = "giant_v2"
            case _:
                raise ValueError(f"Unsupported MPS depth method: {self.depth_method}")

        modelType = "pth"
        self.filename = modelsMap(model=toDownload, modelType=modelType, half=self.half)
        modelPath = resolveWeightPath(
            toDownload,
            self.filename,
            half=self.half,
            modelType=modelType,
        )

        modelConfigs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            }
            if "distill" not in self.baseMethod
            else {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "max_depth": 150.0,
                "mode": "disparity",
                "pretrain_type": "dinov2",
                "del_mask_token": False,
            },
        }

        if "distill" in self.baseMethod and "large" in self.baseMethod:
            from src.depth.distillanydepth.modeling.archs.dam.dam import DepthAnything

            self.model = DepthAnything(**modelConfigs[method])
        else:
            self.model = DepthAnythingV2(**modelConfigs[method])

        self.model.load_state_dict(torch.load(modelPath, map_location="cpu"))

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        self._finalizeModelPrecision()

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            tensors = []
            sizes = []
            for frame in frames:
                image, (h, w) = self.model.image2tensor(frame, self.newHeight)
                image = image.to(self.device, dtype=self.dtype)
                tensors.append(image)
                sizes.append((h, w))

            batch = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            depth = self.model.forward(batch)
            torch.mps.synchronize()
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            else:
                depth = depth[:, :1]
            depth = depth.cpu()

            for i, (h, w) in enumerate(sizes):
                d = F.interpolate(
                    depth[i : i + 1], (h, w), mode="bilinear", align_corners=True
                )
                d = d[0, 0]
                if self.normalizer is not None:
                    d = self.normalizer.normalize(d)
                else:
                    d = (d - d.min()) / (d.max() - d.min())
                d = (d * 255.0).byte()
                d = d.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
                self.encodeBuffer.put(d)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        with ProgressBarLogic(self.totalFrames) as bar:
            while True:
                frames = []
                for _ in range(self.depthBatch):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if not frames:
                    break
                self.processBatch(frames)
                frameCount += len(frames)
                bar(len(frames))
                if len(frames) < self.depthBatch:
                    break

        logging.info(f"Processed {frameCount} frames")
        self.encodeBuffer.put(None)

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            self.output.write(frame)

        self.output.release()


class OGDepthV3MPS(OGDepthV2MPS):
    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS depth model: {self.depth_method}..."}
            )

        from .. import depth_anything_3 as depth_anything_3_pkg

        sys.modules.setdefault("depth_anything_3", depth_anything_3_pkg)
        MonocularDepthAnything3 = importlib.import_module(
            "depth_anything_3.mono"
        ).MonocularDepthAnything3

        toDownload = self.baseMethod
        modelMap = {
            "small_v3": "da3-small",
            "base_v3": "da3-base",
            "large_v3": "da3mono-large",
            "og_large_v3": "da3metric-large",
        }
        modelName = modelMap[self.baseMethod]

        self.filename = modelsMap(model=toDownload, modelType="pth", half=self.half)
        modelPath = resolveWeightPath(
            toDownload,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        self.model = MonocularDepthAnything3.from_pretrained(
            modelPath,
            model_name=modelName,
            strict=False,
        ).to(self.device)

        from ..fold_layerscale import fold_layerscale_

        fold_layerscale_(self.model)

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        self.processRes = calculateAspectRatio(
            self.width, self.height, self.depthQuality, True
        )
        self.processResMethod = (
            "lower_bound_resize"
            if self.depthQuality == "high"
            else "upper_bound_resize"
        )

        if self.processResMethod == "upper_bound_resize":
            scale = self.processRes / max(self.width, self.height)
        else:
            scale = self.processRes / min(self.width, self.height)
        tgt_w = max(14, (max(1, round(self.width * scale)) // 14) * 14)
        tgt_h = max(14, (max(1, round(self.height * scale)) // 14) * 14)
        self._decodeWidth = tgt_w
        self._decodeHeight = tgt_h
        self._decodeResize = True

        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on MPS depth backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

    @torch.inference_mode()
    def _inferBatch(self, frames):
        imgsList = []
        sizes = []
        for frame in frames:
            imgsCpu, _, _ = self.model.input_processor(
                [frame], None, None, self.processRes, self.processResMethod
            )
            imgsList.append(imgsCpu)
            sizes.append(frame.shape[:2])

        batch = torch.stack(imgsList, dim=0).to(self.device).float()
        prediction = self.model.output_processor(self.model.forward(batch))
        torch.mps.synchronize()

        depths = []
        for i, (h, w) in enumerate(sizes):
            depth = np.asarray(prediction.depth[i]).squeeze()
            if depth.shape[0] != h or depth.shape[1] != w:
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            depths.append(
                np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            )
        return depths

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            rawDepths = self._inferBatch(frames)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")
            return

        for depth in rawDepths:
            try:
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                validMask = depth > 0

                if validMask.sum() <= 10:
                    gray = np.zeros(depth.shape, dtype=np.uint8)
                    self.encodeBuffer.put(np.stack([gray] * 3, axis=-1))
                    continue

                disparity = np.zeros_like(depth, dtype=np.float32)
                disparity[validMask] = 1.0 / depth[validMask]

                if self.normalizer is not None:
                    gray = self.normalizer.normalize(disparity)
                else:
                    disp_min = np.percentile(disparity[validMask], 2)
                    disp_max = np.percentile(disparity[validMask], 98)
                    if disp_min == disp_max:
                        disp_min -= 1e-6
                        disp_max += 1e-6
                    gray = ((disparity - disp_min) / (disp_max - disp_min)).clip(0, 1)
                gray = (gray * 255.0).astype(np.uint8)
                self.encodeBuffer.put(np.stack([gray] * 3, axis=-1))
            except Exception as e:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            self.output.write(frame)

        self.output.release()
