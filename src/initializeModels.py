"""
Model Initialization and Processing Functions

This module handles the initialization and execution of various AI models
for video processing operations including object detection, auto-clipping,
segmentation, depth estimation, and the main processing pipeline.
"""

import logging


def objectDetection(self):
    """
    Initialize and execute object detection processing.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    if "directml" in self.objDetectMethod:
        from src.objectDetection.objectDetection import ObjectDetectionDML

        ObjectDetectionDML(
            self.input,
            self.output,
            self.width,
            self.height,
            self.fps,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.half,
            self.objDetectMethod,
            self.totalFrames,
        )
    elif "tensorrt" in self.objDetectMethod:
        from src.objectDetection.objectDetection import ObjectDetectionTensorRT

        ObjectDetectionTensorRT(
            self.input,
            self.output,
            self.width,
            self.height,
            self.fps,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.half,
            self.objDetectMethod,
            self.totalFrames,
        )
    else:
        from src.objectDetection.objectDetection import ObjectDetection

        ObjectDetection(
            self.input,
            self.output,
            self.width,
            self.height,
            self.fps,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.half,
        )


def autoClip(self):
    """
    Initialize and execute automatic scene detection and clipping.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    from src.autoclip.autoclip import AutoClip

    AutoClip(
        self.input,
        self.autoclipSens,
        self.inpoint,
        self.outpoint,
    )


def segment(self):
    """
    Initialize and execute video segmentation processing.

    Args:
        self: VideoProcessor instance containing processing parameters

    Raises:
        NotImplementedError: If cartoon segmentation method is selected
    """
    if self.segmentMethod == "anime":
        from src.segment.animeSegment import AnimeSegment

        AnimeSegment(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segmentMethod == "anime-tensorrt":
        from src.segment.animeSegment import AnimeSegmentTensorRT

        AnimeSegmentTensorRT(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segmentMethod == "anime-directml":
        from src.segment.animeSegment import AnimeSegmentDirectML

        AnimeSegmentDirectML(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segmentMethod == "cartoon":
        raise NotImplementedError("Cartoon segment is not implemented yet")


def depth(self):
    """
    Initialize and execute depth estimation processing.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    match self.depthMethod:
        case (
            "small_v2"
            | "base_v2"
            | "large_v2"
            | "distill_small_v2"
            | "distill_base_v2"
            | "distill_large_v2"
        ):
            from src.depth.depth import DepthCuda

            DepthCuda(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case (
            "small_v2-tensorrt"
            | "base_v2-tensorrt"
            | "large_v2-tensorrt"
            | "distill_small_v2-tensorrt"
            | "distill_base_v2-tensorrt"
            | "distill_large_v2-tensorrt"
        ):
            from src.depth.depth import DepthTensorRTV2

            DepthTensorRTV2(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )

        case (
            "small_v2-directml"
            | "base_v2-directml"
            | "large_v2-directml"
            | "distill_small_v2-directml"
            | "distill_base_v2-directml"
            | "distill_large_v2-directml"
        ):
            from src.depth.depth import DepthDirectMLV2

            DepthDirectMLV2(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )
        case (
            "og_small_v2"
            | "og_base_v2"
            | "og_large_v2"
            | "og_distill_small_v2"
            | "og_distill_base_v2"
            | "og_distill_large_v2"
        ):
            from src.depth.depth import OGDepthV2CUDA

            OGDepthV2CUDA(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case "og_video_small_v2":
            from src.depth.depth import VideoDepthAnythingCUDA

            VideoDepthAnythingCUDA(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case (
            "og_small_v2-tensorrt"
            | "og_base_v2-tensorrt"
            | "og_large_v2-tensorrt"
            | "og_distill_small_v2-tensorrt"
            | "og_distill_base_v2-tensorrt"
            | "og_distill_large_v2-tensorrt"
        ):
            from src.depth.depth import OGDepthV2TensorRT

            OGDepthV2TensorRT(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )


def initializeModels(self):
    """
    Initialize all AI models for the video processing pipeline.

    Args:
        self: VideoProcessor instance containing processing parameters

    Returns:
        tuple: Contains output dimensions and initialized processing functions
            - outputWidth (int): Final output video width
            - outputHeight (int): Final output video height
            - upscaleProcess: Upscaling model function or None
            - interpolateProcess: Interpolation model function or None
            - restoreProcess: Restoration model function or None
            - dedupProcess: Deduplication function or None
            - scenechangeProcess: Scene change detection function or None
    """
    outputWidth = self.width
    outputHeight = self.height
    upscaleProcess = None
    interpolateProcess = None
    restoreProcess = None
    dedupProcess = None
    scenechangeProcess = None

    if self.upscale:
        from src.unifiedUpscale import UniversalPytorch

        outputWidth *= self.upscaleFactor
        outputHeight *= self.upscaleFactor
        logging.info(f"Upscaling to {outputWidth}x{outputHeight}")
        match self.upscaleMethod:
            case (
                "shufflecugan"
                | "compact"
                | "ultracompact"
                | "superultracompact"
                | "span"
                | "open-proteus"
                | "aniscale2"
                | "shufflespan"
                | "rtmosr"
                | "fallin_soft"
                | "fallin_strong"
            ):
                upscaleProcess = UniversalPytorch(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                    self.compileMode,
                )

            case (
                "compact-directml"
                | "ultracompact-directml"
                | "superultracompact-directml"
                | "span-directml"
                | "open-proteus-directml"
                | "aniscale2-directml"
                | "shufflespan-directml"
                | "rtmosr-directml"
                | "fallin_soft-directml"
                | "fallin_strong-directml"
            ):
                from src.unifiedUpscale import UniversalDirectML

                upscaleProcess = UniversalDirectML(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                )

            case "shufflecugan-ncnn" | "span-ncnn":
                from src.unifiedUpscale import UniversalNCNN

                upscaleProcess = UniversalNCNN(
                    self.upscaleMethod,
                    self.upscaleFactor,
                )

            case (
                "shufflecugan-tensorrt"
                | "compact-tensorrt"
                | "ultracompact-tensorrt"
                | "superultracompact-tensorrt"
                | "span-tensorrt"
                | "open-proteus-tensorrt"
                | "aniscale2-tensorrt"
                | "shufflespan-tensorrt"
                | "rtmosr-tensorrt"
                | "fallin_soft-tensorrt"
                | "fallin_strong-tensorrt"
            ):
                from src.unifiedUpscale import UniversalTensorRT

                upscaleProcess = UniversalTensorRT(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                    self.forceStatic,
                )
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolateFactor, '.3f')}fps"
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

                interpolateProcess = RifeCuda(
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

                interpolateProcess = RifeNCNN(
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

                interpolateProcess = RifeTensorRT(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "gmfss":
                from src.gmfss.gmfss import GMFSS

                interpolateProcess = GMFSS(
                    int(self.interpolateFactor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    self.ensemble,
                    compileMode=self.compileMode,
                )

            case "gmfss-tensorrt":
                from src.gmfss.gmfss import GMFSSTensorRT

                interpolateProcess = GMFSSTensorRT(
                    int(self.interpolateFactor),
                    outputWidth,
                    outputHeight,
                    self.half,
                    self.ensemble,
                )

            case "rife4.6-directml":
                from src.unifiedInterpolate import RifeDirectML

                interpolateProcess = RifeDirectML(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

    if self.restore:
        match self.restoreMethod:
            case (
                "scunet"
                | "dpir"
                | "nafnet"
                | "real-plksr"
                | "anime1080fixer"
                | "gater3"
                | "deh264_real"
                | "deh264_span"
            ):
                from src.unifiedRestore import UnifiedRestoreCuda

                restoreProcess = UnifiedRestoreCuda(
                    self.restoreMethod,
                    self.half,
                )

            case (
                "anime1080fixer-tensorrt"
                | "gater3-tensorrt"
                | "scunet-tensorrt"
                | "codeformer-tensorrt"
                | "deh264_real-tensorrt"
                | "deh264_span-tensorrt"
            ):
                from src.unifiedRestore import UnifiedRestoreTensorRT

                restoreProcess = UnifiedRestoreTensorRT(
                    self.restoreMethod,
                    self.half,
                    self.width,
                    self.height,
                    self.forceStatic,
                )

            case (
                "anime1080fixer-directml"
                | "gater3-directml"
                | "scunet-directml"
                | "codeformer-directml"
                | "deh264_real-directml"
                | "deh264_span-directml"
            ):
                from src.unifiedRestore import UnifiedRestoreDirectML

                restoreProcess = UnifiedRestoreDirectML(
                    self.restoreMethod,
                    self.half,
                    self.width,
                    self.height,
                )
            case "fastlinedarken":
                from src.fastlinedarken import FastLineDarkenWithStreams

                restoreProcess = FastLineDarkenWithStreams(
                    self.half,
                )
            case "fastlinedarken-tensorrt":
                from src.fastlinedarken import FastLineDarkenTRT

                restoreProcess = FastLineDarkenTRT(
                    self.half,
                    self.height,
                    self.width,
                )

    if self.dedup:
        match self.dedupMethod:
            case "ssim":
                from src.dedup.dedup import DedupSSIM

                dedupProcess = DedupSSIM(
                    self.dedupSens,
                )

            case "mse":
                from src.dedup.dedup import DedupMSE

                dedupProcess = DedupMSE(
                    self.dedupSens,
                )

            case "ssim-cuda":
                from src.dedup.dedup import DedupSSIMCuda

                dedupProcess = DedupSSIMCuda(
                    self.dedupSens,
                    self.half,
                )

            case "mse-cuda":
                from src.dedup.dedup import DedupMSECuda

                dedupProcess = DedupMSECuda(
                    self.dedupSens,
                    self.half,
                )

            case "flownets":
                from src.dedup.dedup import DedupFlownetS

                dedupProcess = DedupFlownetS(
                    half=self.half,
                    dedupSens=self.dedupSens,
                    height=self.height,
                    width=self.width,
                )

    if self.scenechange:
        match self.scenechangeMethod:
            case "maxxvit-tensorrt" | "shift_lpips-tensorrt":
                from src.scenechange import SceneChangeTensorRT

                scenechangeProcess = SceneChangeTensorRT(
                    self.half,
                    self.scenechangeSens,
                    self.scenechangeMethod,
                )
            case "maxxvit-directml":
                from src.scenechange import SceneChange

                scenechangeProcess = SceneChange(
                    self.half,
                    self.scenechangeSens,
                )
            case "differential":
                from src.scenechange import SceneChangeCPU

                scenechangeProcess = SceneChangeCPU(
                    self.scenechangeSens,
                )
            case "differential-cuda":
                from src.scenechange import SceneChangeCuda

                scenechangeProcess = SceneChangeCuda(
                    self.scenechangeSens,
                )
            case "differential-tensorrt":
                from src.scenechange import DifferentialTensorRT

                scenechangeProcess = DifferentialTensorRT(
                    self.scenechangeSens,
                    self.height,
                    self.width,
                )
            case "differential-directml":
                # from src.scenechange import DifferentialDirectML
                # scenechangeProcess = DifferentialDirectML(
                #     self.scenechangeSens,
                # )
                raise NotImplementedError(
                    "Differential DirectML is not implemented yet"
                )
            case _:
                raise ValueError(
                    f"Unknown scenechange method: {self.scenechangeMethod}"
                )

    return (
        outputWidth,
        outputHeight,
        upscaleProcess,
        interpolateProcess,
        restoreProcess,
        dedupProcess,
        scenechangeProcess,
    )
