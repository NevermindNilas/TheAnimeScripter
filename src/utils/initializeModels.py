import logging


def ObjectDetection(self):
    from src.objectDetection.objectDetection import ObjectDetection

    ObjectDetection(
        self.input,
        self.output,
        self.width,
        self.height,
        self.fps,
        self.inpoint,
        self.outpoint,
        self.encode_method,
        self.custom_encoder,
        self.benchmark,
        self.totalFrames,
        self.half,
    )


def AutoClip(self):
    from src.autoclip.autoclip import AutoClip

    AutoClip(
        self.input,
        self.autoclip_sens,
        self.inpoint,
        self.outpoint,
    )


def Segment(self):
    if self.segment_method == "anime":
        from src.segment.animeSegment import AnimeSegment

        AnimeSegment(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encode_method,
            self.custom_encoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segment_method == "anime-tensorrt":
        from src.segment.animeSegment import AnimeSegmentTensorRT

        AnimeSegmentTensorRT(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encode_method,
            self.custom_encoder,
            self.benchmark,
            self.totalFrames,
        )

    elif self.segment_method == "anime-directml":
        from src.segment.animeSegment import AnimeSegmentDirectML

        AnimeSegmentDirectML(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encode_method,
            self.custom_encoder,
            self.benchmark,
            self.totalFrames,
        )

    elif self.segment_method == "cartoon":
        raise NotImplementedError("Cartoon segment is not implemented yet")


def Depth(self):
    match self.depth_method:
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
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
            )

        case "small_v2-tensorrt" | "base_v2-tensorrt" | "large_v2-tensorrt":
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
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
            )

        case "small_v2-directml" | "base_v2-directml" | "large_v2-directml":
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
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
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
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
            )


def initializeModels(self):
    outputWidth = self.width
    outputHeight = self.height
    upscale_process = None
    interpolate_process = None
    restore_process = None
    dedup_process = None
    scenechange_process = None

    if self.upscale:
        from src.unifiedUpscale import UniversalPytorch

        outputWidth *= self.upscale_factor
        outputHeight *= self.upscale_factor
        logging.info(f"Upscaling to {outputWidth}x{outputHeight}")
        match self.upscale_method:
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
            ):
                upscale_process = UniversalPytorch(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
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
            ):
                from src.unifiedUpscale import UniversalDirectML

                upscale_process = UniversalDirectML(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                )

            case "shufflecugan-ncnn" | "span-ncnn":
                from src.unifiedUpscale import UniversalNCNN

                upscale_process = UniversalNCNN(
                    self.upscale_method,
                    self.upscale_factor,
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
            ):
                from src.unifiedUpscale import UniversalTensorRT

                upscale_process = UniversalTensorRT(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.forceStatic,
                )
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolate_factor, '.3f')}fps"
        )
        match self.interpolate_method:
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

                interpolate_process = RifeCuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolate_method,
                    self.ensemble,
                    self.interpolate_factor,
                    self.dynamic_scale,
                    self.static_step,
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

                interpolate_process = RifeNCNN(
                    self.interpolate_method,
                    self.ensemble,
                    self.width,
                    self.height,
                    self.half,
                    self.interpolate_factor,
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

                interpolate_process = RifeTensorRT(
                    self.interpolate_method,
                    self.interpolate_factor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "gmfss":
                from src.gmfss.gmfss import GMFSS

                interpolate_process = GMFSS(
                    int(self.interpolate_factor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    self.ensemble,
                )

            case "gmfss-tensorrt":
                from src.gmfss.gmfss import GMFSSTensorRT

                interpolate_process = GMFSSTensorRT(
                    int(self.interpolate_factor),
                    outputWidth,
                    outputHeight,
                    self.half,
                    self.ensemble,
                )

            case "rife4.6-directml":
                from src.unifiedInterpolate import RifeDirectML

                interpolate_process = RifeDirectML(
                    self.interpolate_method,
                    self.interpolate_factor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

    if self.restore:
        match self.restore_method:
            case (
                "scunet"
                | "dpir"
                | "nafnet"
                | "real-plksr"
                | "anime1080fixer"
                | "gater3"
            ):
                from src.unifiedRestore import UnifiedRestoreCuda

                restore_process = UnifiedRestoreCuda(
                    self.restore_method,
                    self.half,
                )

            case "anime1080fixer-tensorrt" | "gater3-tensorrt" | "scunet-tensorrt":
                from src.unifiedRestore import UnifiedRestoreTensorRT

                restore_process = UnifiedRestoreTensorRT(
                    self.restore_method,
                    self.half,
                    self.width,
                    self.height,
                    self.forceStatic,
                )

            case "anime1080fixer-directml" | "gater3-directml" | "scunet-directml":
                from src.unifiedRestore import UnifiedRestoreDirectML

                restore_process = UnifiedRestoreDirectML(
                    self.restore_method,
                    self.half,
                    self.width,
                    self.height,
                )
            case "fastlinedarken":
                from src.fastlinedarken import FastLineDarkenWithStreams

                restore_process = FastLineDarkenWithStreams(
                    self.half,
                )
            case "fastlinedarken-tensorrt":
                from src.fastlinedarken import FastLineDarkenTRT

                restore_process = FastLineDarkenTRT(
                    self.half,
                    self.height,
                    self.width,
                )

    if self.dedup:
        match self.dedup_method:
            case "ssim":
                from src.dedup.dedup import DedupSSIM

                dedup_process = DedupSSIM(
                    self.dedup_sens,
                )

            case "mse":
                from src.dedup.dedup import DedupMSE

                dedup_process = DedupMSE(
                    self.dedup_sens,
                )

            case "ssim-cuda":
                from src.dedup.dedup import DedupSSIMCuda

                dedup_process = DedupSSIMCuda(
                    self.dedup_sens,
                    self.half,
                )

            case "mse-cuda":
                from src.dedup.dedup import DedupMSECuda

                dedup_process = DedupMSECuda(
                    self.dedup_sens,
                    self.half,
                )

            case "flownets":
                from src.dedup.dedup import DedupFlownetS

                dedup_process = DedupFlownetS(
                    half=self.half,
                    dedupSens=self.dedup_sens,
                    height=self.height,
                    width=self.width,
                )

    if self.scenechange:
        match self.scenechange_method:
            case "maxxvit-tensorrt" | "shift_lpips-tensorrt":
                from src.scenechange import SceneChangeTensorRT

                scenechange_process = SceneChangeTensorRT(
                    self.half,
                    self.scenechange_sens,
                    self.scenechange_method,
                )
            case "maxxvit-directml":
                from src.scenechange import SceneChange

                scenechange_process = SceneChange(
                    self.half,
                    self.scenechange_sens,
                )
            case "differential":
                from src.scenechange import SceneChangeCPU

                scenechange_process = SceneChangeCPU(
                    self.scenechange_sens,
                )
            case "differential-cuda":
                from src.scenechange import SceneChangeCuda

                scenechange_process = SceneChangeCuda(
                    self.scenechange_sens,
                )
            case "differential-tensorrt":
                from src.scenechange import DifferentialTensorRT

                scenechange_process = DifferentialTensorRT(
                    self.scenechange_sens,
                    self.height,
                    self.width,
                )
            case "differential-directml":
                # from src.scenechange import DifferentialDirectML
                # scenechange_process = DifferentialDirectML(
                #     self.scenechange_sens,
                # )
                raise NotImplementedError(
                    "Differential DirectML is not implemented yet"
                )
            case _:
                raise ValueError(
                    f"Unknown scenechange method: {self.scenechange_method}"
                )

    return (
        outputWidth,
        outputHeight,
        upscale_process,
        interpolate_process,
        restore_process,
        dedup_process,
        scenechange_process,
    )
