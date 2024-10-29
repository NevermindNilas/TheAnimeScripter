import logging


def AutoClip(self, mainPath):
    from src.autoclip.autoclip import AutoClip

    AutoClip(
        self.input,
        self.autoclip_sens,
        mainPath,
        self.inpoint,
        self.outpoint,
    )


def Segment(self, mainPath):
    if self.segment_method == "anime":
        from src.segment.animeSegment import AnimeSegment

        AnimeSegment(
            self.input,
            self.output,
            self.ffmpeg_path,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encode_method,
            self.custom_encoder,
            self.buffer_limit,
            self.benchmark,
            self.totalFrames,
            mainPath=mainPath,
            decodeThreads=self.decode_threads,
        )
    elif self.segment_method == "anime-tensorrt":
        from src.segment.animeSegment import AnimeSegmentTensorRT

        AnimeSegmentTensorRT(
            self.input,
            self.output,
            self.ffmpeg_path,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encode_method,
            self.custom_encoder,
            self.buffer_limit,
            self.benchmark,
            self.totalFrames,
            mainPath=mainPath,
            decodeThreads=self.decode_threads,
        )

    elif self.segment_method == "anime-directml":
        from src.segment.animeSegment import AnimeSegmentDirectML

        AnimeSegmentDirectML(
            self.input,
            self.output,
            self.ffmpeg_path,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encode_method,
            self.custom_encoder,
            self.buffer_limit,
            self.benchmark,
            self.totalFrames,
            mainPath=mainPath,
            decodeThreads=self.decode_threads,
        )

    elif self.segment_method == "cartoon":
        raise NotImplementedError("Cartoon segment is not implemented yet")


def Depth(self, mainPath):
    match self.depth_method:
        case "small_v2" | "base_v2" | "large_v2":
            from src.depth.depth import DepthCuda

            DepthCuda(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.buffer_limit,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
                mainPath=mainPath,
                decodeThreads=self.decode_threads,
            )

        case "small_v2-tensorrt" | "base_v2-tensorrt" | "large_v2-tensorrt":
            from src.depth.depth import DepthTensorRTV2

            DepthTensorRTV2(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.buffer_limit,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
                mainPath=mainPath,
                decodeThreads=self.decode_threads,
            )

        case "small_v2-directml" | "base_v2-directml" | "large_v2-directml":
            from src.depth.depth import DepthDirectMLV2

            DepthDirectMLV2(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.buffer_limit,
                self.benchmark,
                self.totalFrames,
                self.bit_depth,
                self.depth_quality,
                mainPath=mainPath,
                decodeThreads=self.decode_threads,
            )


def initializeModels(self):
    outputWidth = self.width
    outputHeight = self.height
    upscale_process = None
    interpolate_process = None
    restore_process = None
    dedup_process = None
    scenechange_process = None
    upscaleSkipProcess = None
    interpolateSkipProcess = None

    if self.upscale:
        if self.upscale_skip:
            if not "ncnn" or "directml" in self.upscale_method:
                from src.dedup.dedup import DedupSSIMCuda

                upscaleSkipProcess = DedupSSIMCuda(
                    0.999,
                    1080
                    if max(self.width, self.height) >= 1080
                    else max(self.width, self.height),
                    half=self.half,
                )
            else:
                from src.dedup.dedup import DedupSSIM

                upscaleSkipProcess = DedupSSIM(
                    0.999,
                    1080
                    if max(self.width, self.height) >= 1080
                    else max(self.width, self.height),
                )
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
            ):
                upscale_process = UniversalPytorch(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    upscaleSkipProcess,
                )

            case (
                "compact-directml"
                | "ultracompact-directml"
                | "superultracompact-directml"
                | "span-directml"
                | "shufflecugan-directml"
                | "open-proteus-directml"
                | "aniscale2-directml"
                | "shufflespan-directml"
            ):
                from src.unifiedUpscale import UniversalDirectML

                upscale_process = UniversalDirectML(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    upscaleSkipProcess,
                )

            case "shufflecugan-ncnn" | "span-ncnn":
                from src.unifiedUpscale import UniversalNCNN

                upscale_process = UniversalNCNN(
                    self.upscale_method,
                    self.upscale_factor,
                    upscaleSkipProcess,
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
            ):
                from src.unifiedUpscale import UniversalTensorRT

                upscale_process = UniversalTensorRT(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    upscaleSkipProcess,
                    self.forceStatic,
                )
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolate_factor, '.3f')}fps"
        )

        if self.interpolate_skip:
            from src.dedup.dedup import DedupSSIM

            interpolateSkipProcess = DedupSSIM(
                0.999,
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
            ):
                from src.unifiedInterpolate import RifeCuda

                interpolate_process = RifeCuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolate_method,
                    self.ensemble,
                    self.interpolate_factor,
                    self.fps,
                    interpolateSkipProcess,
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
            ):
                from src.unifiedInterpolate import RifeTensorRT

                interpolate_process = RifeTensorRT(
                    self.interpolate_method,
                    self.interpolate_factor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                    interpolateSkipProcess,
                )

            case "gmfss":
                from src.gmfss.gmfss_fortuna_union import GMFSS

                interpolate_process = GMFSS(
                    int(self.interpolate_factor),
                    self.half,
                    outputWidth,
                    outputHeight,
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
                    interpolateSkipProcess,
                )

    if self.restore:
        match self.restore_method:
            case "scunet" | "dpir" | "nafnet" | "real-plksr" | "anime1080fixer":
                from src.unifiedRestore import UnifiedRestore

                restore_process = UnifiedRestore(
                    self.restore_method,
                    self.half,
                )

            case "anime1080fixer-tensorrt":
                from src.unifiedRestore import UnifiedRestoreTensorRT

                restore_process = UnifiedRestoreTensorRT(
                    self.restore_method,
                    self.half,
                    self.width,
                    self.height,
                    self.forceStatic,
                )

            case "anime1080fixer-directml":
                from src.unifiedRestore import UnifiedRestoreDirectML

                restore_process = UnifiedRestoreDirectML(
                    self.restore_method,
                    self.half,
                    self.width,
                    self.height,
                )

    if self.dedup:
        match self.dedup_method:
            case "ssim":
                from src.dedup.dedup import DedupSSIM

                dedup_process = DedupSSIM(
                    self.dedup_sens,
                    self.sample_size,
                )

            case "mse":
                from src.dedup.dedup import DedupMSE

                dedup_process = DedupMSE(
                    self.dedup_sens,
                    self.sample_size,
                )

            case "ssim-cuda":
                from src.dedup.dedup import DedupSSIMCuda

                dedup_process = DedupSSIMCuda(
                    self.dedup_sens,
                    self.sample_size,
                    self.half,
                )

            case "mse-cuda":
                from src.dedup.dedup import DedupMSECuda

                dedup_process = DedupMSECuda(
                    self.dedup_sens,
                    self.sample_size,
                    self.half,
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
