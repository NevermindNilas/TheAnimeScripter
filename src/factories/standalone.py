"""
Standalone processing drivers that bypass the main frame loop.
Each function receives the VideoProcessor instance and runs its own
read/process/write pipeline to completion.
"""


def objectDetection(self):
    if "directml" in self.objDetectMethod or "openvino" in self.objDetectMethod:
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
            self.objDetectDisableAnnotations,
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
            self.objDetectDisableAnnotations,
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
            self.totalFrames,
            self.half,
        )


def autoClip(self):
    if self.autoclipMethod == "pyscenedetect":
        from src.autoclip.autoclip import AutoClip

        AutoClip(
            self.input,
            self.autoclipSens,
            self.inpoint,
            self.outpoint,
        )
    elif self.autoclipMethod in ("maxxvit-directml", "maxxvit-tensorrt"):
        from src.autoclip.autoclipMaxxvit import AutoClipMaxxvit

        AutoClipMaxxvit(
            self.input,
            self.autoclipMethod,
            self.autoclipSens,
            self.inpoint,
            self.outpoint,
            self.half,
        )
    elif self.autoclipMethod == "transnetv2":
        from src.autoclip.autoclipTransnetv2 import AutoClipTransnetv2

        AutoClipTransnetv2(
            self.input,
            self.autoclipSens,
            self.inpoint,
            self.outpoint,
            self.half,
        )
    else:
        raise ValueError(f"Unknown autoclip_method: {self.autoclipMethod}")


def segment(self):
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
    elif self.segmentMethod == "anime-openvino":
        from src.segment.animeSegment import AnimeSegmentOpenVino

        AnimeSegmentOpenVino(
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
    match self.depthMethod:
        case (
            "small_v2"
            | "base_v2"
            | "large_v2"
            | "giant_v2"
            | "distill_small_v2"
            | "distill_base_v2"
            | "distill_large_v2"
        ):
            from src.depth.backends.cuda import DepthCuda

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
                depthNorm=self.depthNorm,
            )

        case (
            "small_v2-tensorrt"
            | "base_v2-tensorrt"
            | "large_v2-tensorrt"
            | "distill_small_v2-tensorrt"
            | "distill_base_v2-tensorrt"
            | "distill_large_v2-tensorrt"
        ):
            from src.depth.backends.tensorrt import DepthTensorRTV2

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
                depthNorm=self.depthNorm,
            )

        case (
            "small_v2-directml"
            | "base_v2-directml"
            | "large_v2-directml"
            | "distill_small_v2-directml"
            | "distill_base_v2-directml"
            | "distill_large_v2-directml"
            | "small_v2-openvino"
            | "base_v2-openvino"
            | "large_v2-openvino"
            | "distill_small_v2-openvino"
            | "distill_base_v2-openvino"
            | "distill_large_v2-openvino"
        ):
            from src.depth.backends.directml import DepthDirectMLV2

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
                depthNorm=self.depthNorm,
            )

        case (
            "og_small_v2"
            | "og_base_v2"
            | "og_large_v2"
            | "og_giant_v2"
            | "og_distill_small_v2"
            | "og_distill_base_v2"
            | "og_distill_large_v2"
        ):
            from src.depth.backends.cuda import OGDepthV2CUDA

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
                depthNorm=self.depthNorm,
            )

        case "og_video_small_v2" | "og_video_base_v2" | "og_video_large_v2":
            from src.depth.backends.video import VideoDepthAnythingCUDA

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

        case "video_small_v2" | "video_large_v2":
            from src.depth.backends.video import VideoDepthAnythingTorch

            VideoDepthAnythingTorch(
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
                depth_window=self.depthWindow,
            )

        case (
            "og_small_v2-tensorrt"
            | "og_base_v2-tensorrt"
            | "og_large_v2-tensorrt"
            | "og_distill_small_v2-tensorrt"
            | "og_distill_base_v2-tensorrt"
            | "og_distill_large_v2-tensorrt"
        ):
            from src.depth.backends.tensorrt import OGDepthV2TensorRT

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
                depthNorm=self.depthNorm,
            )

        case (
            "og_small_v2-directml"
            | "og_base_v2-directml"
            | "og_large_v2-directml"
            | "og_small_v2-openvino"
            | "og_base_v2-openvino"
            | "og_large_v2-openvino"
        ):
            from src.depth.backends.directml import OGDepthV2DirectML

            OGDepthV2DirectML(
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
                depthNorm=self.depthNorm,
            )

        case "small_v3" | "base_v3" | "large_v3" | "og_large_v3":
            from src.depth.backends.cuda import OGDepthV3Cuda

            OGDepthV3Cuda(
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
                depthNorm=self.depthNorm,
            )

        case (
            "small_v3-directml"
            | "base_v3-directml"
            | "small_v3-openvino"
            | "base_v3-openvino"
        ):
            from src.depth.backends.directml import DepthDirectMLV3

            DepthDirectMLV3(
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

        case "small_v3-tensorrt" | "base_v3-tensorrt":
            from src.depth.backends.tensorrt import DepthTensorRTV3

            DepthTensorRTV3(
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

        case _:
            raise ValueError(
                f"Unsupported depth_method: {self.depthMethod}. "
                f"This value is accepted by the CLI but has no model wired up "
                f"in initializeModels.depth()."
            )


def motionBlur(self):
    from src.motionBlur import MotionBlurPipeline

    MotionBlurPipeline(
        self.input,
        self.output,
        self.width,
        self.height,
        self.fps,
        half=self.half,
        inpoint=self.inpoint,
        outpoint=self.outpoint,
        encode_method=self.encodeMethod,
        custom_encoder=self.customEncoder,
        benchmark=self.benchmark,
        totalFrames=self.totalFrames,
        bitDepth=self.bitDepth,
        interpolate_method=self.moblurMethod,
        interpolate_factor=self.moblurFactor,
        moblur_strength=self.moblurStrength,
        moblur_shutter_angle=self.moblurShutterAngle,
        moblur_gamma=self.moblurLinearBlend,
        moblur_mask=self.moblurMask,
        ensemble=self.ensemble,
        dynamic_scale=self.dynamicScale,
        static_step=self.staticStep,
        compile_mode=self.compileMode,
        decode_method=self.decodeMethod,
    )


def stabilize(self):
    from src.stabilize.stabilize import VideoStabilize

    VideoStabilize(
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
        self.totalFrames,
        self.bitDepth,
    )
