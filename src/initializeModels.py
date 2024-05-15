import logging


def Segment(self):
    # Lazy loading for startup time reasons

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
            self.nt,
            self.buffer_limit,
            self.benchmark,
        )
    else:
        from src.segment.segmentAnything import SegmentAnything
        SegmentAnything(
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
            self.nt,
            self.buffer_limit,
            self.benchmark,
            self.segment_method,
        )

def initializeModels(self):
    outputWidth = self.width
    outputHeight = self.height
    upscale_process = None
    interpolate_process = None
    denoise_process = None
    dedup_process = None

    if self.upscale:
        from src.unifiedUpscale import UniversalPytorch

        outputWidth *= self.upscale_factor
        outputHeight *= self.upscale_factor
        logging.info(f"Upscaling to {outputWidth}x{outputHeight}")
        match self.upscale_method:
            case (
                "shufflecugan"
                | "cugan"
                | "compact"
                | "ultracompact"
                | "superultracompact"
                | "span"
                | "omnisr"
                | "realesrgan"
                | "apisr"
            ):
                upscale_process = UniversalPytorch(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )

            case (
                "compact-directml"
                | "ultracompact-directml"
                | "superultracompact-directml"
                | "span-directml"
                | "cugan-directml"
                | "shufflecugan-directml"
                | "realesrgan-directml"
            ):
                from .unifiedUpscale import UniversalDirectML

                upscale_process = UniversalDirectML(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )

            case "shufflecugan-ncnn" | "cugan-ncnn" | "span-ncnn" | "realesrgan-ncnn":
                from .unifiedUpscaleNCNN import UniversalNCNN

                upscale_process = UniversalNCNN(
                    self.upscale_method,
                    self.upscale_factor,
                    self.nt,
                )

            case "shufflecugan-tensorrt" | "cugan-tensorrt" | "compact-tensorrt" | "ultracompact-tensorrt" | "superultracompact-tensorrt" | "span-tensorrt" | "realesrgan-tensorrt":
                from .unifiedUpscale import UniversalTensorRT

                upscale_process = UniversalTensorRT(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolate_factor, '.3f')}fps"
        )

        UHD = True if outputWidth >= 3840 or outputHeight >= 2160 else False
        match self.interpolate_method:
            case (
                "rife"
                | "rife4.6"
                | "rife4.15"
                | "rife4.15-lite"
                | "rife4.16-lite"
            ):
                from src.unifiedInterpolate import RifeCuda

                interpolate_process = RifeCuda(
                    self.half,
                    outputWidth,
                    outputHeight,
                    UHD,
                    self.interpolate_method,
                    self.ensemble,
                    self.nt,
                    self.interpolate_factor,
                )
            case "gmfss":
                from src.gmfss.gmfss_fortuna_union import GMFSS

                interpolate_process = GMFSS(
                    int(self.interpolate_factor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    UHD,
                    self.ensemble,
                    self.nt,
                )

            case (
                "rife-ncnn"
                | "rife4.6-ncnn"
                | "rife4.15-ncnn"
                | "rife4.15-lite-ncnn"
                | "rife4.16-lite-ncnn"
            ):
                from src.rifencnn.rifencnn import rifeNCNN

                interpolate_process = rifeNCNN(
                    self.interpolate_method,
                    self.ensemble,
                    self.nt,
                    outputWidth,
                    outputHeight,
                )

            case (
                "rife-tensorrt"
                | "rife4.6-tensorrt"
                | "rife4.15-tensorrt"
                | "rife4.15-lite-tensorrt"
            ):
                
                from src.unifiedInterpolate import RifeTensorRT

                interpolate_process = RifeTensorRT(
                    self.interpolate_method,
                    self.interpolate_factor,
                    outputWidth,
                    outputHeight,
                    self.half,
                    self.ensemble,
                    self.nt,
                )
            

    if self.denoise:
        match self.denoise_method:
            case "scunet" | "dpir" | "nafnet" | "span":
                from src.unifiedDenoise import UnifiedDenoise

                denoise_process = UnifiedDenoise(
                    self.denoise_method,
                    self.width,
                    self.height,
                    self.half,
                    self.custom_model,
                    self.nt,
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

            # case ffmpeg, ffmpeg works on decode, refer to ffmpegSettings.py ReadBuffer class.

    return (
        outputWidth,
        outputHeight,
        upscale_process,
        interpolate_process,
        denoise_process,
        dedup_process,
    )
