import logging


def intitialize_models(self):
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

    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolate_factor, '.3f')}fps"
        )

        UHD = True if outputWidth >= 3840 or outputHeight >= 2160 else False
        match self.interpolate_method:
            case "rife" | "rife4.6" | "rife4.14" | "rife4.15" | "rife4.16-lite":
                from src.unifiedInterpolate import RifeCuda

                interpolate_process = RifeCuda(
                    int(self.interpolate_factor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    UHD,
                    self.interpolate_method,
                    self.ensemble,
                    self.nt,
                )
            case "rife-ncnn" | "rife4.6-ncnn" | "rife4.14-ncnn" | "rife4.15-ncnn":
                from src.rifencnn.rifencnn import rifeNCNN

                interpolate_process = rifeNCNN(
                    UHD,
                    self.interpolate_method,
                    self.ensemble,
                    self.nt,
                    outputWidth,
                    outputHeight,
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
                "rife-directml"
                | "rife4.6-directml"
                | "rife4.14-directml"
                | "rife4.15-directml"
                | "rife4.15-lite-directml"
            ):
                from src.unifiedInterpolate import RifeDirectML

                interpolate_process = RifeDirectML(
                    self.interpolate_method,
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
                    self.sample_size,  # Should probably remain 32, values higher result in no real benefits from subjective testing.
                )

            case "mse":
                from src.dedup.dedup import DedupMSE

                dedup_process = DedupMSE(
                    self.dedup_sens,
                    self.sample_size,  # Should probably remain 32, values higher result in no real benefits from subjective testing.
                )

            case "ffmpeg":
                # FFMPEG works on decode so it's not possible to use it with the current processing pipeline
                pass

    return (
        outputWidth,
        outputHeight,
        upscale_process,
        interpolate_process,
        denoise_process,
        dedup_process,
    )
