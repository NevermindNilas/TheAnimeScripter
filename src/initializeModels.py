import logging


def intitialize_models(self):
    new_width = self.width
    new_height = self.height
    upscale_process = None
    interpolate_process = None
    denoise_process = None
    dedup_process = None

    if self.upscale:
        from src.unifiedUpscale import Upscaler

        new_width *= self.upscale_factor
        new_height *= self.upscale_factor
        logging.info(f"Upscaling to {new_width}x{new_height}")
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
                upscale_process = Upscaler(
                    self.upscale_method,
                    self.upscale_factor,
                    self.cugan_kind,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )

            case "cugan-ncnn" | "span-ncnn" | "realesrgan-ncnn" | "shufflecugan-ncnn":
                from .unifiedUpscaleNCNN import UniversalNCNN

                upscale_process = UniversalNCNN(
                    self.upscale_method,
                    self.upscale_factor,
                    self.cugan_kind,
                    self.nt,
                )

    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolate_factor, '.3f')}fps"
        )

        UHD = True if new_width >= 3840 or new_height >= 2160 else False
        match self.interpolate_method:
            case (
                "rife"
                | "rife4.6"
                | "rife4.14"
                | "rife4.15"
                | "rife4.16-lite"
            ):
                from src.rife.rife import Rife

                interpolate_process = Rife(
                    int(self.interpolate_factor),
                    self.half,
                    new_width,
                    new_height,
                    UHD,
                    self.interpolate_method,
                    self.ensemble,
                    self.nt,
                )
            case (
                "rife-ncnn"
                | "rife4.6-ncnn"
                | "rife4.14-ncnn"
                | "rife4.15-ncnn"
            ):
                from src.rifencnn.rifencnn import rifeNCNN

                interpolate_process = rifeNCNN(
                    UHD,
                    self.interpolate_method,
                    self.ensemble,
                    self.nt,
                    new_width,
                    new_height,
                )
            case "gmfss":
                from src.gmfss.gmfss_fortuna_union import GMFSS

                interpolate_process = GMFSS(
                    int(self.interpolate_factor),
                    self.half,
                    new_width,
                    new_height,
                    UHD,
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
        new_width,
        new_height,
        upscale_process,
        interpolate_process,
        denoise_process,
        dedup_process,
    )
