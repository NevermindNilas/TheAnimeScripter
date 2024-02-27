import logging


def intitialize_models(self):
    new_width = self.width
    new_height = self.height
    upscale_process = None
    interpolate_process = None
    denoise_process = None
    dedup_process = None

    if self.upscale:
        new_width *= self.upscale_factor
        new_height *= self.upscale_factor
        logging.info(f"Upscaling to {new_width}x{new_height}")
        match self.upscale_method:
            case "shufflecugan" | "cugan":
                from src.cugan.cugan import Cugan

                upscale_process = Cugan(
                    self.upscale_method,
                    self.upscale_factor,
                    self.cugan_kind,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )
            case "cugan-ncnn":
                from src.cugan.cugan import CuganNCNN

                upscale_process = CuganNCNN(
                    self.nt, self.upscale_factor,
                )
            case "compact" | "ultracompact" | "superultracompact":
                from src.compact.compact import Compact

                upscale_process = Compact(
                    self.upscale_method,
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )
            case "swinir":
                from src.swinir.swinir import Swinir

                upscale_process = Swinir(
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )
            case "span":
                from src.span.span import SpanSR

                upscale_process = SpanSR(
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )
            case "omnisr":
                from src.omnisr.omnisr import OmniSR

                upscale_process = OmniSR(
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                )

            case "span-ncnn":
                from src.span.span import spanNCNN

                upscale_process = spanNCNN(
                    self.upscale_factor,
                    self.half,
                    self.width,
                    self.height,
                    self.custom_model,
                )
            
            case "realesrgan":
                from src.realesrgan.realesrgan import RealEsrgan
                
                upscale_process = RealEsrgan(
                    self.upscale_factor,
                    self.width,
                    self.height,
                    self.custom_model,
                    self.nt,
                    self.half,
                )
                
            case "realesrgan-ncnn":
                from src.realesrgan.realesrgan import RealEsrganNCNN
                
                upscale_process = RealEsrganNCNN(
                    self.upscale_factor,
                )
                
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolate_factor, '.3f')}fps")
        
        UHD = True if new_width >= 3840 or new_height >= 2160 else False
        match self.interpolate_method:
            case "rife" | "rife4.6" | "rife4.13-lite" | "rife4.14-lite" | "rife4.14":
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
                | "rife4.13-lite-ncnn"
                | "rife4.14-lite-ncnn"
                | "rife4.14-ncnn"
            ):
                from src.rifencnn.rifencnn import rifeNCNN

                interpolate_process = rifeNCNN(
                    UHD, self.interpolate_method, self.ensemble
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
        from src.dpir.dpir import DPIR

        denoise_process = DPIR(
            self.half, new_width, new_height, self.custom_model, self.nt
        )
        
    if self.dedup:
        match self.dedup_method:
            case "ssim":
                from src.dedup.dedup import DedupSSIM

                dedup_process = DedupSSIM(
                    self.dedup_sens, 32 # sample size will remain constant until further testing
                )
                
            case "mse":
                raise NotImplementedError("MSE deduplication is not implemented yet")
                """
                from src.dedup.dedup import DedupMSE

                dedup_process = DedupMSE(
                    self.dedup_sensitivity, self.sample_size
                )
                """
            
            case "psnr":
                from src.dedup.dedup import DedupPSNR

                dedup_process = DedupPSNR(
                    30, 32 # Needs a bit more testing, proof of concept for now.
                )
                
            case "ffmpeg":
                # FFMPEG works on decode so it's not possible to use it with the current processing pipeline
                pass

    return new_width, new_height, upscale_process, interpolate_process, denoise_process, dedup_process
