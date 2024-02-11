import logging

def intitialize_models(self):
    new_width = self.width
    new_height = self.height
    upscale_process = None
    interpolate_process = None
    
    if self.upscale:
        new_width *= self.upscale_factor
        new_height *= self.upscale_factor
        logging.info(
            f"Upscaling to {new_width}x{new_height}")
        match self.upscale_method:
            case "shufflecugan" | "cugan":
                from src.cugan.cugan import Cugan
                upscale_process = Cugan(
                    self.upscale_method, self.upscale_factor, self.cugan_kind, self.half, self.width, self.height, self.custom_model, self.nt)
            case "cugan-ncnn":
                from src.cugan.cugan import CuganNCNN
                upscale_process = CuganNCNN(
                    self.nt, self.upscale_factor, self.custom_model)
            case "compact" | "ultracompact" | "superultracompact":
                from src.compact.compact import Compact
                upscale_process = Compact(
                    self.upscale_method, self.upscale_factor, self.half, self.width, self.height, self.custom_model, self.nt)
            case "swinir":
                from src.swinir.swinir import Swinir
                upscale_process = Swinir(
                    self.upscale_factor, self.half, self.width, self.height, self.custom_model, self.nt)
            case "span":
                from src.span.span import SpanSR
                upscale_process = SpanSR(
                    self.upscale_factor, self.half, self.width, self.height, self.custom_model, self.nt)
            case "omnisr":
                from src.omnisr.omnisr import OmniSR
                upscale_process = OmniSR(
                    self.upscale_factor, self.half, self.width, self.height, self.custom_model, self.nt)
            case "shufflecugan_directml":
                from src.cugan.cugan import cuganDirectML
                upscale_process = cuganDirectML(
                    self.upscale_method, self.upscale_factor, self.cugan_kind, self.half, self.width, self.height, self.custom_model)
            case "span-ncnn":
                from src.span.span import spanNCNN
                upscale_process = spanNCNN(
                    self.upscale_factor, self.half, self.width, self.height, self.custom_model)
                
    if self.interpolate:
        UHD = True if new_width >= 3840 or new_height >= 2160 else False
        match self.interpolate_method:
            case "rife" | "rife4.6" | "rife4.13-lite" | "rife4.14-lite" | "rife4.14":
                from src.rife.rife import Rife
                interpolate_process = Rife(
                    int(self.interpolate_factor), self.half, new_width, new_height, UHD, self.interpolate_method, self.ensemble)
            case "rife-ncnn" | "rife4.6-ncnn" | "rife4.13-lite-ncnn" | "rife4.14-lite-ncnn" | "rife4.14-ncnn":
                from src.rifencnn.rifencnn import rifeNCNN
                interpolate_process = rifeNCNN(
                    UHD, self.interpolate_method, self.ensemble)
            case "gmfss":
                from src.gmfss.gmfss_fortuna_union import GMFSS
                interpolate_process = GMFSS(
                    int(self.interpolate_factor), self.half, new_width, new_height, UHD, self.ensemble)
                
    return new_width, new_height, upscale_process, interpolate_process