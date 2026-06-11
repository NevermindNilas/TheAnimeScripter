"""
Upscale backend factory.

buildUpscaleProcess(self) -> callable
  Instantiates and returns the correct upscale backend for self.upscaleMethod.
  Returns None when self.upscale is False (callers should guard before calling).
"""

import logging


def buildUpscaleProcess(self):
    from src.upscale.pytorch import UniversalPytorch

    logging.info(
        f"Upscaling to {self.width * self.upscaleFactor}x{self.height * self.upscaleFactor}"
    )

    match self.upscaleMethod:
        case (
            "shufflecugan"
            | "adore"
            | "span"
            | "open-proteus"
            | "aniscale2"
            | "shufflespan"
            | "rtmosr"
            | "saryn"
            | "fallin_soft"
            | "fallin_strong"
            | "gauss"
            | "figsr"
            | "smosr"
        ):
            return UniversalPytorch(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
                self.compileMode,
            )

        case (
            "span-directml"
            | "open-proteus-directml"
            | "aniscale2-directml"
            | "shufflespan-directml"
            | "shufflecugan-directml"
            | "adore-directml"
            | "rtmosr-directml"
            | "saryn-directml"
            | "fallin_soft-directml"
            | "fallin_strong-directml"
            | "span-openvino"
            | "open-proteus-openvino"
            | "aniscale2-openvino"
            | "shufflespan-openvino"
            | "shufflecugan-openvino"
            | "adore-openvino"
            | "rtmosr-openvino"
            | "saryn-openvino"
            | "fallin_soft-openvino"
            | "fallin_strong-openvino"
            | "gauss-openvino"
            | "gauss-directml"
            | "smosr-directml"
            | "smosr-openvino"
        ):
            from src.upscale.directml import UniversalDirectML

            return UniversalDirectML(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
            )

        case (
            "shufflecugan-mps"
            | "adore-mps"
            | "span-mps"
            | "open-proteus-mps"
            | "aniscale2-mps"
            | "shufflespan-mps"
            | "rtmosr-mps"
            | "saryn-mps"
            | "fallin_soft-mps"
            | "fallin_strong-mps"
            | "gauss-mps"
            | "figsr-mps"
            | "smosr-mps"
        ):
            from src.upscale.pytorch import UniversalPytorchMPS

            return UniversalPytorchMPS(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
                self.compileMode,
            )

        case "animesr-openvino" | "animesr-directml":
            from src.upscale.directml import AnimeSRDirectML

            return AnimeSRDirectML(
                self.upscaleMethod,
                self.half,
                self.width,
                self.height,
            )

        case "shufflecugan-ncnn" | "adore-ncnn" | "span-ncnn":
            from src.upscale.ncnn import UniversalNCNN

            return UniversalNCNN(
                self.upscaleMethod,
                self.upscaleFactor,
            )

        case (
            "shufflecugan-tensorrt"
            | "adore-tensorrt"
            | "span-tensorrt"
            | "open-proteus-tensorrt"
            | "aniscale2-tensorrt"
            | "shufflespan-tensorrt"
            | "rtmosr-tensorrt"
            | "saryn-tensorrt"
            | "fallin_soft-tensorrt"
            | "fallin_strong-tensorrt"
            | "gauss-tensorrt"
            | "smosr-tensorrt"
        ):
            from src.upscale.tensorrt import UniversalTensorRT

            return UniversalTensorRT(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
                self.forceStatic,
            )

        case "animesr":
            from src.upscale.misc import AnimeSR

            return AnimeSR(
                2,
                self.half,
                self.width,
                self.height,
                self.compileMode,
            )

        case "animesr-tensorrt":
            from src.upscale.tensorrt import AnimeSRTensorRT

            return AnimeSRTensorRT(
                2,
                self.half,
                self.width,
                self.height,
            )

        case (
            "artcnn_c4f16-tensorrt"
            | "artcnn_c4f16_dn-tensorrt"
            | "artcnn_c4f16_ds-tensorrt"
            | "artcnn_c4f32-tensorrt"
            | "artcnn_c4f32_dn-tensorrt"
            | "artcnn_c4f32_ds-tensorrt"
            | "artcnn_r8f64-tensorrt"
            | "artcnn_r16f96-tensorrt"
        ):
            from src.upscale.artcnn import ArtCNNTensorRT

            return ArtCNNTensorRT(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
                self.forceStatic,
            )

        case (
            "artcnn_c4f16-directml"
            | "artcnn_c4f16_dn-directml"
            | "artcnn_c4f16_ds-directml"
            | "artcnn_c4f32-directml"
            | "artcnn_c4f32_dn-directml"
            | "artcnn_c4f32_ds-directml"
            | "artcnn_r8f64-directml"
            | "artcnn_r16f96-directml"
            | "artcnn_c4f16-openvino"
            | "artcnn_c4f16_dn-openvino"
            | "artcnn_c4f16_ds-openvino"
            | "artcnn_c4f32-openvino"
            | "artcnn_c4f32_dn-openvino"
            | "artcnn_c4f32_ds-openvino"
            | "artcnn_r8f64-openvino"
            | "artcnn_r16f96-openvino"
        ):
            from src.upscale.artcnn import ArtCNNDirectML

            return ArtCNNDirectML(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
            )

        case (
            "maxine-bicubic"
            | "maxine-low"
            | "maxine-medium"
            | "maxine-high"
            | "maxine-ultra"
            | "maxine-highbitrate_low"
            | "maxine-highbitrate_medium"
            | "maxine-highbitrate_high"
            | "maxine-highbitrate_ultra"
        ):
            from src.unifiedUpscale import NvidiaVSR

            return NvidiaVSR(
                self.upscaleMethod,
                self.upscaleFactor,
                self.half,
                self.width,
                self.height,
                self.customModel,
                self.compileMode,
            )
