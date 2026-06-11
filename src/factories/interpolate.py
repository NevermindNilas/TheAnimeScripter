"""
Interpolate backend factory.

buildInterpolateProcess(self, outputWidth, outputHeight) -> callable
  outputWidth/outputHeight: post-upscale dimensions (needed by GMFSS).
"""

import logging


def buildInterpolateProcess(self, outputWidth, outputHeight):
    logging.info(
        f"Interpolating from {format(self.fps, '.3f')}fps to "
        f"{format(self.fps * self.interpolateFactor, '.3f')}fps"
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
            from src.interpolate.rife import RifeCuda

            return RifeCuda(
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
            "rife-mps"
            | "rife4.6-mps"
            | "rife4.15-lite-mps"
            | "rife4.16-lite-mps"
            | "rife4.17-mps"
            | "rife4.18-mps"
            | "rife4.20-mps"
            | "rife4.21-mps"
            | "rife4.22-mps"
            | "rife4.22-lite-mps"
            | "rife4.25-mps"
            | "rife4.25-lite-mps"
            | "rife_elexor-mps"
            | "rife4.25-heavy-mps"
        ):
            from src.interpolate.rife import RifeMPS

            return RifeMPS(
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

        case "rife4.25-depth":
            from src.rifearches.IFNet_distildrba import DepthGuidedRifeCuda

            return DepthGuidedRifeCuda(
                width=self.width,
                height=self.height,
                half=self.half,
                interpolate_method="rife4.25",
                depth_method=self.depthMethod,
                depth_quality=self.depthQuality,
                ensemble=self.ensemble,
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
            from src.interpolate.rife_ncnn import RifeNCNN

            return RifeNCNN(
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
            from src.interpolate.rife_tensorrt import RifeTensorRT

            return RifeTensorRT(
                self.interpolateMethod,
                self.interpolateFactor,
                self.width,
                self.height,
                self.half,
                self.ensemble,
            )

        case "gmfss":
            from src.gmfss.gmfss import GMFSS

            return GMFSS(
                int(self.interpolateFactor),
                self.half,
                outputWidth,
                outputHeight,
                self.ensemble,
                compileMode=self.compileMode,
            )

        case (
            "rife4.6-directml"
            | "rife4.6-openvino"
            | "rife4.15-directml"
            | "rife4.17-directml"
            | "rife4.18-directml"
            | "rife4.20-directml"
            | "rife4.21-directml"
            | "rife4.22-directml"
            | "rife4.22-lite-directml"
            | "rife4.25-directml"
            | "rife4.25-lite-directml"
            | "rife4.25-heavy-directml"
            | "rife4.15-openvino"
            | "rife4.17-openvino"
            | "rife4.18-openvino"
            | "rife4.20-openvino"
            | "rife4.21-openvino"
            | "rife4.22-openvino"
            | "rife4.22-lite-openvino"
            | "rife4.25-openvino"
            | "rife4.25-lite-openvino"
            | "rife4.25-heavy-openvino"
        ):
            from src.interpolate.rife_directml import RifeDirectML

            return RifeDirectML(
                self.interpolateMethod,
                self.interpolateFactor,
                self.width,
                self.height,
                self.half,
                self.ensemble,
            )

        case "distildrba" | "distildrba-lite":
            from src.interpolate.distildrba import DistilDRBACuda, DistilDRBATensorRT

            return DistilDRBACuda(
                self.half,
                self.width,
                self.height,
                self.interpolateMethod,
                interpolateFactor=self.interpolateFactor,
                compileMode=self.compileMode,
            )

        case "distildrba-lite-tensorrt" | "distildrba-tensorrt":
            from src.interpolate.distildrba import DistilDRBATensorRT

            return DistilDRBATensorRT(
                self.half,
                self.width,
                self.height,
                self.interpolateMethod,
                interpolateFactor=self.interpolateFactor,
            )
