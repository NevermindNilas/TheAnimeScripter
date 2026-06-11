"""
Restore backend factory.

RestoreChain: chains multiple restore callables into one.
buildRestoreProcess(self) -> callable
  Builds either a single backend or a RestoreChain for multi-method restore.
"""

import logging
import torch


class RestoreChain:
    def __init__(self, restore_processes: list):
        self.restore_processes = restore_processes
        logging.info(f"Initialized restore chain with {len(restore_processes)} models")

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        for restore_process in self.restore_processes:
            frame = restore_process(frame)
        return frame


def buildRestoreProcess(self):
    restoreMethods = (
        self.restoreMethod
        if isinstance(self.restoreMethod, list)
        else [self.restoreMethod]
    )
    restoreProcesses = []

    for method in restoreMethods:
        match method:
            case (
                "scunet"
                | "dpir"
                | "nafnet"
                | "real-plksr"
                | "anime1080fixer"
                | "gater3"
                | "deepdeband-f"
                | "deh264_real"
                | "deh264_span"
                | "hurrdeblur"
                | "dehalo"
                | "scunet-openvino"
                | "anime1080fixer-openvino"
                | "gater3-openvino"
                | "deh264_real-openvino"
                | "deh264_span-openvino"
                | "hurrdeblur-openvino"
                | "dehalo-openvino"
            ):
                from src.unifiedRestore import UnifiedRestoreCuda

                restoreProcesses.append(
                    UnifiedRestoreCuda(
                        method,
                        self.half,
                    )
                )

            case (
                "scunet-mps"
                | "dpir-mps"
                | "nafnet-mps"
                | "real-plksr-mps"
                | "anime1080fixer-mps"
                | "gater3-mps"
                | "deh264_real-mps"
                | "deh264_span-mps"
                | "hurrdeblur-mps"
                | "dehalo-mps"
            ):
                from src.unifiedRestore import UnifiedRestoreMPS

                restoreProcesses.append(
                    UnifiedRestoreMPS(
                        method,
                        self.half,
                    )
                )

            case (
                "anime1080fixer-tensorrt"
                | "gater3-tensorrt"
                | "scunet-tensorrt"
                | "deh264_real-tensorrt"
                | "deh264_span-tensorrt"
                | "hurrdeblur-tensorrt"
                | "dehalo-tensorrt"
            ):
                from src.unifiedRestore import UnifiedRestoreTensorRT

                restoreProcesses.append(
                    UnifiedRestoreTensorRT(
                        method,
                        self.half,
                        self.width,
                        self.height,
                        self.forceStatic,
                    )
                )

            case (
                "anime1080fixer-directml"
                | "anime1080fixer-openvino"
                | "gater3-directml"
                | "scunet-directml"
                | "deh264_real-directml"
                | "deh264_span-directml"
                | "hurrdeblur-directml"
                | "dehalo-directml"
            ):
                from src.unifiedRestore import UnifiedRestoreDirectML

                restoreProcesses.append(
                    UnifiedRestoreDirectML(
                        method,
                        self.half,
                        self.width,
                        self.height,
                    )
                )

            case "fastlinedarken":
                from src.extraArches.fastlinedarken import FastLineDarkenWithStreams

                restoreProcesses.append(
                    FastLineDarkenWithStreams(
                        self.half,
                    )
                )

            case "fastlinedarken-tensorrt":
                from src.extraArches.fastlinedarken import FastLineDarkenTRT

                restoreProcesses.append(
                    FastLineDarkenTRT(
                        self.half,
                        self.height,
                        self.width,
                    )
                )

            case "autocas":
                from src.unifiedRestore import AutoCAS

                restoreProcesses.append(
                    AutoCAS(
                        self.half,
                    )
                )

            case (
                "linethinner-lite"
                | "linethinner-medium"
                | "linethinner-heavy"
                | "linethinner-lite-cuda"
                | "linethinner-medium-cuda"
                | "linethinner-heavy-cuda"
            ):
                from src.extraArches.linethinner import LineThin

                device = "cuda" if "cuda" in method else "cpu"
                variant = method.replace("-cuda", "").replace("linethinner-", "")

                restoreProcesses.append(
                    LineThin(
                        variant=variant,
                        half=self.half,
                        device=device,
                    )
                )

            case (
                "maxine-denoise_low"
                | "maxine-denoise_medium"
                | "maxine-denoise_high"
                | "maxine-denoise_ultra"
                | "maxine-deblur_low"
                | "maxine-deblur_medium"
                | "maxine-deblur_high"
                | "maxine-deblur_ultra"
            ):
                from src.unifiedRestore import MaxineRestore

                restoreProcesses.append(
                    MaxineRestore(
                        method,
                        self.half,
                        self.width,
                        self.height,
                    )
                )

    if len(restoreProcesses) == 1:
        return restoreProcesses[0]
    return RestoreChain(restoreProcesses)
