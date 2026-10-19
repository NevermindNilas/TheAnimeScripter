"""
Dedup backend factory.

buildDedupProcess(self, method=None, sens=None, width=None, height=None) -> callable
"""


def buildDedupProcess(self, method=None, sens=None, width=None, height=None):
    """Build a duplicate detector.

    ``method``/``sens`` default to the --dedup knobs. `--smooth_dedup` drives the same
    backends off its own --smooth_dedup_method/--smooth_dedup_sens, so it passes them in.
    ``width``/``height`` are the frame size the detector sees (FlowNetS is built for
    it); they default to the source size.
    """
    method = self.dedupMethod if method is None else method
    sens = self.dedupSens if sens is None else sens
    width = self.width if width is None else width
    height = self.height if height is None else height

    match method:
        case "ssim":
            from src.dedup.dedup import DedupSSIM

            return DedupSSIM(
                sens,
            )

        case "mse":
            from src.dedup.dedup import DedupMSE

            return DedupMSE(
                sens,
            )

        case "ssim-cuda":
            from src.dedup.dedup import DedupSSIMCuda

            return DedupSSIMCuda(
                sens,
                self.half,
            )

        case "ssim-rocm":
            from src.dedup.dedup import DedupSSIMROCm

            return DedupSSIMROCm(
                sens,
                self.half,
            )

        case "vmaf" | "vmaf-cuda" | "vmaf-rocm":
            from src.dedup.dedup import DedupVMAF

            return DedupVMAF(
                dedupMethod=method,
                treshold=sens,
                half=self.half,
            )

        case "mse-cuda":
            from src.dedup.dedup import DedupMSECuda

            return DedupMSECuda(
                sens,
                self.half,
            )

        case "mse-rocm":
            from src.dedup.dedup import DedupMSEROCm

            return DedupMSEROCm(
                sens,
                self.half,
            )

        case "flownets":
            from src.dedup.dedup import DedupFlownetS

            return DedupFlownetS(
                half=self.half,
                dedupSens=sens,
                height=height,
                width=width,
            )

        case "flownets-rocm":
            from src.dedup.dedup import DedupFlownetSROCm

            return DedupFlownetSROCm(
                half=self.half,
                dedupSens=sens,
                height=height,
                width=width,
            )

        case _:
            raise ValueError(f"No dedup backend is wired up for method '{method}'.")
