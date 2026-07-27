"""
Dedup backend factory.

buildDedupProcess(self, method=None, sens=None) -> callable
"""


def buildDedupProcess(self, method=None, sens=None):
    """Build a duplicate detector.

    ``method``/``sens`` default to the --dedup knobs. `--smooth_dedup` drives the same
    backends off its own --smooth_dedup_method/--smooth_dedup_sens, so it passes them in.
    """
    method = self.dedupMethod if method is None else method
    sens = self.dedupSens if sens is None else sens

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

        case "vmaf" | "vmaf-cuda":
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

        case "flownets":
            from src.dedup.dedup import DedupFlownetS

            return DedupFlownetS(
                half=self.half,
                dedupSens=sens,
                height=self.height,
                width=self.width,
            )

        case _:
            raise ValueError(f"No dedup backend is wired up for method '{method}'.")
