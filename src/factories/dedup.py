"""
Dedup backend factory.

buildDedupProcess(self) -> callable
"""


def buildDedupProcess(self):
    match self.dedupMethod:
        case "ssim":
            from src.dedup.dedup import DedupSSIM

            return DedupSSIM(
                self.dedupSens,
            )

        case "mse":
            from src.dedup.dedup import DedupMSE

            return DedupMSE(
                self.dedupSens,
            )

        case "ssim-cuda":
            from src.dedup.dedup import DedupSSIMCuda

            return DedupSSIMCuda(
                self.dedupSens,
                self.half,
            )

        case "vmaf" | "vmaf-cuda":
            from src.dedup.dedup import DedupVMAF

            return DedupVMAF(
                dedupMethod=self.dedupMethod,
                treshold=self.dedupSens,
                half=self.half,
            )

        case "mse-cuda":
            from src.dedup.dedup import DedupMSECuda

            return DedupMSECuda(
                self.dedupSens,
                self.half,
            )

        case "flownets":
            from src.dedup.dedup import DedupFlownetS

            return DedupFlownetS(
                half=self.half,
                dedupSens=self.dedupSens,
                height=self.height,
                width=self.width,
            )
