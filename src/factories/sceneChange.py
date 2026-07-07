"""
Streaming scene-change detector factory.

buildSceneChangeProcess(self) -> callable(frame) -> bool

Only built when interpolation is enabled and ``--scenechange`` is set. The
returned callable holds its own reference frame and returns True on a hard cut.
Mirrors src/factories/dedup.py.
"""


def buildSceneChangeProcess(self):
    method = self.sceneChangeMethod
    threshold = self.sceneChangeThreshold

    match method:
        case "ssim":
            from src.sceneChange.detector import SceneChangeSSIM

            return SceneChangeSSIM(threshold)

        case "ssim-cuda":
            from src.sceneChange.detector import SceneChangeSSIMCuda

            return SceneChangeSSIMCuda(threshold, self.half)

        case "mse":
            from src.sceneChange.detector import SceneChangeMSE

            return SceneChangeMSE(threshold)

        case "mse-cuda":
            from src.sceneChange.detector import SceneChangeMSECuda

            return SceneChangeMSECuda(threshold, self.half)

        case "maxxvit-tensorrt" | "maxxvit-directml":
            from src.sceneChange.detector import SceneChangeScorer6chDetector

            return SceneChangeScorer6chDetector(method, threshold, self.half, size=224)

        case _:
            raise ValueError(f"Unknown scenechange_method: {method}")
