import torch
import time
import unittest


def dynamicScale(
    img1: torch.Tensor,
    img2: torch.Tensor,
    minScale: float = 0.25,
    maxScale: float = 2.0,
) -> float:
    """
    This function calculates the scale factor between two images
    The scale factor is calculated as the difference between the two images
    The scale factor is then rounded to the nearest 0.25 and clamped between minScale and maxScale
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, got {img1.shape} and {img2.shape}"
        )

    if img1.device != img2.device:
        raise ValueError(
            f"Both images must be on the same device, got {img1.device} and {img2.device}"
        )

    diff = torch.abs(img1 - img2).mean().item()

    scale = maxScale - diff
    scale = max(minScale, min(maxScale, scale))
    scale = round(scale / 0.25) * 0.25
    return scale


def dynamicScaleList(
    img1: torch.Tensor,
    img2: torch.Tensor,
    minScale: float = 0.25,
    maxScale: float = 2.0,
    scaleList: list = [],
) -> list:
    """
    This function is a wrapper around dynamicScale that applies the scale to a list of scales
    Should simplify the amount of extra repeated code in the main function
    """
    scale = dynamicScale(img1, img2, minScale, maxScale)
    for i in range(len(scaleList)):
        scaleList[i] = scaleList[i] / scale

    return scaleList


def fps(img1: torch.Tensor, img2: torch.Tensor, iterations: int = 10000) -> float:
    startTime = time.time()
    for _ in range(iterations):
        dynamicScale(img1, img2)
    endTime = time.time()
    totalTime = endTime - startTime
    fps = iterations / totalTime
    print(f"FPS: {fps:.2f}")
    return fps


class TestDynamicScale(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.img1 = torch.rand(1, 3, 1080, 1920, device=self.device)
        self.img2 = torch.rand(1, 3, 1080, 1920, device=self.device)

    def testIdenticalTensors(self):
        scale = dynamicScale(self.img1, self.img1)
        self.assertEqual(scale, 2.0)

    def testScaleBoundaryLower(self):
        img2 = self.img1 - 1.75  # Create a large difference
        scale = dynamicScale(self.img1, img2)
        self.assertEqual(scale, 0.25)

    def testScaleBoundaryUpper(self):
        img2 = self.img1 - 0.0  # Zero difference
        scale = dynamicScale(self.img1, img2)
        self.assertEqual(scale, 2.0)

    def testDifferentShapes(self):
        img2 = torch.rand(1, 3, 720, 1280, device=self.device)
        with self.assertRaises(ValueError):
            dynamicScale(self.img1, img2)

    def testDifferentDevices(self):
        if torch.cuda.is_available():
            imgCpu = self.img1.cpu()
            with self.assertRaises(ValueError):
                dynamicScale(self.img1, imgCpu)
        else:
            self.skipTest("CUDA not available")

    def testFpsReturnsFloat(self):
        result = fps(self.img1, self.img2, iterations=100)
        self.assertIsInstance(result, float)

    def testFpsPerformance(self):
        try:
            fps(self.img1, self.img2, iterations=10)
        except Exception as e:
            self.fail(f"fps raised an exception {e}")

    # Tests for dynamicScaleList
    def testDynamicScaleListEmpty(self):
        scale_list = []
        result = dynamicScaleList(self.img1, self.img2, scaleList=scale_list)
        self.assertEqual(result, [])
        self.assertEqual(scale_list, [])

    def testDynamicScaleListNormal(self):
        scale_list = [1.0, 2.0, 3.0]
        expected_scale = dynamicScale(self.img1, self.img2)
        expected_result = [s / expected_scale for s in scale_list]
        result = dynamicScaleList(self.img1, self.img2, scaleList=scale_list.copy())
        self.assertEqual(result, expected_result)

    def testDynamicScaleListDifferentShapes(self):
        scale_list = [1.0, 2.0, 3.0]
        img2 = torch.rand(1, 3, 720, 1280, device=self.device)
        with self.assertRaises(ValueError):
            dynamicScaleList(self.img1, img2, scaleList=scale_list)

    def testDynamicScaleListDifferentDevices(self):
        scale_list = [1.0, 2.0, 3.0]
        if torch.cuda.is_available():
            imgCpu = self.img1.cpu()
            with self.assertRaises(ValueError):
                dynamicScaleList(self.img1, imgCpu, scaleList=scale_list)
        else:
            self.skipTest("CUDA not available")

    def testDynamicScaleListImmutableDefault(self):
        # Ensure that the default scaleList is not modified
        result = dynamicScaleList(self.img1, self.img2)
        self.assertEqual(result, [])

    def testDynamicScaleListScaleRounding(self):
        # Test if the scale is correctly rounded to the nearest 0.25
        img2 = self.img1 - 0.5  # Adjust difference to achieve scale = 1.5
        scale = dynamicScale(self.img1, img2)
        expected_scale = 1.5
        scale_list = [0.5, 1.5, 2.5]
        expected_result = [s / expected_scale for s in scale_list]
        result = dynamicScaleList(self.img1, img2, scaleList=scale_list.copy())
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
