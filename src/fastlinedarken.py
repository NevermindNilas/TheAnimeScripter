import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.downloadModels import weightsDir
from src.utils.isCudaInit import CudaChecker

checker = CudaChecker()


# TRT throws a fit with internal Streams so I duplicated the class
class FastLineDarkenWithStreams(nn.Module):
    def __init__(
        self,
        half: bool = False,
        thinEdges: bool = True,
        gaussian: bool = True,
        darkenStrength: float = 0.7,
    ):
        super(FastLineDarkenWithStreams, self).__init__()
        self.half = half
        self.thinEdges = thinEdges
        self.gaussian = gaussian
        self.darkenStrength = darkenStrength

        self.weights = torch.tensor([0.2989, 0.5870, 0.1140])
        self.sobelX = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobelY = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.weightsLocal = self.weights.to(checker.device)
        self.sobelXLocal = self.sobelX.to(checker.device)
        self.sobelYLocal = self.sobelY.to(checker.device)

        if self.half and checker.cudaAvailable:
            self.weightsLocal = self.weightsLocal.half()
            self.sobelXLocal = self.sobelXLocal.half()
            self.sobelYLocal = self.sobelYLocal.half()

        if checker.cudaAvailable:
            self.stream = torch.cuda.Stream()

    def gaussianBlur(self, img, kernelSize=3, sigma=1.0):
        channels, _, _ = img.shape
        x = torch.arange(-kernelSize // 2 + 1.0, kernelSize // 2 + 1.0)
        x = torch.exp(-(x**2) / (2 * sigma**2))
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.to(img.device, dtype=img.dtype)
        kernel2d = kernel2d.expand(channels, 1, kernelSize, kernelSize)
        img = img.unsqueeze(0)
        blurredImg = F.conv2d(img, kernel2d, padding=kernelSize // 2, groups=channels)
        return blurredImg.squeeze(0)

    def applyFilter(self, img, kernel):
        img = img.unsqueeze(0).unsqueeze(0)
        filteredImg = F.conv2d(img, kernel, padding=1)
        return filteredImg.squeeze(0).squeeze(0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        if checker.cudaAvailable:
            return self._cudaWorkflow(image)
        else:
            return self._cpuWorkflow(image)

    def _cudaWorkflow(self, image):
        with torch.cuda.stream(self.stream):
            image = image.half() if self.half else image.float()
            image = image.squeeze(0)
            grayscale = torch.tensordot(image, self.weightsLocal, dims=([0], [0]))
            edgesX = self.applyFilter(grayscale, self.sobelXLocal)
            edgesY = self.applyFilter(grayscale, self.sobelYLocal)
            edges = torch.sqrt(edgesX**2 + edgesY**2)
            edges = (edges - edges.min()) / (edges.max() - edges.min())
            if self.thinEdges:
                edges = edges.unsqueeze(0).unsqueeze(0)
                thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
                thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
            else:
                thinnedEdges = edges
            if self.gaussian:
                softenedEdges = self.gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
            else:
                softenedEdges = thinnedEdges
            enhancedImage = image.sub_(self.darkenStrength * softenedEdges)
            enhancedImage = enhancedImage.clamp(0, 1)
            enhancedImage = enhancedImage.unsqueeze(0)
        self.stream.synchronize()
        return enhancedImage

    def _cpuWorkflow(self, image):
        image = image.half() if self.half else image.float()
        image = image.permute(2, 0, 1)

        grayscale = torch.tensordot(image, self.weightsLocal, dims=([0], [0]))

        edgesX = self.applyFilter(grayscale, self.sobelXLocal)
        edgesY = self.applyFilter(grayscale, self.sobelYLocal)
        edges = torch.sqrt(edgesX**2 + edgesY**2)

        edges = (edges - edges.min()) / (edges.max() - edges.min())

        if self.thinEdges:
            edges = edges.unsqueeze(0).unsqueeze(0)
            thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
            thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
        else:
            thinnedEdges = edges

        if self.gaussian:
            softenedEdges = self.gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
        else:
            softenedEdges = thinnedEdges

        enhancedImage = image.sub_(self.darkenStrength * softenedEdges)
        return enhancedImage.clamp(0, 1).unsqueeze(0)


# This is the original code from the FastLineDarken class
class FastLineDarken(nn.Module):
    def __init__(
        self,
        half: bool = False,
        thinEdges: bool = True,
        gaussian: bool = True,
        darkenStrength: float = 0.8,
    ):
        super(FastLineDarken, self).__init__()
        self.half = half
        self.thinEdges = thinEdges
        self.gaussian = gaussian
        self.darkenStrength = darkenStrength

        self.weights = torch.tensor([0.2989, 0.5870, 0.1140])
        self.sobelX = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobelY = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.weightsLocal = self.weights.to(checker.device)
        self.sobelXLocal = self.sobelX.to(checker.device)
        self.sobelYLocal = self.sobelY.to(checker.device)

        if self.half and checker.cudaAvailable:
            self.weightsLocal = self.weightsLocal.half()
            self.sobelXLocal = self.sobelXLocal.half()
            self.sobelYLocal = self.sobelYLocal.half()

    def gaussianBlur(self, img, kernelSize=3, sigma=1.0):
        channels, _, _ = img.shape
        x = torch.arange(-kernelSize // 2 + 1.0, kernelSize // 2 + 1.0)
        x = torch.exp(-(x**2) / (2 * sigma**2))
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.to(img.device, dtype=img.dtype)
        kernel2d = kernel2d.expand(channels, 1, kernelSize, kernelSize)
        img = img.unsqueeze(0)
        blurredImg = F.conv2d(img, kernel2d, padding=kernelSize // 2, groups=channels)
        return blurredImg.squeeze(0)

    def applyFilter(self, img, kernel):
        img = img.unsqueeze(0).unsqueeze(0)
        filteredImg = F.conv2d(img, kernel, padding=1)
        return filteredImg.squeeze(0).squeeze(0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input image must be a torch.Tensor")

        if checker.cudaAvailable:
            return self._cudaWorkflow(image)
        else:
            return self._cpuWorkflow(image)

    def _cudaWorkflow(self, image):
        image = image.half() if self.half else image.float()
        image = image.squeeze(0)
        grayscale = torch.tensordot(image, self.weightsLocal, dims=([0], [0]))
        edgesX = self.applyFilter(grayscale, self.sobelXLocal)
        edgesY = self.applyFilter(grayscale, self.sobelYLocal)
        edges = torch.sqrt(edgesX**2 + edgesY**2)
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        if self.thinEdges:
            edges = edges.unsqueeze(0).unsqueeze(0)
            thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
            thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
        else:
            thinnedEdges = edges
        if self.gaussian:
            softenedEdges = self.gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
        else:
            softenedEdges = thinnedEdges
        enhancedImage = image.sub_(self.darkenStrength * softenedEdges)
        enhancedImage = enhancedImage.clamp(0, 1)
        enhancedImage = enhancedImage.unsqueeze(0)
        return enhancedImage

    def _cpuWorkflow(self, image):
        image = image.half() if self.half else image.float()
        image = image.permute(2, 0, 1)

        grayscale = torch.tensordot(image, self.weightsLocal, dims=([0], [0]))

        edgesX = self.applyFilter(grayscale, self.sobelXLocal)
        edgesY = self.applyFilter(grayscale, self.sobelYLocal)
        edges = torch.sqrt(edgesX**2 + edgesY**2)

        edges = (edges - edges.min()) / (edges.max() - edges.min())

        if self.thinEdges:
            edges = edges.unsqueeze(0).unsqueeze(0)
            thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
            thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
        else:
            thinnedEdges = edges

        if self.gaussian:
            softenedEdges = self.gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
        else:
            softenedEdges = thinnedEdges

        enhancedImage = image.sub_(self.darkenStrength * softenedEdges)
        return enhancedImage.clamp(0, 1).unsqueeze(0)


class FastLineDarkenTRT(FastLineDarken):
    def __init__(
        self,
        half: bool = False,
        height: int = 224,
        width: int = 224,
        forceStatic: bool = False,
    ):
        super().__init__(half)
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.height = height
        self.width = width

        self.dtype = torch.float16 if self.half else torch.float32
        self.device = torch.device("cuda" if checker.cudaAvailable else "cpu")

        self.model = FastLineDarken(half=self.half)
        self.model.eval()

        folderPath = os.path.join(weightsDir, "fastlinedarken")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        self.modelPath = os.path.join(
            folderPath, f"fastlinedarken{width}x{height}.onnx"
        )

        if not os.path.exists(self.modelPath):
            torch.onnx.export(
                self.model,
                torch.randn(1, 3, height, width, device=self.device, dtype=self.dtype),
                self.modelPath,
                input_names=["input"],
                output_names=["output"],
                opset_version=20,
                dynamo=False,
            )

        enginePath = tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, height, width],
        )

        self.engine, self.context = tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, self.height, self.width],
                inputsOpt=[1, 3, self.height, self.width],
                inputsMax=[1, 3, self.height, self.width],
                forceStatic=forceStatic,
            )

        try:
            os.remove(self.modelPath)
        except FileNotFoundError:
            pass

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensorName = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensorName) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensorName, self.dummyInput.shape)

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()
        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(frame, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def processOutput(self):
        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.clone().detach()
        self.outputStream.synchronize()
        return output

    @torch.inference_mode()
    def __call__(self, frame):
        self.processFrame(frame)

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()
        return self.processOutput()
