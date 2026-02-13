import os
import cv2
import numpy as np
import torch
import logging
from concurrent.futures import ThreadPoolExecutor

from src.utils.logAndPrint import logAndPrint
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.isCudaInit import CudaChecker
from .yolov9_mit import draw_detections, draw_masks, draw_box, colors
from src.constants import ADOBE

if ADOBE:
    from src.utils.aeComms import progressState

checker = CudaChecker()

__all__ = [
    "ObjectDetection",
    "ObjectDetectionDML",
    "ObjectDetectionTensorRT",
    "draw_detections",
]


class ObjectDetectionDML:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encodeMethod="x264",
        customEncoder="",
        benchmark=False,
        half=False,
        objDetectMethod="yolov9_small-directml",
        totalFrames=0,
        disableAnnotations=False,
    ):
        import onnxruntime as ort

        self.ort = ort
        self.ort.set_default_logger_severity(3)

        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.benchmark = benchmark
        self.half = half
        self.objDetectMethod = objDetectMethod
        self.totalFrames = totalFrames
        self.disableAnnotations = disableAnnotations
        self.confThreshold = 0.25

        if "openvino" in objDetectMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encodeMethod,
                self.customEncoder,
                self.width,
                self.height,
                self.fps,
                sharpen=False,
                sharpen_sens=None,
                grayscale=False,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading object detection model: {self.objDetectMethod}..."}
            )

        folderName = self.objDetectMethod.replace("-directml", "-onnx").replace(
            "-openvino", "-onnx"
        )
        filename = modelsMap(
            model=self.objDetectMethod, modelType="onnx", half=self.half
        )

        folderPath = os.path.join(weightsDir, folderName)
        os.makedirs(folderPath, exist_ok=True)
        self.modelPath = os.path.join(folderPath, filename)

        if not os.path.exists(self.modelPath):
            self.modelPath = downloadModels(
                model=self.objDetectMethod,
                modelType="onnx",
                half=self.half,
            )

        providers = self.ort.get_available_providers()

        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.objDetectMethod:
                logging.info(
                    "DirectML provider available. Using DirectML for object detection"
                )
                self.session = self.ort.InferenceSession(
                    self.modelPath, providers=["DmlExecutionProvider"]
                )
            elif "openvino" in self.objDetectMethod:
                logging.info("Using OpenVINO for object detection")
                self.session = self.ort.InferenceSession(
                    self.modelPath, providers=["OpenVINOExecutionProvider"]
                )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU for object detection"
            )
            self.session = self.ort.InferenceSession(
                self.modelPath, providers=["CPUExecutionProvider"]
            )

        modelInputs = self.session.get_inputs()
        modelOutputs = self.session.get_outputs()

        self.inputName = modelInputs[0].name
        self.outputNames = [o.name for o in modelOutputs]

        inputShape = modelInputs[0].shape
        self.inputHeight = inputShape[2] if isinstance(inputShape[2], int) else 640
        self.inputWidth = inputShape[3] if isinstance(inputShape[3], int) else 640

    def prepareInput(self, image):
        self.imgHeight, self.imgWidth = image.shape[:2]
        inputImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputImg = cv2.resize(inputImg, (self.inputWidth, self.inputHeight))
        inputImg = inputImg / 255.0
        inputImg = inputImg.transpose(2, 0, 1)
        inputTensor = inputImg[np.newaxis, :, :, :].astype(np.float32)
        return inputTensor

    def processOutput(self, output):
        boxes = output[:, :-2]
        confidences = output[:, -2]
        classIds = output[:, -1].astype(int)

        mask = confidences > self.confThreshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        classIds = classIds[mask]

        boxes = self.rescaleBoxes(boxes)
        return classIds, boxes, confidences

    def rescaleBoxes(self, boxes):
        inputShape = np.array(
            [self.inputWidth, self.inputHeight, self.inputWidth, self.inputHeight]
        )
        boxes = np.divide(boxes, inputShape, dtype=np.float32)
        boxes *= np.array(
            [self.imgWidth, self.imgHeight, self.imgWidth, self.imgHeight]
        )
        return boxes

    def detect(self, image):
        inputTensor = self.prepareInput(image)
        outputs = self.session.run(self.outputNames, {self.inputName: inputTensor})
        return self.processOutput(outputs[0])

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frameNp = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frameNp = (frameNp * 255).astype(np.uint8)
            frameNp = cv2.cvtColor(frameNp, cv2.COLOR_RGB2BGR)

            classIds, boxes, confidences = self.detect(frameNp)

            if self.disableAnnotations:
                outputImg = frameNp.copy()
                outputImg = draw_masks(outputImg, boxes, classIds, mask_alpha=0.3)
                for classId, box in zip(classIds, boxes):
                    color = colors[classId]
                    draw_box(outputImg, box, color)
            else:
                outputImg = draw_detections(frameNp, boxes, confidences, classIds)

            outputImg = cv2.cvtColor(outputImg, cv2.COLOR_BGR2RGB)
            outputTensor = (
                torch.from_numpy(outputImg).permute(2, 0, 1).unsqueeze(0).float()
                / 255.0
            )

            self.writeBuffer.write(outputTensor)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()


class ObjectDetectionTensorRT:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encodeMethod="x264",
        customEncoder="",
        benchmark=False,
        half=False,
        objDetectMethod="yolov9_small-tensorrt",
        totalFrames=0,
        disableAnnotations=False,
    ):
        import tensorrt as trt
        from src.utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.benchmark = benchmark
        self.half = half
        self.objDetectMethod = objDetectMethod
        self.totalFrames = totalFrames
        self.disableAnnotations = disableAnnotations
        self.confThreshold = 0.25

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encodeMethod,
                self.customEncoder,
                self.width,
                self.height,
                self.fps,
                sharpen=False,
                sharpen_sens=None,
                grayscale=False,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading TensorRT object detection model: {self.objDetectMethod}..."}
            )

        folderName = self.objDetectMethod.replace("-tensorrt", "-onnx")
        filename = modelsMap(
            model=self.objDetectMethod, modelType="onnx", half=self.half
        )

        folderPath = os.path.join(weightsDir, folderName)
        os.makedirs(folderPath, exist_ok=True)
        self.modelPath = os.path.join(folderPath, filename)

        if not os.path.exists(self.modelPath):
            self.modelPath = downloadModels(
                model=self.objDetectMethod,
                modelType="onnx",
                half=self.half,
            )

        logging.info("Using TensorRT for object detection")

        import onnx
        onnxModel = onnx.load(self.modelPath)
        inputName = onnxModel.graph.input[0].name
        outputName = onnxModel.graph.output[0].name
        logging.info(f"Model input name: {inputName}, output name: {outputName}")

        inputDims = list(onnxModel.graph.input[0].type.tensor_type.shape.dim)
        self.inputHeight = inputDims[2].dim_value if inputDims[2].dim_value > 0 else 640
        self.inputWidth = inputDims[3].dim_value if inputDims[3].dim_value > 0 else 640
        logging.info(f"Model input shape: 1x3x{self.inputHeight}x{self.inputWidth}")

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.inputHeight, self.inputWidth],
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if self.engine is None or self.context is None or not os.path.exists(enginePath):
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, self.inputHeight, self.inputWidth],
                inputsOpt=[1, 3, self.inputHeight, self.inputWidth],
                inputsMax=[1, 3, self.inputHeight, self.inputWidth],
                inputName=[inputName],
            )

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()

        self.dtype = torch.float16 if self.half else torch.float32
        self.dummyInput = torch.zeros(
            (1, 3, self.inputHeight, self.inputWidth),
            device=checker.device,
            dtype=self.dtype,
        )

        self.maxDetections = 1000
        self.dummyOutput = torch.zeros(
            (self.maxDetections, 6),
            device=checker.device,
            dtype=self.dtype,
        )

        self.inputName = inputName
        self.outputName = outputName

        for i in range(self.engine.num_io_tensors):
            tensorName = self.engine.get_tensor_name(i)
            logging.info(f"Tensor {i}: {tensorName}, mode: {self.engine.get_tensor_mode(tensorName)}")
            if self.engine.get_tensor_mode(tensorName) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensorName, self.dummyInput.shape)
                self.context.set_tensor_address(tensorName, self.dummyInput.data_ptr())
            else:
                self.context.set_tensor_address(tensorName, self.dummyOutput.data_ptr())

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.colorsTensor = torch.tensor(colors, device=checker.device, dtype=torch.float32)

    @torch.inference_mode()
    def prepareInput(self, frame):
        self.imgHeight, self.imgWidth = frame.shape[2], frame.shape[3]
        inputTensor = torch.nn.functional.interpolate(
            frame, size=(self.inputHeight, self.inputWidth), mode="bilinear", align_corners=False
        )
        return inputTensor

    @torch.inference_mode()
    def processOutput(self, output):
        if output.dim() == 3:
            output = output.squeeze(0)
        boxes = output[:, :-2]
        confidences = output[:, -2]
        classIds = output[:, -1].long()

        mask = confidences > self.confThreshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        classIds = classIds[mask]

        boxes = self.rescaleBoxes(boxes)
        
        logging.debug(f"Detected {boxes.shape[0]} objects")
        if boxes.shape[0] > 0:
            logging.debug(f"First box: {boxes[0]}, confidence: {confidences[0]}, class: {classIds[0]}")
        
        return classIds, boxes, confidences

    @torch.inference_mode()
    def rescaleBoxes(self, boxes):
        scaleFactorX = self.imgWidth / self.inputWidth
        scaleFactorY = self.imgHeight / self.inputHeight
        boxes[:, 0] *= scaleFactorX
        boxes[:, 1] *= scaleFactorY
        boxes[:, 2] *= scaleFactorX
        boxes[:, 3] *= scaleFactorY
        return boxes

    @torch.inference_mode()
    def detect(self, frame):
        inputTensor = self.prepareInput(frame)
        
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(inputTensor, non_blocking=True)
            self.normStream.synchronize()

        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

        outputShape = self.context.get_tensor_shape(self.outputName)
        logging.info(f"TRT Output shape: {outputShape}")
        logging.info(f"Dummy output shape: {self.dummyOutput.shape}")
        logging.info(f"Dummy output first 3 rows: {self.dummyOutput[:3]}")
        logging.info(f"Dummy output max: {self.dummyOutput.max()}, min: {self.dummyOutput.min()}")

        numDetections = outputShape[0] if len(outputShape) > 0 and outputShape[0] > 0 else 0
        logging.info(f"Num detections: {numDetections}")

        if numDetections > 0:
            output = self.dummyOutput[:numDetections]
            logging.info(f"First detection: {output[0]}")
            return self.processOutput(output)
        
        return torch.tensor([], device=checker.device, dtype=torch.long), torch.tensor([], device=checker.device, dtype=self.dtype), torch.tensor([], device=checker.device, dtype=self.dtype)

    @torch.inference_mode()
    def drawBoxesTorch(self, frame, boxes, classIds):
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].int()
            color = self.colorsTensor[classIds[i].item()]
            frame[:, :, y1:y2, x1:x2] = color.view(1, 3, 1, 1) * 0.3 + frame[:, :, y1:y2, x1:x2] * 0.7
        return frame

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            classIds, boxes, confidences = self.detect(frame)

            if boxes.numel() > 0 and boxes.shape[0] > 0:
                if self.disableAnnotations:
                    frame = self.drawBoxesTorch(frame, boxes, classIds)
                else:
                    frameNp = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    frameNp = (frameNp * 255).astype(np.uint8)
                    frameNp = cv2.cvtColor(frameNp, cv2.COLOR_RGB2BGR)

                    boxesNp = boxes.cpu().numpy()
                    classIdsNp = classIds.cpu().numpy()
                    confidencesNp = confidences.cpu().numpy()

                    outputImg = draw_detections(frameNp, boxesNp, confidencesNp, classIdsNp)
                    outputImg = cv2.cvtColor(outputImg, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(outputImg).permute(2, 0, 1).unsqueeze(0).float().to(checker.device) / 255.0

            self.writeBuffer.write(frame)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()


class ObjectDetection:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encodeMethod="x264",
        customEncoder="",
        benchmark=False,
        totalFrames=0,
        half=False,
        disableAnnotations=False,
    ):
        logging.error("CUDA object detection not yet implemented")
        raise NotImplementedError(
            "CUDA object detection is not yet implemented. Please use DirectML backend with --obj_detect_method yolov9_small-directml"
        )
