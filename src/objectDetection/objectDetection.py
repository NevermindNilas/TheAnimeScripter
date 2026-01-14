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
from .yolov9_mit import draw_detections, draw_masks, draw_box, colors

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
        logging.error("TensorRT object detection not yet implemented")
        raise NotImplementedError(
            "TensorRT object detection is not yet implemented. Please use DirectML backend with --obj_detect_method yolov9_small-directml"
        )


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
