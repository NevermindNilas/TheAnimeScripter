import logging
import os
import torch
import cv2
import numpy as np

from torch.nn import functional as F
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from concurrent.futures import ThreadPoolExecutor
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.isCudaInit import CudaChecker

checker = CudaChecker()

# FROM https://github.com/ibaiGorordo/ONNX-YOLOv9-MIT-Object-Detection
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

available_models = [
    "v9-s_mit",
    "v9-m_mit",
    "v9-c_mit",
    "gelan-c",
    "gelan-e",
    "yolov9-c",
    "yolov9-e",
]


def draw_masks(
    image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3
) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(
    image: np.ndarray,
    text: str,
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    font_size: float = 0.001,
    text_thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_size,
        thickness=text_thickness,
    )
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    return cv2.putText(
        image,
        text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f"{label} {int(score * 100)}%"
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


class ObjectDetection:
    def __init__(
        self,
        input: str,
        output: str,
        ffmpegPath: str,
        width: int,
        height: int,
        outputFPS: int,
        inpoint: int,
        outpoint: int,
        encodeMethod: str,
        customEncoder: str,
        benchmark: bool,
        totalFrames: int,
        half: bool,
    ):
        self.input = input
        self.output = output
        self.ffmpegPath = ffmpegPath
        self.width = width
        self.height = height
        self.outputFPS = outputFPS
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.half = half

        self._handleModel()

        self._start()

    def _handleModel(self):
        filename = modelsMap("yolo11n")
        if not os.path.exists(os.path.join(weightsDir, "yolo11n", filename)):
            modelPath = downloadModels(model="yolo11n")
        else:
            modelPath = os.path.join(weightsDir, "yolo11n", filename)

        self.model = torch.load(modelPath, map_location=checker.device)["model"]

        self.model.eval()
        self.model.to(checker.device)
        self.model.half() if self.half else self.model.float()

        self.stream = torch.cuda.Stream()

    def _start(self):
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.outputFPS,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                ffmpegPath=self.ffmpegPath,
                encode_method=self.encodeMethod,
                custom_encoder=self.customEncoder,
                grayscale=False,
                width=self.width,
                height=self.height,
                fps=self.outputFPS,
                sharpen=False,
                transparent=True,
                audio=False,
                benchmark=self.benchmark,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    @torch.inference_mode()
    def resizeFrame(self, frame, height=640, width=640):
        try:
            with torch.cuda.stream(self.stream):
                frame = F.interpolate(frame, size=(height, width), mode="bilinear")
            return frame
        except Exception as e:
            logging.error(f"An error occurred while resizing the frame: {e}")

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            reference = self.resizeFrame(frame)

            with torch.cuda.stream(self.stream):
                predictions = self.model(reference)

            ogFrame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            ogFrame = (ogFrame * 255).astype("uint8")

            if isinstance(predictions, tuple) and len(predictions) > 0:
                detections = predictions[0]

                for det in detections:
                    if len(det) >= 6:
                        box = det[1:5].int().cpu().numpy()
                        conf = float(det[5])
                        cls = int(det[6])

                        cv2.rectangle(
                            ogFrame,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 255, 0),
                            2,
                        )

                        label = f"Class {cls}: {conf:.2f}"
                        cv2.putText(
                            ogFrame,
                            label,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

            outputTensor = torch.from_numpy(ogFrame).to(checker.device)
            outputTensor = outputTensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

            self.writeBuffer.write(outputTensor)

        except Exception as e:
            logging.error(f"An error occurred while processing the frame: {e}")
            self.writeBuffer.write(frame)

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class ObjectDetectionDML:
    def __init__(
        self,
        input: str,
        output: str,
        width: int,
        height: int,
        outputFPS: int,
        inpoint: int,
        outpoint: int,
        encodeMethod: str,
        customEncoder: str,
        benchmark: bool,
        half: bool,
        objDetectMethod: str = "yolov9_small-directml",
        totalFrames: int = 0,
    ):
        """Object Detection using DirectML (DML) with ONNX Runtime.

        Args:
            input (str): Path to the input video file.
            output (str): Path to the output video file.
            width (int): Width of the output video.
            height (int): Height of the output video.
            outputFPS (int): Frames per second for the output video.
            inpoint (int): Start point in seconds for processing the video.
            outpoint (int): End point in seconds for processing the video.
            encodeMethod (str): Encoding method for the output video.
            customEncoder (str): Custom encoder to use for the output video.
            benchmark (bool): Whether to enable benchmarking.
            half (bool): Whether to use half precision for the model.
        """

        import onnxruntime as ort

        self.ort = ort

        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.outputFPS = outputFPS
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.benchmark = benchmark
        self.half = half
        self.objDetectMethod = objDetectMethod
        self.totalFrames = totalFrames

        self.handleModel()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                resize=False,
                toTorch=False,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encodeMethod,
                self.customEncoder,
                self.width,
                self.height,
                self.outputFPS,
                sharpen=False,
                sharpen_sens=None,
                grayscale=True,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)
        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModel(self):
        self.filename = modelsMap(
            model=self.objDetectMethod, modelType="onnx", half=self.half
        )

        folderName = self.objDetectMethod.replace("-directml", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(
                model=self.objDetectMethod,
                half=self.half,
                modelType="onnx",
            )
        else:
            modelPath = os.path.join(weightsDir, folderName, self.filename)

        providers = self.ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        if self.half:
            self.numpyDType = np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = np.float32
            self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, 480, 640),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 6),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.IoBinding.bind_output(
            name="pred_bbox",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = frame.astype(np.float32)
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            frame = torch.from_numpy(frame).to(self.deviceType).to(self.torchDType)
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            print("Frame shape:", frame.shape, "dtype:", frame.dtype)
            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)
            output = self.dummyOutput.cpu().numpy()

            self.writeBuffer.write(output)

        except Exception as e:
            logging.error(f"An error occurred while processing the frame: {e}")
            self.writeBuffer.write(frame)

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
