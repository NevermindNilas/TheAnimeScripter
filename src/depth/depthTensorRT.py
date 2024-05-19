import os
import torch
import logging
import cv2
import torch.nn.functional as F

# Attempt to lazy load for faster startup
from polygraphy.backend.trt import (
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    CreateConfig,
    Profile,
    EngineFromBytes,
    SaveEngine,
)
from polygraphy.backend.common import BytesFromPath

from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import Compose

from src.coloredPrints import yellow
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.downloadModels import downloadModels, weightsDir, modelsMap
from .transform import NormalizeImage, PrepareForNet, Resize

class DepthTensorRT:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        nt=1,
        buffer_limit=50,
        benchmark=False,
    ):
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.nt = nt
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                input=self.input,
                ffmpegPath=self.ffmpeg_path,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                dedup=False,
                dedupSens=None,
                width=self.width,
                height=self.height,
                resize=False,
                resizeMethod=None,
                queueSize=self.buffer_limit,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                self.buffer_limit,
                sharpen=False,
                sharpen_sens=None,
                grayscale=True,
                audio=False,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        if not os.path.exists(os.path.join(weightsDir, self.filename, self.filename)):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="onnx",
            )
        else:
            modelPath = os.path.join(weightsDir, self.filename, self.filename)

        enginePrecision = "fp16" if "float16" in self.filename else "fp32"
        
        if not os.path.exists(modelPath.replace(".onnx", f"_{enginePrecision}.engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            aspect_ratio = self.width / self.height

            # Fix height at 518 and adjust width
            new_height = 518
            new_width = round(new_height * aspect_ratio / 14) * 14

            # Ensure new_width is a multiple of 14
            new_width = (new_width // 14) * 14

            profiles = [
                Profile().add(
                    "image",
                    min=(1, 3, new_height, new_width),
                    opt=(1, 3, new_height, new_width),
                    max=(1, 3, new_height, new_width),
                ),
            ]
            self.engine = engine_from_network(
                network_from_onnx_path(modelPath),
                config=CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = SaveEngine(
                self.engine, modelPath.replace(".onnx", f"_{enginePrecision}.engine")
            )

        else:
            self.engine = EngineFromBytes(
                BytesFromPath(modelPath.replace(".onnx", f"_{enginePrecision}.engine"))
            )

        self.runner = TrtRunner(self.engine)
        self.runner.activate()

        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        
        """
        # Warmup
        dummyInput = torch.zeros(
            (1, 3, new_height, new_width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        for _ in range(5):
            self.runner.infer(
                {
                    "image": dummyInput,
                },
                check_inputs=False,
            )
        """

        self.dummyOutput = torch.zeros(
            (1, 1, 518, 924),
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            # input is a torch.uint8 tensor
            frame = (frame / 255.0).numpy()
            frame = self.transform({"image": frame})["image"]

            frame = torch.from_numpy(frame).unsqueeze(0).to(self.device)

            if self.half and self.isCudaAvailable:
                frame = frame.half()
            else:
                frame = frame.float()

            self.dummyOutput.copy_(self.runner.infer(
                {
                    "image": frame,
                },
                check_inputs=False,
            )["depth"].float(), non_blocking=True
            )

            depth = F.interpolate(
                self.dummyOutput,
                size=[self.height, self.width],
                mode="bilinear",
                align_corners=False,
            )

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

        finally:
            self.semaphore.release()

    def process(self):
        frameCount = 0
        self.semaphore = Semaphore(self.nt * 4)
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            while True:
                frame = self.readBuffer.read()
                if frame is None:
                    if (
                        self.readBuffer.isReadingDone()
                        and self.readBuffer.getSizeOfQueue() == 0
                    ):
                        break

                self.semaphore.acquire()
                executor.submit(self.processFrame, frame)
                frameCount += 1

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
