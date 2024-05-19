import os
import torch
import logging
import cv2
import torch.nn.functional as F
import tensorrt as trt

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
        
        aspectRatio = self.width / self.height
        # Fix height at 518 and adjust width
        newHeight = 518
        newWidth = round(newHeight * aspectRatio / 14) * 14
        # Ensure newWidth is a multiple of 14
        newWidth = (newWidth // 14) * 14

        if not os.path.exists(modelPath.replace(".onnx", f"_{enginePrecision}.engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            profiles = [
                Profile().add(
                    "image",
                    min=(1, 3, newHeight, newWidth),
                    opt=(1, 3, newHeight, newWidth),
                    max=(1, 3, newHeight, newWidth),
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

            with TrtRunner(self.engine) as runner:
                self.runner = runner

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
        
        with open(modelPath.replace(".onnx", f"_{enginePrecision}.engine"), "rb") as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read()) 
            self.context = self.engine.create_execution_context()

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, newHeight, newWidth),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, newHeight, newWidth),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)
        
        # Warmup
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        with torch.cuda.stream(self.stream):
            for _ in range(10):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()
        

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.stream):
            try:
                # input is a torch.uint8 tensor
                frame = (frame / 255.0).numpy()
                frame = self.transform({"image": frame})["image"]

                frame = torch.from_numpy(frame).unsqueeze(0).to(self.device)

                if self.half and self.isCudaAvailable:
                    frame = frame.half()
                else:
                    frame = frame.float()

                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

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
                self.stream.synchronize()
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
