import os
import torch
import logging
import cv2
import torch.nn.functional as F
import tensorrt as trt

from polygraphy.backend.trt import (
    engine_from_network,
    network_from_onnx_path,
    CreateConfig,
    Profile,
    SaveEngine,
)

from torchvision.transforms import Compose
from concurrent.futures import ThreadPoolExecutor
from .dpt import DPT_DINOv2
from .transform import Resize, NormalizeImage, PrepareForNet

from src.coloredPrints import yellow
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.downloadModels import downloadModels, weightsDir, modelsMap

os.environ["TORCH_HOME"] = os.path.dirname(os.path.realpath(__file__))
class Depth:
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
        match self.depth_method:
            case "small":
                model = "vits"
                self.model = DPT_DINOv2(
                    encoder="vits",
                    features=64,
                    out_channels=[48, 96, 192, 384],
                    localhub=False,
                )
            case "base":
                model = "vitb"
                self.model = DPT_DINOv2(
                    encoder="vitb",
                    features=128,
                    out_channels=[96, 192, 384, 768],
                    localhub=False,
                )

            case "large":
                model = "vitl"
                self.model = DPT_DINOv2(
                    encoder="vitl",
                    features=256,
                    out_channels=[256, 512, 1024, 1024],
                    localhub=False,
                )

        modelPath = os.path.join(weightsDir, model, f"depth_anything_{model}14.pth")

        if not os.path.exists(modelPath):
            print("Couldn't find the depth model, downloading it now...")

            logging.info("Couldn't find the depth model, downloading it now...")

            os.makedirs(weightsDir, exist_ok=True)
            modelPath = downloadModels(model=model)

        self.isCudaAvailable = torch.cuda.is_available()

        if self.isCudaAvailable:
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        self.model.load_state_dict(
            torch.load(modelPath, map_location="cpu"), strict=True
        )

        if self.half and self.isCudaAvailable:
            self.model = self.model.half()
        else:
            self.half = False

        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        aspectRatio = self.width / self.height
        # Fix height at 518 and adjust width
        self.newHeight = 518
        newWidth = round(self.newHeight * aspectRatio / 14) * 14
        # Ensure newWidth is a multiple of 14
        self.newWidth = (newWidth // 14) * 14

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = frame.to(self.device).mul(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)
            
            frame = F.interpolate(
                frame.float(), (self.newHeight, self.newWidth), mode="bilinear", align_corners=False
            )
            frame = ((frame.to(self.device) - self.mean_tensor) / self.std_tensor)
            if self.half and self.isCudaAvailable:
                frame = frame.half()
    
            depth = self.model(frame)
            depth = F.interpolate(
                depth[None],
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
            self.writeBuffer.write(depth)
    
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        while True:
            frame = self.readBuffer.read()
            if frame is None:
                break
            self.processFrame(frame)
            frameCount += 1

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()

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
        self.newHeight = 518
        newWidth = round(self.newHeight * aspectRatio / 14) * 14
        # Ensure newWidth is a multiple of 14
        self.newWidth = (newWidth // 14) * 14

        if not os.path.exists(modelPath.replace(".onnx", f"_{enginePrecision}.engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            profiles = [
                Profile().add(
                    "image",
                    min=(1, 3, self.newHeight, self.newWidth),
                    opt=(1, 3, self.newHeight, self.newWidth),
                    max=(1, 3, self.newHeight, self.newWidth),
                ),
            ]
            self.engine = engine_from_network(
                network_from_onnx_path(modelPath),
                config=CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = SaveEngine(
                self.engine, modelPath.replace(".onnx", f"_{enginePrecision}.engine")
            )

            self.engine.__call__()
        
        with open(modelPath.replace(".onnx", f"_{enginePrecision}.engine"), "rb") as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read()) 
            self.context = self.engine.create_execution_context()

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, newWidth),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 1, self.newHeight, newWidth),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )
        
        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)
    
        with torch.cuda.stream(self.stream):
            for _ in range(10):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()
        

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.stream):
            try:
                frame = frame.to(self.device).mul(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)
                
                frame = F.interpolate(
                    frame.float(), (self.newHeight, self.newWidth), mode="bilinear", align_corners=False
                )
                frame = ((frame.to(self.device) - self.mean_tensor) / self.std_tensor)
                if self.half and self.isCudaAvailable:
                    frame = frame.half()

                self.dummyInput.copy_(frame)
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

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

    def process(self):
        frameCount = 0
        while True:
            frame = self.readBuffer.read()
            if frame is None:
                break
            self.processFrame(frame)
            frameCount += 1

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
