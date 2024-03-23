import os
import torch
import numpy as np
import logging

from spandrel import ImageModelDescriptor, ModelLoader
from .downloadModels import downloadModels, weightsDir, modelsMap

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")



class Upscaler:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        cuganKind: str = "conservative",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        nt: int = 1,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            cuganKind (str): The kind of cugan to use
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            nt (int): The number of threads to use
            trt (bool): Whether to use tensorRT
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.cuganKind = cuganKind
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        self.trt = False
        if not self.trt:
            if not self.customModel:
                self.filename = modelsMap(
                    self.upscaleMethod, self.upscaleFactor, self.cuganKind
                )
                if not os.path.exists(
                    os.path.join(weightsDir, self.upscaleMethod, self.filename)
                ):
                    modelPath = downloadModels(
                        model=self.upscaleMethod,
                        cuganKind=self.cuganKind,
                        upscaleFactor=self.upscaleFactor,
                    )

                else:
                    modelPath = os.path.join(
                        weightsDir, self.upscaleMethod, self.filename
                    )

            else:
                if os.path.isfile(self.customModel):
                    modelPath = self.customModel

                else:
                    raise FileNotFoundError(
                        f"Custom model file {self.customModel} not found"
                    )

            try:
                self.model = ModelLoader().load_from_file(modelPath)
            except Exception as e:
                logging.error(f"Error loading model: {e}")

            if self.customModel:
                assert isinstance(self.model, ImageModelDescriptor)

            self.isCudaAvailable = torch.cuda.is_available()
            self.model = (
                self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
            )

            self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

            if self.isCudaAvailable:
                #self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
                #self.currentStream = 0
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                if self.half:
                    torch.set_default_dtype(torch.float16)
                    self.model.half()

    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a frame using a desired model, and return the upscaled frame
        Expects a numpy array of shape (height, width, 3) and dtype uint8
        """
        with torch.no_grad():
            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .mul_(1 / 255)
            )

            frame = frame.contiguous(memory_format=torch.channels_last)

            if self.isCudaAvailable:
                #torch.cuda.set_stream(self.stream[self.currentStream])
                frame = frame.cuda(non_blocking=True)
                if self.half:
                    frame = frame.half()

            frame = self.model(frame)
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).byte()

            """
            if self.isCudaAvailable:
                torch.cuda.synchronize(self.stream[self.currentStream])
                self.currentStream = (self.currentStream + 1) % len(self.stream)
            """

            return frame.cpu().numpy()


"""
import os
import torch
import numpy as np
import logging
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from spandrel import ImageModelDescriptor, ModelLoader
from .downloadModels import downloadModels, weightsDir, modelsMap

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class Upscaler:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        cuganKind: str = "conservative",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        nt: int = 1,
    ):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.cuganKind = cuganKind
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        self.trt = False
        if not self.trt:
            if not self.customModel:
                self.filename = modelsMap(
                    self.upscaleMethod, self.upscaleFactor, self.cuganKind
                )
                if not os.path.exists(
                    os.path.join(weightsDir, self.upscaleMethod, self.filename)
                ):
                    modelPath = downloadModels(
                        model=self.upscaleMethod,
                        cuganKind=self.cuganKind,
                        upscaleFactor=self.upscaleFactor,
                    )

                else:
                    modelPath = os.path.join(
                        weightsDir, self.upscaleMethod, self.filename
                    )

            else:
                if os.path.isfile(self.customModel):
                    modelPath = self.customModel

                else:
                    raise FileNotFoundError(
                        f"Custom model file {self.customModel} not found"
                    )

            try:
                self.model = ModelLoader().load_from_file(modelPath)
            except Exception as e:
                logging.error(f"Error loading model: {e}")

            if self.customModel:
                assert isinstance(self.model, ImageModelDescriptor)

            self.isCudaAvailable = torch.cuda.is_available()
            self.model = (
                self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
            )

            self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

            if self.isCudaAvailable:
                self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
                self.currentStream = 0
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                if self.half:
                    torch.set_default_dtype(torch.float16)
                    self.model.half()
        else:
            # Last time I checked only Me and God knew what was going on
            # Now only God knows
            modelPath = r"G:\TheAnimeScripter\sudo_shuffle_cugan_fp16_op18_clamped_9.584.969 (1).onnx"
            args = type("", (), {})()
            args.mode = "fp16"
            args.onnx_file_path = modelPath
            args.batch_size = 1
            args.engine_file_path = r"G:\TheAnimeScripter\engine.trt"

            self.engine = self.loadEngine2TensorRT(args.engine_file_path)
            # self.engine = self.ONNX2TRT(args)
            self.context = self.engine.create_execution_context()

            self.h_input = cuda.pagelocked_empty(
                (1, 3, self.height, self.width), dtype=np.float32
            )
            self.h_output = cuda.pagelocked_empty(
                (
                    1,
                    3,
                    self.height * self.upscaleFactor,
                    self.width * self.upscaleFactor,
                ),
                dtype=np.float32,
            )
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.stream = cuda.Stream()

    def ONNX2TRT(self, args, calib=None):
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(G_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, trt.OnnxParser(network, G_LOGGER) as parser:
            builder.max_batch_size = args.batch_size

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30

            profile = builder.create_optimization_profile()
            profile.set_shape(
                "input", (1, 3, 8, 8), (1, 3, 1080, 1920), (1, 3, 1080, 1920)
            )
            config.add_optimization_profile(profile)
            # builder.max_workspace_size = 1 << 30
            if args.mode.lower() == "int8":
                assert builder.platform_has_fast_int8, "not support int8"
                assert calib is not None, "need calib!"
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib
            elif args.mode.lower() == "fp16":
                assert builder.platform_has_fast_fp16, "not support fp16"
                config.set_flag(trt.BuilderFlag.FP16)

            print("Loading ONNX file from path {}...".format(args.onnx_file_path))
            with open(args.onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    for e in range(parser.num_errors):
                        print(parser.get_error(e))
                    raise TypeError("Parser parse failed.")

            print("Parsing ONNX file complete!")

            print(
                "Building an engine from file {}; this may take a while...".format(
                    args.onnx_file_path
                )
            )
            engine = builder.build_engine(network, config)
            if engine is not None:
                print("Create engine success! ")
            else:
                print("ERROR: Create engine failed! ")
                return

            print("Saving TRT engine file to path {}...".format(args.engine_file_path))
            with open(args.engine_file_path, "wb") as f:
                f.write(engine.serialize())

            print("Engine file has already saved to {}!".format(args.engine_file_path))

            return engine

    def loadEngine2TensorRT(self, filepath):
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine

    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
    
        if not self.trt:
            with torch.no_grad():
                frame = (
                    torch.from_numpy(frame)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .mul_(1 / 255)
                )

                frame = frame.contiguous(memory_format=torch.channels_last)

                if self.isCudaAvailable:
                    torch.cuda.set_stream(self.stream[self.currentStream])
                    frame = frame.cuda(non_blocking=True)
                    if self.half:
                        frame = frame.half()

                frame = self.model(frame)
                frame = (
                    frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
                )

                if self.isCudaAvailable:
                    torch.cuda.synchronize(self.stream[self.currentStream])
                    self.currentStream = (self.currentStream + 1) % len(self.stream)

                return frame.cpu().numpy()
        else:
            frame = np.ascontiguousarray(frame)
            frame = frame.transpose((2, 0, 1))

            frame = frame.reshape(1, 3, 1080, 1920)
            np.copyto(self.h_input, frame)
            #print(f"Shape of h_input: {self.h_input.shape}")
            #print(f"Shape of frame: {frame.shape}")

            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle,
            )
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

            frame = self.h_output.reshape(
                (3, self.height * self.upscaleFactor, self.width * self.upscaleFactor)
            )

            frame = frame.transpose((1, 2, 0))
            frame = frame.astype(np.uint8)

            return frame
"""
