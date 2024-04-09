import os
import torch
import numpy as np
import logging
import torch.nn.functional as F
#import torch_tensorrt as trt

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
                modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)
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

    
        self.padWidth = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.padHeight = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

        """
        dummyInput = torch.zeros((1, 3, self.height, self.width), device='cuda', dtype=torch.float16)
        self.model = trt.compile(self.model, dummyInput, enabled_precisions={torch.half})
        """

    @torch.inference_mode()
    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.padWidth, 0, self.padHeight])
        return frame


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
                # torch.cuda.set_stream(self.stream[self.currentStream])
                frame = frame.cuda(non_blocking=True)
                if self.half:
                    frame = frame.half()
            
            if self.padWidth != 0 or self.padHeight != 0:
                frame = self.pad_frame(frame)
                
            frame = self.model(frame)
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).byte()

            if self.isCudaAvailable:
                torch.cuda.synchronize(self.stream[self.currentStream])
                self.currentStream = (self.currentStream + 1) % len(self.stream)

            return frame.cpu().numpy()

"""

Another TensorRT implementation that uses CUDA for data transfer

NOTE:
    Not functional yet
    
class shuffleCuganDirectML:
    def __init__(self):
        model = r"G:\TheAnimeScripter\2x_AnimeJaNai_HD_V3_Compact_583k-fp16.onnx"
        enginePath = r"G:\TheAnimeScripter\2x_AnimeJaNai_HD_V3_Compact_583k-fp16.engine"
        print(f"Using model: {model}")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        # Create a PyCUDA context and make it current
        self.context = cuda.Device(0).make_context()

        # Load the engine if it exists, otherwise build it and save it
        if os.path.exists(enginePath):
            with open(enginePath, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
        else:
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.fp16_mode = True
                with open(model, 'rb') as model:
                    parser.parse(model.read())
                engine = builder.build_cuda_engine(network)
                with open(enginePath, 'wb') as f:
                    f.write(engine.serialize())

        self.execution_context = engine.create_execution_context()
        input = np.empty([1,3,1080,1920], dtype=np.float16)
        input = np.ascontiguousarray(input)

        output = np.empty([1,3,2160,3840], dtype=np.float16)
        output = np.ascontiguousarray(output)

        self.d_input = cuda.mem_alloc(1 * input.size * input.dtype.itemsize)
        self.d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def run(self, frame: np.ndarray) -> np.ndarray:
        self.context.push()
        try:
            frame = frame.transpose((2, 0, 1))
            frame = frame.reshape(1, 3, 1080, 1920)
            frame = frame.astype(np.float16) / 255.0
            print(1
            frame = np.ascontiguousarray(frame)  # Ensure that frame is contiguous

            cuda.memcpy_htod(self.d_input, frame)

            # Execute model
            self.execution_context.execute(1, self.bindings)
            
            output = np.empty([1,3,2160,3840], dtype=np.float16)
            cuda.memcpy_dtoh(output, self.d_output)

            output = output.reshape(3, 2160, 3840)        
            output = (output * 255).astype(np.uint8)

        except Exception as e:
            logging.error(f"Error running model: {e}")
            
        finally:
            self.context.pop()

        return output


"""

"""
ONNXRUNTIME DIRECTML

NOTE:
    Output needs clamping to [0-255], not sure why but it's a quirk of the models
    Performance is abysmal due to numpy data transfer, ~3FPS for 1080p

class shuffleCuganDirectML:
    def __init__(self):
        model = (
            r"G:\TheAnimeScripter\2x_AnimeJaNai_HD_V3_Compact_583k-fp16.onnx"
        )

        print(f"Using model: {model}")

        providers = ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            self.model = ort.InferenceSession(
                model, providers=["DmlExecutionProvider"]
            )
        else:
            self.model = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

        self.IoBinding = self.model.io_binding()
        self.frame = np.zeros((1, 3, 1080, 1920), dtype=np.float16)
        self.output = np.zeros((1, 3, 2160, 3840), dtype=np.float16)

        self.IoBinding.bind_output(
            name='output',
            device_type='cpu',
            device_id=0,
            element_type=np.float16,
            shape=self.output.shape,  # Output shape
            buffer_ptr=self.output.ctypes.data,
        )

    def run(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape(1, 3, 1080, 1920)
        frame = frame / 255
        frame = frame.astype(np.float16)

        np.copyto(self.frame, frame)

        self.IoBinding.bind_input(
            name='input',
            device_type='cpu',
            device_id=0,
            element_type=np.float16,
            shape=self.frame.shape,
            buffer_ptr=self.frame.ctypes.data,
        )

        self.model.run_with_iobinding(self.IoBinding)

        output = self.output.reshape(3, 2160, 3840)
        output = output.transpose((1, 2, 0))
        output *= 255
        output = np.clip(output, 0, 255)
        output = output.astype(np.uint8)

        return output
"""

"""
ONNXRUNTIME CUDA 

NOTE:         
    # Shufflecugan does some shape ops on the CPU which then forces clipping to [0-255]
    # Performance is unoptimal because of it, 10fps for 1080p with ONNXRuntime CUDA whilst 16FPS with Pytorch CUDA
    # Looking for a workaround, I assume it's pixelshuffle that's causing the issue
    # Compact / UltraCompact / SuperUltraCompact work fine with ONNXRuntime CUDA and the performance is within margin of error (0.5FPS compared to Pytorch CUDA)

class shuffleCuganCUDA:
    def __init__(self):
    

        model = (
            r"G:\TheAnimeScripter\2x_AnimeJaNai_HD_V3_Compact_583k-fp16.onnx"
        )

        print(f"Using model: {model}")

        providers = ort.get_available_providers()
        # Print the list of providers
        print(providers)
        # Specify the CUDA provider
        if ort.get_device() == "GPU":
            self.model = ort.InferenceSession(
                model, providers=["CUDAExecutionProvider"]
            )
        else:
            self.model = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

        torch.set_default_dtype(torch.float16)
        
        isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if isCudaAvailable else "cpu")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Binding the I/O tensors
        # Allows for the model to be run with Torch Tensors which are faster than Numpy arrays
        # Since frame normalizations can be done on the GPU and it allows for less data transfer between CPU and GPU

        self.IoBinding = self.model.io_binding()

        # Input and output tensors
        input = torch.zeros((1, 3, 1080, 1920), device='cuda', dtype=torch.float16)
        self.output = torch.zeros((1, 3, 2160, 3840), device='cuda', dtype=torch.float16)

        # input is a placeholder tensor
        # Pytorch overwrites the input tensor with the frame tensor
        # The input tensor is then bound to the input of the model
        # The output tensor is bound to the output of the model

        self.IoBinding.bind_input(
            name='input',
            device_type='cuda',
            device_id=0,
            element_type=np.float16,
            shape=tuple([1, 3, 1080, 1920]),  # Input shape
            buffer_ptr=input.data_ptr(),
            )
        
        self.IoBinding.bind_output(
            name='output',
            device_type='cuda',
            device_id=0,
            element_type=np.float16,
            shape=tuple([1, 3, 2160, 3840]),  # Output shape
            buffer_ptr=self.output.data_ptr(),
        )

    def run(self, frame: np.ndarray) -> np.ndarray:
        frame = (
            torch.from_numpy(frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
        )

        frame = frame.contiguous()
        frame = frame.to(self.device).half()

        self.model.run_with_iobinding(self.IoBinding)

        frame = self.output.squeeze(0).permute(1, 2, 0).mul_(255).byte().cpu().numpy()
        return frame
        
"""

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
            engine = builder.buildEngine(network, config)
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
