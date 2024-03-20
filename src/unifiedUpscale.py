import os
import torch
import numpy as np
import logging
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# will be on wait for the next release of spandrel
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
            modelPath = r"G:\TheAnimeScripter\sudo_shuffle_cugan_fp16_op17_clamped_9.584.969 (1).onnx"
            # engine_file = self.buildEngine(modelPath)
            engine_file = r"G:\TheAnimeScripter\engine.trt"
            with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()
            input_shape = self.context.get_tensor_shape("input")
            output_shape = self.context.get_tensor_shape("output")
            
            self.h_input = cuda.pagelocked_empty(
                trt.volume(input_shape), dtype=np.float32
            )
            self.h_output = cuda.pagelocked_empty(
                trt.volume(output_shape), dtype=np.float32
            )
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.stream = cuda.Stream()

    def buildEngine(self, model_path):
        with open(model_path, "rb") as f:
            model = f.read()

        builder = trt.Builder(TRT_LOGGER)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.FP16)

        parser = trt.OnnxParser(network, TRT_LOGGER)
        if not parser.parse(model):
            print("Failed to parse the ONNX model.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (1, 3, self.height, self.width),
            (1, 3, self.height, self.width),
            (1, 3, self.height, self.width),
        )
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)

        with open("engine.trt", "wb") as f:
            f.write(engine.serialize())

        return "engine.trt"

    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a frame using a desired model, and return the upscaled frame
        Expects a numpy array of shape (height, width, 3) and dtype uint8
        """
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
                    if self.half:
                        frame = frame.cuda(non_blocking=True).half()
                    else:
                        frame = frame.cuda(non_blocking=True)
                else:
                    frame = frame.cpu()

                frame = self.model(frame)
                frame = (
                    frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
                )

                if self.isCudaAvailable:
                    torch.cuda.synchronize(self.stream[self.currentStream])
                    self.currentStream = (self.currentStream + 1) % len(self.stream)

                return frame.cpu().numpy()
        else:
            frame = frame.transpose((2, 0, 1))
            frame = np.ascontiguousarray(frame)

            np.copyto(self.h_input, frame.ravel())

            """
            WTF IS
            [03/20/2024-02:11:48] [TRT] [E] 1: [genericReformat.cuh::genericReformat::copyVectorizedRunKernel::1587] Error Code 1: Cuda Runtime (invalid resource handle)

            Idk where have I messed up but I'm not able to figure it out
            Piece of...
            """
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

            frame = self.h_output.reshape(
                (3, self.height * self.upscaleFactor, self.width * self.upscaleFactor)
            )

            frame = frame.transpose((1, 2, 0))
            frame = frame.astype(np.uint8)

            return frame
