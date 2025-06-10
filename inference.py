import vapoursynth as vs
import torch
import numpy as np
import logging

from src.utils.isCudaInit import CudaChecker

core = vs.core
core.num_threads = 4
checker = CudaChecker()


class UniversalPytorch:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            trt (bool): Whether to use tensorRT
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        from spandrel import ImageModelDescriptor, ModelLoader
        from spandrel.__helpers.model_descriptor import UnsupportedDtypeError

        modelPath = r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\weights\superultracompact\2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth"
        if self.upscaleMethod not in ["rtmosr"]:  # using  a list for future expansion
            try:
                self.model = ModelLoader().load_from_file(modelPath)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        else:
            if self.upscaleMethod == "rtmosr":
                from src.extraArches.RTMoSR import RTMoSR

                self.model = RTMoSR(unshuffle_mod=True)
            self.model = torch.load(modelPath, map_location="cpu", weights_only=False)

        if self.customModel:
            assert isinstance(self.model, ImageModelDescriptor)

        try:
            # If the model is wrapped in a ModelDescriptor, extract the underlying model
            self.model = self.model.model
        except Exception:
            pass

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision: {e}")
                self.model = self.model.float()
                self.half = False
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        self.dummyInput = (
            torch.zeros(
                (1, 3, self.height, self.width),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .contiguous()
            .to(memory_format=torch.channels_last)
        )

        self.dummyOutput = (
            torch.zeros(
                (
                    1,
                    3,
                    self.height * self.upscaleFactor,
                    self.width * self.upscaleFactor,
                ),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .contiguous()
            .to(memory_format=torch.channels_last)
        )

        self.stream = torch.cuda.Stream()

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.model(self.dummyInput)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            if isinstance(frame, vs.VideoFrame):
                frame_tensor = frame_to_tensor(frame, checker.device)
            else:
                frame_tensor = frame
            self.dummyInput.copy_(
                frame_tensor.to(dtype=self.dummyInput.dtype).to(
                    memory_format=torch.channels_last
                ),
                non_blocking=True,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame) -> torch.tensor:
        self.processFrame(frame)
        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()

        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.clone()
        self.outputStream.synchronize()

        if isinstance(frame, vs.VideoFrame):
            return tensor_to_frame(output, frame, self.stream)

        return output


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    ).unsqueeze(0)


def tensor_to_frame(
    tensor: torch.Tensor, original_frame: vs.VideoFrame, stream: torch.cuda.Stream
) -> vs.VideoFrame:
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).detach()
    tensors = [
        tensor[plane].to("cpu", non_blocking=True) for plane in range(tensor.shape[0])
    ]

    stream.synchronize()

    # Get upscaled dimensions
    height, width = tensor.shape[-2:]

    # Create new frame with upscaled dimensions
    new_frame = core.std.BlankClip(
        width=width, height=height, format=original_frame.format, length=1
    ).get_frame(0)

    # Copy tensor data to new frame
    for plane in range(len(tensors)):
        np.copyto(np.asarray(new_frame[plane]), tensors[plane].numpy())

    return new_frame


def inferenceClip(
    videoPath: str,
    args: list,
    clip: vs.VideoNode = None,
) -> vs.VideoNode:
    """
    Inference function that processes a video clip.
    If a clip is provided, it will be used; otherwise, the videoPath will be loaded.
    """
    try:
        clip = core.bs.VideoSource(r"F:\TheAnimeScripter\input\1080.mp4")
    except Exception as e:
        logging.error(f"Error loading video: {e}")
        return None

    clip = vs.core.resize.Bicubic(
        clip,
        format=vs.RGBH,
        matrix_in_s="709",
    )

    upscaler = UniversalPytorch(
        "superultracompact",
        2,
        half=True,
        width=clip.width,
        height=clip.height,
    )

    # Apply the upscaler to each frame
    @torch.inference_mode()
    def process_frame(n, f):
        try:
            # Process the frame through the upscaler
            return upscaler(f)
        except Exception as e:
            logging.error(f"Error processing frame {n}: {e}")
            return f  # Return original frame on error

    # Use std.ModifyFrame for proper frame processing
    upscaled_clip = core.std.ModifyFrame(clip, clip, process_frame)

    # Convert back to YUV420P8 for output compatibility
    output_clip = vs.core.resize.Bicubic(
        upscaled_clip,
        format=vs.YUV420P8,
        matrix_s="709",
    )

    return output_clip


if __name__ == "__main__":
    clip = inferenceClip("", [])
    clip.set_output()
