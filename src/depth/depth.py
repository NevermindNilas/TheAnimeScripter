import os
import torch
import logging
import cv2
import torch.nn.functional as F

from threading import Semaphore
from torchvision.transforms import Compose
from concurrent.futures import ThreadPoolExecutor
from .dpt import DPT_DINOv2
from .transform import Resize, NormalizeImage, PrepareForNet
from src.ffmpegSettings import BuildBuffer, WriteBuffer

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
                self.model = DPT_DINOv2(
                    encoder="vits",
                    features=64,
                    out_channels=[48, 96, 192, 384],
                    localhub=False,
                )
            case "base":
                self.model = DPT_DINOv2(
                    encoder="vitb",
                    features=128,
                    out_channels=[96, 192, 384, 768],
                    localhub=False,
                )

            case "large":
                self.model = DPT_DINOv2(
                    encoder="vitl",
                    features=256,
                    out_channels=[256, 512, 1024, 1024],
                    localhub=False,
                )

        self.isCudaAvailable = torch.cuda.is_available()

        if self.isCudaAvailable:
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        if self.half and self.isCudaAvailable:
            self.model = self.model.half()
        else:
            self.half = False

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

            depth = self.model(frame)

            depth = F.interpolate(
                depth[None],
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
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
