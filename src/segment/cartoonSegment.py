import logging
import os
import torch
import torch.nn.functional as F

from src.downloadModels import downloadModels, weightsDir, modelsMap
from concurrent.futures import ThreadPoolExecutor
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from alive_progress import alive_bar


class CartoonSegment:  # A bit ambiguous because of .train import AnimeSegmentation but it's fine
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
    ):
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        self.handleModel()
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
                totalFrames=self.totalFrames,
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
                grayscale=False,
                transparent=True,
                audio=False,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):
        filename = modelsMap("segment")
        if not os.path.exists(os.path.join(weightsDir, "segment", filename)):
            modelPath = downloadModels(model="segment")
        else:
            modelPath = os.path.join(weightsDir, "segment", filename)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        from .train import AnimeSegmentation

        self.model = AnimeSegmentation.try_load(
            "isnet_is", modelPath, self.device, img_size=1024
        )
        self.model.eval()
        self.model.to(self.device)

    def get_mask(self, input_img: torch.Tensor) -> torch.Tensor:
        s = 1024
        h, w = h0, w0 = input_img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        input_img = (
            input_img.float().to(self.device).mul(1 / 255).permute(2, 0, 1).unsqueeze(0)
        )
        img_input = F.interpolate(
            input_img,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        img_input = F.pad(img_input, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        with torch.no_grad():
            pred = self.model(img_input)
            pred = pred[:, :, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
            pred = (
                F.interpolate(pred, size=(h0, w0), mode="bilinear", align_corners=False)
                .squeeze_(0)
                .permute(1, 2, 0)
                .mul(255)
                .to(torch.uint8)
            )
            return pred

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            mask = self.get_mask(frame)
            mask = torch.squeeze(mask, dim=2)
            frameWithmask = torch.cat((frame.to(self.device), mask.unsqueeze(2)), dim=2)
            self.writeBuffer.write(frameWithmask)
        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with alive_bar(
            self.totalFrames, title="Processing", bar="smooth", unit="frames"
        ) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                self.processFrame(frame)
                frameCount += 1
                bar(1)

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
