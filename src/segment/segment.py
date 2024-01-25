import numpy as np
import subprocess
import logging
import os
import cv2
import torch
import wget

from concurrent.futures import ThreadPoolExecutor
from .train import AnimeSegmentation
from tqdm import tqdm
from queue import Queue


class Segment():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, inpoint=0, outpoint=0, encode_method="x264"):
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.nframes = nframes
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method

        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        self.handle_model()

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue()

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.build_buffer)
            executor.submit(self.process)
            executor.submit(self.write_buffer)
        

    def handle_model(self):
        filename = "isnetis.ckpt"
        url = f"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/{
            filename}"

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if not os.path.exists(os.path.join(dir_path, "weights")):
            os.mkdir(os.path.join(dir_path, "weights"))

        if not os.path.exists(os.path.join(dir_path, "weights", filename)):
            print("Downloading segmentation model...")
            logging.info(
                "Couldn't find the segmentation model, downloading it now...")
            wget.download(url, out=os.path.join(dir_path, "weights", filename))

        model_path = os.path.join(dir_path, "weights", filename)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = AnimeSegmentation.try_load(
            "isnet_is", model_path, self.device, img_size=1024)
        self.model.eval()
        self.model.to(self.device)
        
    """
    def get_character_bounding_box(self, image) -> tuple:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = [cv2.boundingRect(c) for c in contours]
        x = min([b[0] for b in boxes])
        y = min([b[1] for b in boxes])
        w = max([b[0] + b[2] for b in boxes]) - x
        h = max([b[1] + b[3] for b in boxes]) - y

        return x, y, w, h
    """

    def get_mask(self, input_img):
        s = 1024
        input_img = (input_img / 255).astype(np.float32)
        h, w = h0, w0 = input_img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw //
                  2 + w] = cv2.resize(input_img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        tmpImg = torch.from_numpy(img_input).type(
            torch.FloatTensor).to(self.device)
        with torch.no_grad():
            pred = self.model(tmpImg)
            pred = pred.cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))
            pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
            pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
            return pred

    def build_buffer(self):
        from src.ffmpegSettings import decodeSettings

        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, False, 0, self.ffmpeg_path)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.reading_done = False
        frame_size = self.width * self.height * 3
        frame_count = 0

        try:
            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    continue
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                self.read_buffer.put(frame)
                frame_count += 1

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            logging.info(
                f"Built buffer with {frame_count} frames")

            process.stdout.close()
            process.terminate()

            self.reading_done = True
            self.read_buffer.put(None)

    def process(self):
        green_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        green_img[..., 1] = 255  # 255 for greenscreen
        prev_frame = None
        self.processing_done = False
        frame_count = 0
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    if self.reading_done == True and self.read_buffer.empty():
                        break
                    else:
                        continue
                
                mask = self.get_mask(frame)
                frame = (frame * mask + green_img *
                         (1 - mask)).astype(np.uint8)

                mask = (mask * 255).astype(np.uint8)
                mask = np.squeeze(mask, axis=2)

                frame_with_mask = np.concatenate(
                    (frame, mask[..., np.newaxis]), axis=2)

                self.processed_frames.put(frame_with_mask)
                frame_count += 1

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            if prev_frame is not None:
                self.processed_frames.put(prev_frame)
                frame_count += 1

            logging.info(
                f"Processed {frame_count} frames")

            self.processing_done = True
            self.processed_frames.put(None)

    def write_buffer(self):

        from src.ffmpegSettings import encodeSettings
        command: list = encodeSettings(self.encode_method, self.width, self.height,
                                       self.fps, self.output, self.ffmpeg_path, sharpen=False, sharpen_sens=0, grayscale=False)

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_count = 0
        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_done == True and self.processed_frames.empty():
                        break
                    else:
                        continue

                frame_count += 1
                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update()

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            logging.info(
                f"Wrote {frame_count} frames")

            pipe.stdin.close()
            self.pbar.close()
