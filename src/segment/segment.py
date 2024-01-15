import numpy as np
import _thread
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

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue(maxsize=500)

        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        self.handle_model()

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.write_buffer, ())

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.start_process)

    def handle_model(self):
        filename = "isnetis.ckpt"
        url = r"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/isnetis.ckpt"

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
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = AnimeSegmentation.try_load(
            "isnet_is", model_path, device, img_size=1024)
        self.model.eval()
        self.model.to(device)

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
            torch.FloatTensor).to(self.model.device)
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

        try:
            self.reading_done = False
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            frame_size = self.width * self.height * 3

            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    break
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))
                self.read_buffer.put(frame)

            stderr = process.stderr.read().decode()
            if stderr:
                if "bitrate=" not in stderr:
                    logging.error(f"ffmpeg error: {stderr}")

        except Exception as e:
            logging.exception(f"An error occurred during reading {e}")

        finally:
            # For terminating the pipe and subprocess properly
            process.stdout.close()
            process.stderr.close()
            process.terminate()
            self.read_buffer.put(None)
            self.reading_done = True

    def write_buffer(self):

        from src.ffmpegSettings import encodeSettings
        command: list = encodeSettings(
            self.encode_method, self.width, self.height, self.fps, self.output, self.ffmpeg_path, False, 0)
        
        command = [item.replace('rgb24', 'rgba')
                   if 'rgb24' in item else item for item in command]

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                if not self.processed_frames:
                    if self.read_buffer.empty() and self.processing_done:
                        break
                    continue

                frame = self.processed_frames.get()

                if frame is not None:
                    pipe.stdin.write(frame.tobytes())

                self.pbar.update(1)
        except Exception as e:
            logging.exception(f"An error occurred during writing {e}")

        finally:
            # Close the pipe
            pipe.stdin.close()
            pipe.wait()

            self.pbar.close()
            self.writing_finished = True

    def start_process(self):
        green_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        green_img[..., 1] = 255  # 255 for greenscreen
        last_bounding_box = None  # for caching the bounding box of the last input, TO DO
        self.processing_done = False
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None and self.read_buffer.empty():
                    break

                mask = self.get_mask(frame)

                frame = (frame * mask + green_img *
                         (1 - mask)).astype(np.uint8)

                mask = (mask * 255).astype(np.uint8)
                mask = np.squeeze(mask, axis=2)

                frame_with_mask = np.concatenate(
                    (frame, mask[..., np.newaxis]), axis=2)

                self.processed_frames.put(frame_with_mask)

        except Exception as e:
            logging.exception(f"An error occurred during processing {e}")

        finally:
            logging.info("Processing of Segmentation is now finished")
            self.processing_done = True
