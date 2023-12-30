import numpy as np
import _thread
import subprocess
import logging
import time
import os
import requests
import cv2
import torch

from .train import AnimeSegmentation
from tqdm import tqdm
from queue import Queue

class Segment():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, inpoint=0, outpoint=0):
        self.input = input  # string
        self.output = output  # string
        self.ffmpeg_path = ffmpeg_path  # string
        self.width = width  # int, in my case it's 1920x802
        self.height = height  # int
        self.fps = fps  # 23.976
        self.nframes = nframes  # int
        self.inpoint = inpoint  # int, not used in this case
        self.outpoint = outpoint  # int, not used in this case

        # For debugging
        """
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
            
        """
        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue(maxsize=500)
        self.threads_done = False
        self.writing_finished = False
        
        self.handle_model()

        # Starting thread here just to make sure that the buffer is filled before the processing starts
        _thread.start_new_thread(self.build_buffer, ())

    def handle_model(self):
        filename = "isnetis.ckpt"
        url = r"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/isnetis.ckpt"
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        if not os.path.exists(os.path.join(dir_path , "weights")):
            os.mkdir(os.path.join(dir_path , "weights"))
            
        if not os.path.exists(os.path.join(dir_path, "weights", filename)):
            print("Downloading segmentation model...")
            logging.info("Couldn't find the model, downloading it now...")
            request = requests.get(url)
            with open(os.path.join(dir_path , "weights", filename), "wb") as file:
                file.write(request.content)
        model_path = os.path.join(dir_path , "weights", filename)
        
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        self.model = AnimeSegmentation.try_load("isnet_is", model_path, device, img_size=1024)
        self.model.eval()
        self.model.to(device)
        
    def get_character_bounding_box(self, image) -> tuple:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(self.model.device)
        with torch.no_grad():
            pred = self.model(tmpImg)
            pred = pred.cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))
            pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
            pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
            return pred
    
    def build_buffer(self):
        ffmpeg_command = [
            self.ffmpeg_path,
            "-i", str(self.input),
        ]

        if self.outpoint != 0:
            ffmpeg_command.extend(
                ["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        ffmpeg_command.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "quiet",
            "-stats",
            "-",
        ])

        try:
            process = subprocess.Popen(
                ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            logging.info(f"Running command: {ffmpeg_command}")

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
        except:
            logging.exception("An error occurred during reading")
        finally:
            # For terminating the pipe and subprocess properly
            process.stdout.close()
            process.stderr.close()
            process.terminate()
            self.read_buffer.put(None)

    def write_buffer(self):
        command = [self.ffmpeg_path,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{self.width}x{self.height}',
                   '-pix_fmt', 'rgba',
                   '-r', str(self.fps),
                   '-i', '-',
                   '-an',
                   '-c:v', 'libx264',
                   '-preset', 'veryfast',
                   '-crf', '15',
                   '-tune', 'animation',
                   '-movflags', '+faststart',
                   self.output]
        
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                if not self.processed_frames:
                    if self.read_buffer.empty() and self.threads_done == True:
                        break
                    continue

                frame = self.processed_frames.get()

                # Write the frame to FFmpeg
                if frame is not None:
                    pipe.stdin.write(frame.tobytes())

                self.pbar.update(1)
        except Exception as e:
            logging.exception("An error occurred during writing")

        finally:
            # Close the pipe
            pipe.stdin.close()
            pipe.wait()

            self.pbar.close()
            self.writing_finished = True

    def start_process(self):
        green_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        green_img[..., 1] = 255  # 255 for greenscreen
        last_bounding_box = None # for caching the bounding box of the last input, TO DO
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None and self.read_buffer.empty():
                    break
                                
                mask = self.get_mask(frame)
                
                frame = (frame * mask + green_img * (1 - mask)).astype(np.uint8)
                
                mask = (mask * 255).astype(np.uint8)
                mask = np.squeeze(mask, axis=2)
                
                frame_with_mask = np.concatenate((frame, mask[..., np.newaxis]), axis=2)
                
                self.processed_frames.put(frame_with_mask)

        except:
            logging.exception("An error occurred during processing")

        finally:
            logging.info("Processing of Segmentation is now finished")
            self.threads_done = True

    def run(self):
        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        _thread.start_new_thread(self.start_process, ())
        _thread.start_new_thread(self.write_buffer, ())

        while True:
            if self.threads_done == True and self.read_buffer.empty() and self.processed_frames.empty():
                break
            time.sleep(0.1)
