import os
import requests
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn as nn
import cv2
import _thread
import skvideo.io
import warnings
from tqdm import tqdm
from queue import Queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

#warnings.filterwarnings("ignore")

'''
https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/src/cugan.py
'''
@torch.inference_mode()
class Cugan():
    def __init__(self, video_file, output, scale, half, kind_model, pro, w, h, nt):
        self.video_file = video_file
        self.output = output
        self.scale = scale
        self.half = half
        self.kind_model = kind_model
        self.pro = pro
        self.w = int(w * scale)
        self.h = int(h * scale)
        self.nt = nt
        self._initialize()
    
    def handle_models(self):
        if not os.path.exists("src/cugan/weights"):
            os.makedirs("src/cugan/weights")

        downloading = False
        if not os.path.exists(os.path.join(os.path.abspath("src/cugan/weights"), self.filename)):
            if not downloading:
                print("Downloading Cugan models...")
                downloading = True
            url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{self.filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join("src/cugan/weights", self.filename), "wb") as file:
                    file.write(response.content)
    
    def _initialize(self):
        model_path_prefix = "cugan_pro" if self.pro else "cugan"
        model_path_suffix = "-latest" if not self.pro else ""
        model_path_middle = f"up{self.scale}x"

        if self.scale == 2:
            self.model = UpCunet2x(in_channels=3, out_channels=3)
        elif self.scale == 3:
            self.model = UpCunet3x(in_channels=3, out_channels=3)
        elif self.scale == 4:
            self.model = UpCunet4x(in_channels=3, out_channels=3)

        self.filename = f"{model_path_prefix}_{model_path_middle}{model_path_suffix}-{self.kind_model}.pth"
        self.handle_models()
        
        model_path = os.path.join("src/cugan/weights", self.filename)
        model_path = os.path.abspath(model_path)
        
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval().cuda()
        if self.half:
            self.model.half()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        if not self.video_file is None:
            self.videoCapture = cv2.VideoCapture(self.video_file)
            fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
            self.tot_frame = self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.videoCapture.release()
            self.videogen = skvideo.io.vreader(self.video_file)
            self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        
        if self.output is not None:
            self.vid_out_name = self.output
        self.vid_out = cv2.VideoWriter(self.vid_out_name, self.fourcc, fps, (self.w, self.h))
        
        self.pbar = tqdm(total=self.tot_frame)
        self.write_buffer = Queue(maxsize=500)
        self.read_buffer = Queue(maxsize=500)
        
        _thread.start_new_thread(self._build_read_buffer, ())
        _thread.start_new_thread(self._clear_write_buffer, ())
        
        threads = []
        for _ in range(self.nt):
            thread = UpscaleMT(self.device, self.model, self.nt, self.half, self.read_buffer, self.write_buffer, self.vid_out)
            thread.start()
            threads.append(thread)
        
        while not self.write_buffer.empty():
            time.sleep(0.1)
        
        for thread in threads:
            thread.join()
            
        self.pbar.close()
        if not self.vid_out is None:
            self.vid_out.release()
        
    def _clear_write_buffer(self):
        while True:
            frame = self.write_buffer.get()
            if frame is None:
                break
            else:
                self.pbar.update(1)
                self.vid_out.write(frame[:, :, ::-1])
        self.vid_out.release()
    
    def _build_read_buffer(self):
        try:
            for frame in self.videogen:
                self.read_buffer.put(frame)
        except:
            pass
        self.read_buffer.put(None)
    
    def make_inference(self, frame):
        if self.half:
            frame = frame.half()
        return self.model(frame)

class UpscaleMT(threading.Thread):
    def __init__(self, device, model, nt, half, read_buffer, write_buffer, vid_out):
        threading.Thread.__init__(self)
        self.device = device
        self.model = model
        self.nt = nt
        self.half = half
        self.read_buffer = read_buffer
        self.write_buffer = write_buffer
        self.vid_out = vid_out
    
    def inference(self, frame):
        if self.half:
            frame = frame.half()
        with torch.no_grad():
            return self.model(frame)
        
    def process_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        return frame
    
    def run(self):
        while True:
            frame = self.read_buffer.get()
            if frame is None:
                self.write_buffer.put(None)
                break
            self.write_buffer.put(self.process_frame(frame))
        
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, 1, 0, bias=bias
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, 1, 1, 0, bias=bias
        )

    def forward(self, x):
        if "Half" in x.type():  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z

class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):  # conv234结尾有se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2, x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1, x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

class UpCunet2x(nn.Module):  # 完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x):  # 1.7G
        n, c, h0, w0 = x.shape
        # if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), "reflect")  # 需要保证被2整除
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * 2, : w0 * 2]
        return x

class UpCunet3x(nn.Module):  # 完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet3x, self).__init__()
        self.unet1 = UNet1x3(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x):  # 1.7G
        n, c, h0, w0 = x.shape
        # if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 4 + 1) * 4
        pw = ((w0 - 1) // 4 + 1) * 4
        x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect")  # 需要保证被2整除
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * 3, : w0 * 3]
        return x

class UpCunet4x(nn.Module):  # 完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet4x, self).__init__()
        self.unet1 = UNet1(in_channels, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps = nn.PixelShuffle(2)
        self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)

    def forward(self, x):
        n, c, h0, w0 = x.shape
        x00 = x
        # if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (19, 19 + pw - w0, 19, 19 + ph - h0), "reflect")  # 需要保证被2整除
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        x = self.conv_final(x)
        x = F.pad(x, (-1, -1, -1, -1))
        x = self.ps(x)
        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * 4, : w0 * 4]
        x += F.interpolate(x00, scale_factor=4, mode="nearest")
        return x

class pixel_unshuffle(nn.Module):
    def __init__(self, ratio=2):
        super(pixel_unshuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, "x, y, ratio : {}, {}, {}".format(
            x, y, ratio
        )
        return (
            tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio)
            .permute(0, 1, 3, 5, 2, 4)
            .contiguous()
            .view(b, -1, y // ratio, x // ratio)
        )

class UpCunet2x_fast(nn.Module):  # 完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x_fast, self).__init__()
        self.unet1 = UNet1(12, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps = nn.PixelShuffle(2)
        self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)
        self.inv = pixel_unshuffle(2)

    def forward(self, x):
        n, c, h0, w0 = x.shape
        x00 = x
        # if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (38, 38 + pw - w0, 38, 38 + ph - h0), "reflect")  # 需要保证被2整除
        x = self.inv(x)  # +18
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        x = self.conv_final(x)
        # with open(r"C:\Users\liujing\Desktop\log.txt","a+")as f:
        #     f.write("%s"%(str(x.shape)))
        #     f.flush()
        x = F.pad(x, (-1, -1, -1, -1))
        x = self.ps(x)
        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * 2, : w0 * 2]
        x += F.interpolate(x00, scale_factor=2, mode="nearest")
        return x

   