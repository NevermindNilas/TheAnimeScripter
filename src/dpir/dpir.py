import os
import torch
import torch.nn.functional as F


from models.network_unet import UNetRes


class DPIR:
    def __init__(self, half, width, height, nt):
        self.half = half
        self.width = width
        self.height = height
        self.nt = nt

        self.handle_models()

    def handle_models(self):
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")

        if not os.path.exists("weights"):
            os.makedirs("weights")
        
        self.cuda_available = torch.cuda.is_available()
        
        self.model = UNetRes(in_nc=4, out_nc=3, nc=[
                             64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')

        
        self.model = self.model.eval().cuda() if self.cuda_available else self.model.eval()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        if self.cuda_available:
            self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            self.current_stream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

        self.pad_width = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.pad_height = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

        self.upscaled_height = self.height * self.upscale_factor
        self.upscaled_width = self.width * self.upscale_factor

    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.pad_width, 0, self.pad_height])
        return frame

    @torch.inference_mode()
    def run(self, frame):
        with torch.no_grad():
            frame = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).float().mul_(1/255)

            if self.cuda_available:
                torch.cuda.set_stream(self.stream[self.current_stream])
                if self.half:
                    frame = frame.cuda().half()
                else:
                    frame = frame.cuda()
            else:
                frame = frame.cpu()

            if self.pad_width != 0 or self.pad_height != 0:
                frame = self.pad_frame(frame)

            frame = self.model(frame)
            frame = frame[:, :, :self.upscaled_height, :self.upscaled_width]
            frame = frame.squeeze(0).permute(
                1, 2, 0).mul_(255).clamp_(0, 255).byte()

            if self.cuda_available:
                torch.cuda.synchronize(self.stream[self.current_stream])
                self.current_stream = (
                    self.current_stream + 1) % len(self.stream)

            return frame.cpu().numpy()
