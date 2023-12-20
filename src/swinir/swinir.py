import os
import requests
import torch
import numpy as np

from .swinir_arch import SwinIR as SwinIR_model


class Swinir():
    def __init__(self, upscale_factor, half):
        self.upscale_factor = upscale_factor
        self.half = half

        self.handle_models()

    def handle_models(self):
        """
        Only going to support 2x for now due to the memory requirements of the model

        There's a probability that it won't even work for most users without tiling, 
        I will look into that later
        """

        model_hyperparams = {'upscale': 2, 'in_chans': 3, 'img_size': 64, 'window_size': 8,
                             'img_range': 1., 'mlp_ratio': 2, 'resi_connection': '1conv'}

        self.model = SwinIR_model(depths=[6] * 4, embed_dim=60, num_heads=[6] * 4,
                                  upsampler='pixelshuffledirect', **model_hyperparams)

        dir_name = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(dir_name, "weights")

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        filename = "2x_Bubble_AnimeScale_SwinIR_Small_v1.pth"
        if not os.path.exists(os.path.join(weights_dir, "2x_Bubble_AnimeScale_SwinIR_Small_v1.pth")):
            print(f"Downloading SWINIR model...")
            url = "https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_SwinIR_Small_v1/2x_Bubble_AnimeScale_SwinIR_Small_v1.pth"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(weights_dir, filename), "wb") as file:
                    file.write(response.content)
                    
        model_path = os.path.join(weights_dir, filename)
        
        pretrained_weights = torch.load(os.path.join(model_path), map_location="cpu")
        pretrained_weights = pretrained_weights['params']
                
        self.model.load_state_dict(pretrained_weights, strict=True)
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
    def inference(self, frame):
        if self.half:
            frame = frame.half()
        with torch.no_grad():
            return self.model(frame)
        
    def pad_frame(self, frame):
        frame = torch.cat([frame, torch.flip(frame, [2])], 2)[:, :, :self.h, :]
        frame = torch.cat([frame, torch.flip(frame, [3])], 3)[:, :, :, :self.w]
        return frame     
    
    @torch.inference_mode
    def run(self, frame, frame_size):
        frame = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).div_(255)
        
        if frame_size[0] % 8 != 0 or frame_size[1] % 8 != 0:
            frame = self.pad_frame(frame)           
            
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.cpu().numpy()
    