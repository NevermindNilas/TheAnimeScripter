import os
import torch
import torch.nn.functional as F
import logging

from src.downloadModels import downloadModels, weightsDir
from realcugan_ncnn_py import Realcugan
from .cugan_arch import UpCunet2x, UpCunet3x, UpCunet4x, UpCunet2x_fast

class Cugan:
    def __init__(
        self,
        upscale_method,
        upscale_factor,
        cugan_kind,
        half,
        width,
        height,
        custom_model,
        nt,
    ):
        self.upscale_method = upscale_method
        self.upscale_factor = upscale_factor
        self.cugan_kind = cugan_kind
        self.half = half
        self.width = width
        self.height = height
        self.custom_model = custom_model
        self.nt = nt

        self.handle_models()

    def handle_models(self):
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")
        model_map = {2: UpCunet2x, 3: UpCunet3x, 4: UpCunet4x}

        if self.custom_model == "":
            if self.upscale_method == "shufflecugan":
                self.model = UpCunet2x_fast(in_channels=3, out_channels=3)
                self.filename = "sudo_shuffle_cugan_9.584.969.pth"
            elif self.upscale_method == "cugan":
                model_path_prefix = "cugan"
                model_path_suffix = "-latest"
                model_path_middle = f"up{self.upscale_factor}x"
                self.model = model_map[self.upscale_factor](
                    in_channels=3, out_channels=3
                )
                self.filename = f"{model_path_prefix}_{model_path_middle}{
                    model_path_suffix}-{self.cugan_kind}.pth"

            if not os.path.exists(os.path.join(weightsDir, "cugan", self.filename)):
                model_path = downloadModels(
                    model=self.upscale_method,
                    cuganKind=self.cugan_kind,
                    upscaleFactor=self.upscale_factor,
                )
            else:
                model_path = os.path.join(weightsDir, "cugan", self.filename)

        else:
            if self.upscale_method == "shufflecugan":
                self.model = UpCunet2x_fast(in_channels=3, out_channels=3)
            else:
                self.model = model_map[self.upscale_factor](
                    in_channels=3, out_channels=3
                )

            logging.info(f"Using custom model: {self.custom_model}")

            model_path = self.custom_model

        self.cuda_available = torch.cuda.is_available()

        if model_path.endswith(".pth"):
            state_dict = torch.load(model_path, map_location="cpu")
            if "params" in state_dict:
                self.model.load_state_dict(state_dict["params"])
            else:
                self.model.load_state_dict(state_dict)
        elif model_path.endswith(".onnx"):
            self.model = torch.onnx.load(model_path)

        self.model = (
            self.model.eval().cuda() if self.cuda_available else self.model.eval()
        )
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

    @torch.inference_mode()
    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.pad_width, 0, self.pad_height])
        return frame

    @torch.inference_mode()
    def run(self, frame):
        with torch.no_grad():
            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .mul_(1 / 255)
            )

            frame = frame.contiguous(memory_format=torch.channels_last)

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
            frame = frame[:, :, : self.upscaled_height, : self.upscaled_width]
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

            if self.cuda_available:
                torch.cuda.synchronize(self.stream[self.current_stream])
                self.current_stream = (self.current_stream + 1) % len(self.stream)

            return frame.cpu().numpy()


class CuganNCNN:
    def __init__(self, num_threads, upscale_factor):
        """
        Barebones for now
        """
        self.num_threads = num_threads
        self.upscale_factor = upscale_factor

        self.realcugan = Realcugan(
            num_threads=self.num_threads,
            gpuid=0,
            tta_mode=False,
            scale=self.upscale_factor,
        )

    def run(self, frame):
        frame = self.realcugan.process_cv2(frame)
        return frame


"""

class cuganDirectML():
    def __init__(self, upscale_method, upscale_factor, cugan_kind, half, width, height, custom_model):
    
        #I don't quite fully comprehend this, but it's a start
        
        self.upscale_method = upscale_method
        self.upscale_factor = upscale_factor
        self.cugan_kind = cugan_kind
        self.half = half
        self.width = width
        self.height = height
        self.custom_model = custom_model

        self.handle_models()

    def handle_models(self):
        if not os.path.exists("weights"):
            os.makedirs("weights")
            
        if self.custom_model == "":
            dir_name = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_name, "weights", "sudo_shuffle_cugan_fp16_op17_clamped_9.584.969.onnx")
            
            if not os.path.exists(model_path):
                url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sudo_shuffle_cugan_fp16_op17_clamped_9.584.969.onnx"
                wget.download(url, out=model_path)
                
            providers = ort.get_all_providers()
            print(providers)
            sess_options = ort.SessionOptions()
            if 'DmlExecutionProvider' in ort.get_all_providers():
                sess_options.add_session_config_entry('session_device_id', '0')
                sess_options.add_session_config_entry(
                    'session_execution_provider', 'DmlExecutionProvider')
            else:
                print("DirectML not available, using CPU instead.")
                sess_options.add_session_config_entry(
                    'session_execution_provider', 'CPUExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, sess_options)
        
        else:
            raise NotImplementedError("Custom models are not supported yet.")
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.pad_width = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.pad_height = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

        self.upscaled_height = self.height * self.upscale_factor
        self.upscaled_width = self.width * self.upscale_factor
    
    def pad_frame(self, frame):
        frame = np.pad(frame, ((0, self.pad_height), (0, self.pad_width), (0, 0)), mode='constant')
        return frame
    
    def run(self, frame):
        frame = frame.astype(np.float16)
        
        frame = frame.transpose(2, 0, 1) / 255.0
        
        if self.pad_width != 0 or self.pad_height != 0:
            frame = self.pad_frame(frame)
        
        frame = frame[np.newaxis, ...]
        
        output = self.session.run([self.output_name], {self.input_name: frame})
        output = output[0]   
        
        output = output[:, :, :self.upscaled_height, :self.upscaled_width]
        
        output = output.squeeze(0).transpose(1, 2, 0) * 255.0
        
        return output
        
    """
