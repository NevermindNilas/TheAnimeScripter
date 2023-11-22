import os
import torch
import numpy as np
from tqdm.autonotebook import tqdm

from .network_swinir import SwinIR as SwinIR_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHTS_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0'
WEIGHTS_FOLDER = f'{os.path.dirname(__file__)}/weights'
WEIGHTS_NAME = {
    'classical_sr': '001_classicalSR_DF2K_s64w8_SwinIR-M_x<scale>.pth',
    'lightweight': '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x<scale>.pth',
    'real_sr': '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
}
MODEL_TYPES = ['classical_sr', 'lightweight', 'real_sr']
MODEL_SCALES = {
    'classical_sr': [2, 3, 4, 8],
    'lightweight': [2, 3, 4],
    'real_sr': [4]
}


class SwinIR_SR:
    def __init__(self, model_type: str, scale: int):
        assert model_type in MODEL_TYPES, f'unknown model_type, please choose from: {MODEL_TYPES}'
        assert scale in MODEL_SCALES[model_type], f'unsupported scale for model type {model_type}, please choose from {MODEL_SCALES[model_type]}'

        self.model_type = model_type
        self.scale = scale
        self.model = self._load_model().to(device)

    def _download_model_weights(self):
        """downloads the pre-trained weights from GitHub model zoo."""
        weights_name = WEIGHTS_NAME[self.model_type].replace('<scale>', f'{self.scale}')
        weights_path = f'{WEIGHTS_FOLDER}/{weights_name}'
        if not os.path.exists(weights_path):
            os.system(f'wget {WEIGHTS_URL}/{weights_name} -P {WEIGHTS_FOLDER}')
            print(f'downloading weights to {weights_path}')

        return weights_path

    def _load_pretrained_weights(self):
        weights_path = self._download_model_weights()
        pretrained_weights = torch.load(weights_path)

        if self.model_type == 'classical_sr':
            return pretrained_weights['params']

        elif self.model_type == 'lightweight':
            return pretrained_weights['params']

        elif self.model_type == 'real_sr':
            return pretrained_weights['params_ema']

    def _load_raw_model(self):
        model_hyperparams = {'upscale': self.scale, 'in_chans': 3, 'img_size': 64, 'window_size': 8,
                             'img_range': 1., 'mlp_ratio': 2, 'resi_connection': '1conv'}

        if self.model_type == 'classical_sr':
            model = SwinIR_model(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                                 upsampler='pixelshuffle', **model_hyperparams)

        elif self.model_type == 'lightweight':
            model = SwinIR_model(depths=[6] * 4, embed_dim=60, num_heads=[6] * 4,
                                 upsampler='pixelshuffledirect', **model_hyperparams)

        elif self.model_type == 'real_sr':
            model = SwinIR_model(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                                 upsampler='nearest+conv', **model_hyperparams)

        return model

    def _load_model(self):
        os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
        pretrained_weights = self._load_pretrained_weights()

        model = self._load_raw_model()
        model.load_state_dict(pretrained_weights, strict=True)
        model.eval()
        return model

    @staticmethod
    def _process_img_for_model(img: np.array) -> torch.tensor:
        """cv2 format - np.array HWC-BGR -> model format torch.tensor NCHW-RGB. (from the official repo)"""
        img = img.astype(np.float32) / 255.  # image to HWC-BGR, float32
        img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
        return img

    @staticmethod
    def _pad_img_for_model(img: torch.tensor, window_size=8) -> torch.tensor:
        """pad input image to be a multiple of window_size (pretrained with window_size=8). (from the official repo)"""
        _, _, h_old, w_old = img.size()
        h_new = (h_old // window_size + 1) * window_size
        w_new = (w_old // window_size + 1) * window_size
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_new, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_new]
        return img

    @staticmethod
    def _model_output_to_numpy(output: torch.tensor) -> np.array:
        """convert the output of the SR model to cv2 format np.array. (from the official repo)"""
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output

    def upscale(self, img: np.array) -> np.array:
        """feed the given image to the super resolution model."""
        h_in, w_in, _ = img.shape
        h_out, w_out = h_in * self.scale, w_in * self.scale

        with torch.no_grad():
            img = self._process_img_for_model(img)
            img = self._pad_img_for_model(img)
            img_upscale_torch = self.model(img)[..., :h_out, :w_out]
            img_upscale_numpy = self._model_output_to_numpy(img_upscale_torch)

        return img_upscale_numpy

    def upscale_using_patches(self, img_lq: np.array, slice_dim=256, slice_overlap=0, keep_pbar=False) -> np.array:
        """Apply super resolution on smaller patches and return full image"""
        scale = self.scale
        h, w, c = img_lq.shape
        img_hq = np.zeros((h * scale, w * scale, c))

        slice_step = slice_dim - slice_overlap
        num_patches = int(np.ceil(h / slice_step) * np.ceil(w / slice_step))
        with tqdm(total=num_patches, unit='patch', desc='Performing SR on patches', leave=keep_pbar) as pbar:
            for h_slice in range(0, h, slice_step):
                for w_slice in range(0, w, slice_step):
                    h_max = min(h_slice + slice_dim, h)
                    w_max = min(w_slice + slice_dim, w)
                    pbar.set_postfix(Status=f'[{h_slice:4d}-{h_max:4d}, {w_slice:4d}-{w_max:4d}]')

                    # apply super resolution on slice
                    img_slice = img_lq[h_slice:h_max, w_slice:w_max]
                    img_slice_hq = self.upscale(img_slice)

                    # update full image
                    img_hq[h_slice * scale:h_max * scale, w_slice * scale:w_max * scale] = img_slice_hq
                    pbar.update(1)

            pbar.set_postfix(Status='Done')

        return np.uint8(img_hq)
