from Marigold.marigold import MarigoldPipeline
import numpy as np
import torch
from PIL import Image

pipe: MarigoldPipeline = None

def apply_marigold(img: np.ndarray, checkpoint: str = 'prs-eth/marigold-lcm-v1-0', denoise_steps: int = 4, ensemble_size: int = 5, half_precision: bool = False, processing_res: int = 768, seed: int = 0, device: str = 'cuda', **kwargs):
    global pipe

    if pipe is None:

        if half_precision:
            dtype = torch.float16
            variant = "fp16"
        else:
            dtype = torch.float32
            variant = None

        pipe = MarigoldPipeline.from_pretrained(
            checkpoint, variant=variant, torch_dtype=dtype
        )

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass  # run without xformers

        pipe = pipe.to(device=device)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    with torch.no_grad():
        pipe_out = pipe(
            img,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=True,
            batch_size=0,
            color_map="Spectral",
            show_progress_bar=True,
            resample_method="bilinear",
            seed=seed,
        )

        depth_pred: np.ndarray = pipe_out.depth_np

    return depth_pred