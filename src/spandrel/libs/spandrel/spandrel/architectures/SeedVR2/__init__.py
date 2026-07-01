from typing import Any, override

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from ...util import KeyCondition


class SeedVR2(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self._anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        self._state_dict: dict[str, Any] | None = None
        self._loaded = False

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ):
        self._state_dict = state_dict
        return torch.nn.modules.module._IncompatibleKeys([], [])

    @staticmethod
    def _dtype() -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load(self, device: torch.device) -> None:
        if self._loaded:
            return

        from src.model.downloadModels import resolveSeedVR2Sidecar

        if self.variant == "3b":
            from .runtime.models.dit_3b.nadit import NaDiT

            dit = NaDiT(
                vid_in_channels=33,
                vid_out_channels=16,
                vid_dim=2560,
                vid_out_norm="fusedrms",
                txt_in_dim=5120,
                txt_in_norm="fusedln",
                txt_dim=2560,
                emb_dim=15360,
                heads=20,
                head_dim=128,
                expand_ratio=4,
                norm="fusedrms",
                norm_eps=1e-5,
                ada="single",
                qk_bias=False,
                qk_norm="fusedrms",
                patch_size=(1, 2, 2),
                num_layers=32,
                mm_layers=10,
                mlp_type="swiglu",
                msa_type=None,
                block_type=["mmdit_sr"] * 32,
                window=[(4, 3, 3)] * 32,
                window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"]
                * 16,
                rope_type="mmrope3d",
                rope_dim=128,
            )
            model_name = "seedvr2-3b"
        else:
            from .runtime.models.dit_7b.nadit import NaDiT

            dit = NaDiT(
                vid_in_channels=33,
                vid_out_channels=16,
                vid_dim=3072,
                txt_in_dim=5120,
                txt_dim=3072,
                emb_dim=18432,
                heads=24,
                head_dim=128,
                expand_ratio=4,
                norm="fusedrms",
                norm_eps=1e-5,
                ada="single",
                qk_bias=False,
                qk_rope=True,
                qk_norm="fusedrms",
                patch_size=(1, 2, 2),
                num_layers=36,
                shared_mlp=False,
                shared_qkv=False,
                mlp_type="normal",
                block_type=["mmdit_sr"] * 36,
                window=[(4, 3, 3)] * 36,
                window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"]
                * 18,
            )
            model_name = "seedvr2-7b"

        dtype = self._dtype()
        dit.load_state_dict(self._state_dict, strict=False, assign=True)
        self._state_dict = None
        self.dit = dit.eval().to(device=device, dtype=dtype)

        from .runtime.models.video_vae_v3.modules.video_vae import (
            VideoAutoencoderKLWrapper,
        )

        vae = VideoAutoencoderKLWrapper(
            in_channels=3,
            out_channels=3,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            latent_channels=16,
            use_quant_conv=False,
            use_post_quant_conv=False,
            temporal_scale_num=2,
            inflation_mode="pad",
            slicing_sample_min_size=4,
            spatial_downsample_factor=8,
            temporal_downsample_factor=4,
            freeze_encoder=False,
        )
        vae_state = load_file(
            resolveSeedVR2Sidecar(model_name, "ema_vae_fp16.safetensors"),
            device="cpu",
        )
        vae.load_state_dict(vae_state, strict=False, assign=True)
        vae.set_causal_slicing(split_size=4, memory_device="same")
        vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
        self.vae = vae.eval().to(device=device, dtype=dtype)

        self.texts_pos = torch.load(
            resolveSeedVR2Sidecar(model_name, "pos_emb.pt"),
            map_location=device,
            weights_only=True,
        ).to(device=device, dtype=dtype)
        self.texts_neg = torch.load(
            resolveSeedVR2Sidecar(model_name, "neg_emb.pt"),
            map_location=device,
            weights_only=True,
        ).to(device=device, dtype=dtype)
        self._loaded = True

    @staticmethod
    def _cfg(pos, neg, scale: float, rescale: float = 0.0):
        pred = neg + scale * (pos - neg)
        if rescale:
            factor = pos.std() / pred.std()
            pred = rescale * pred * factor + (1 - rescale) * pred
        return pred

    def _timesteps(self, shape: Tensor, steps: int = 50) -> Tensor:
        ts = torch.arange(
            1.0, 0.0, -1.0 / steps, device=shape.device, dtype=self._dtype()
        ) * 1000.0
        frames = (shape[:, 0] - 1) * 4 + 1
        heights = shape[:, 1] * 8
        widths = shape[:, 2] * 8

        def line(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            return lambda x: m * x + (y1 - m * x1)

        img_shift = line(256 * 256, 1.0, 1024 * 1024, 3.2)
        vid_shift = line(256 * 256 * 37, 1.0, 1280 * 720 * 145, 5.0)
        shift = torch.where(frames > 1, vid_shift(heights * widths * frames), img_shift(heights * widths))
        ts = ts[:, None] / 1000.0
        ts = shift * ts / (1 + (shift - 1) * ts)
        return (ts * 1000.0).flatten()

    def forward(self, image: Tensor) -> Tensor:
        if not image.is_cuda:
            raise RuntimeError("SeedVR2 currently supports CUDA only.")
        self._load(image.device)

        dtype = self._dtype()
        _, _, h, w = image.shape
        pad_h = (-h) % 16
        pad_w = (-w) % 16
        x = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        x = x.to(dtype=dtype).mul(2).sub(1).unsqueeze(2)

        scale = 0.9152
        with torch.no_grad():
            encoded = self.vae.encode(x).latent
            latent = encoded[0].permute(1, 2, 0).unsqueeze(0).contiguous() * scale
            torch.manual_seed(666)
            noise = torch.randn_like(latent)
            condition = torch.zeros(
                (*latent.shape[:-1], latent.shape[-1] + 1),
                device=image.device,
                dtype=dtype,
            )
            condition[..., :-1] = latent
            condition[..., -1:] = 1.0
            vid_shape = torch.tensor([noise.shape[:-1]], device=image.device)
            txt_shape = torch.tensor([[self.texts_pos.shape[0]]], device=image.device)
            txt_pos = self.texts_pos.reshape(-1, self.texts_pos.shape[-1])
            txt_neg = self.texts_neg.reshape(-1, self.texts_neg.shape[-1])

            t = self._timesteps(vid_shape, steps=1)[0]
            pos = self.dit(
                vid=torch.cat([noise, condition], dim=-1).reshape(-1, 33),
                txt=txt_pos,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=t.repeat(1),
            ).vid_sample.reshape_as(noise)
            neg = self.dit(
                vid=torch.cat([noise, condition], dim=-1).reshape(-1, 33),
                txt=txt_neg,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=t.repeat(1),
            ).vid_sample.reshape_as(noise)
            sample = noise - (t / 1000.0).view(1, 1, 1, 1) * self._cfg(pos, neg, 1.0)

            decoded = sample.permute(3, 0, 1, 2).unsqueeze(0) / scale
            decoded = self.vae.decode(decoded).sample
            return decoded.squeeze(2).add(1).mul(0.5).clamp(0, 1)[..., :h, :w]


class SeedVR2Arch(Architecture[SeedVR2]):
    def __init__(self) -> None:
        super().__init__(
            id="SeedVR2",
        detect=KeyCondition.has_all(
            "vid_in.proj.weight",
            "blocks.0.attn.proj_qkv.vid.weight",
            "emb_in.proj_in.weight",
        ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SeedVR2]:
        dim = state_dict["vid_in.proj.weight"].shape[0]
        variant = "7b" if dim >= 3000 else "3b"
        model = SeedVR2(variant)

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[variant],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=3,
            output_channels=3,
        )


__all__ = ["SeedVR2Arch", "SeedVR2"]
