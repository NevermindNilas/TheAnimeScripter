from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.smosr import SMoSR

# Mirrors `SampleMods` in .__arch.smosr; index stored in the UniUpsampleV4_light
# `MetaUpsample` buffer (version 254).
_SAMPLE_MODS = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "pa_up",
)


class SMoSRArch(Architecture[SMoSR]):
    def __init__(self) -> None:
        super().__init__(
            id="SMoSR",
            name="SMoSR",
            detect=KeyCondition.has_all(
                "short.weight",
                "blocks_1.0.short.weight",
                "end_block.1.eval_conv.weight",
                "upsampler.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SMoSR]:
        in_ch = state_dict["short.weight"].shape[1]
        dim = state_dict["blocks_1.0.short.weight"].shape[0]

        # blocks_2 is an nn.Sequential of `n_mb` SMB blocks.
        n_mb = get_seq_len(state_dict, "blocks_2")

        # rep=True swaps every DOConv2d for a ConvNXC, which carries a `.sk` skip
        # DOConv2d. The skip's depthwise-orthogonal weight `.sk.W` therefore exists
        # only in the reparameterizable variant.
        rep = any(k.endswith(".sk.W") for k in state_dict)

        meta = state_dict["upsampler.MetaUpsample"]
        upsampler = _SAMPLE_MODS[int(meta[1])]
        scale = int(meta[2])
        out_ch = int(meta[4])
        upsampler_mid_dim = int(meta[5])

        model = SMoSR(
            in_ch=in_ch,
            out_ch=out_ch,
            dim=dim,
            scale=scale,
            rep=rep,
            n_mb=n_mb,
            upsampler=upsampler,
            upsampler_mid_dim=upsampler_mid_dim,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_mb}nmb", "rep" if rep else "norep"],
            supports_half=True,
            supports_bfloat16=False,
            scale=scale,
            input_channels=in_ch,
            output_channels=out_ch,
        )


__all__ = ["SMoSRArch", "SMoSR"]
