from typing_extensions import override

from ...util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.figsr import FIGSR

# Mirrors `SampleMods` in .__arch.figsr; index stored in the UniUpsampleV3
# `MetaUpsample` buffer (version 3).
_SAMPLE_MODS = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv",
    "lda",
    "pa_up",
)


class FIGSRArch(Architecture[FIGSR]):
    def __init__(self) -> None:
        super().__init__(
            id="FIGSR",
            name="FIGSR",
            detect=KeyCondition.has_all(
                "in_to_dim.weight",
                "cat_to_dim.weight",
                "shift",
                "scale_norm",
                "upscale.MetaUpsample",
                "gfisr_body_half.0.fc1.weight",
                "gfisr_body_half.0.conv.convhw.weight",
                "gfisr_body_half_2.0.fc1.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FIGSR]:
        dim = state_dict["in_to_dim.weight"].shape[0]
        in_nc = state_dict["in_to_dim.weight"].shape[1]

        # gfisr_body_half holds n_blocks//2 GatedCNNBlocks; gfisr_body_half_2 holds
        # the remaining (n_blocks - n_blocks//2) blocks plus one trailing Conv2d.
        half1 = get_seq_len(state_dict, "gfisr_body_half")
        half2 = get_seq_len(state_dict, "gfisr_body_half_2")
        n_blocks = half1 + half2 - 1

        # hidden = int(expansion_ratio * dim) // 8 * 8  (always a multiple of 8).
        # fc1 produces hidden * 2 channels. Reconstruct an expansion_ratio that the
        # ctor's int()//8*8 maps back to exactly `hidden` (+4 dodges fp floor error).
        hidden = state_dict["gfisr_body_half.0.fc1.weight"].shape[0] // 2
        expansion_ratio = (hidden + 4) / dim

        gc = state_dict["gfisr_body_half.0.conv.convhw.weight"].shape[0]
        square_kernel_size = state_dict["gfisr_body_half.0.conv.convhw.weight"].shape[2]
        band_kernel_size = state_dict["gfisr_body_half.0.conv.convw.weight"].shape[3]

        meta = state_dict["upscale.MetaUpsample"]
        upsampler = _SAMPLE_MODS[int(meta[1])]
        scale = int(meta[2])
        out_nc = int(meta[4])
        mid_dim = int(meta[5])

        model = FIGSR(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            scale=scale,
            out_nc=out_nc,
            upsampler=upsampler,
            mid_dim=mid_dim,
            n_blocks=n_blocks,
            gc=gc,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_blocks}nb"],
            supports_half=True,
            supports_bfloat16=False,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
        )


__all__ = ["FIGSRArch", "FIGSR"]
