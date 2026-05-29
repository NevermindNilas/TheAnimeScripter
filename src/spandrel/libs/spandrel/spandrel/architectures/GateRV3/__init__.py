from typing_extensions import override

from ...util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.gaterv3 import GateRV3

# Mirrors `SampleMods` in .__arch.gaterv3; index stored in the UniUpsample
# `MetaUpsample` buffer (version 1).
_SAMPLE_MODS = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
)


class GateRV3Arch(Architecture[GateRV3]):
    def __init__(self) -> None:
        super().__init__(
            id="GateRV3",
            name="GateRV3",
            detect=KeyCondition.has_all(
                "in_to_dim.weight",
                "sisr_cat_conv.weight",
                "span_block0.c1_r.sk.weight",
                "gater_encode.0.gated.0.gamma0",
                "latent.0.norm.scale",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GateRV3]:
        dim = state_dict["in_to_dim.weight"].shape[0]
        in_ch = state_dict["in_to_dim.weight"].shape[1]

        num_enc = get_seq_len(state_dict, "gater_encode")
        enc_blocks = tuple(
            get_seq_len(state_dict, f"gater_encode.{i}.gated") for i in range(num_enc)
        )
        num_dec = get_seq_len(state_dict, "decode")
        dec_blocks = tuple(
            get_seq_len(state_dict, f"decode.{i}.gated") for i in range(num_dec)
        )
        num_latent = get_seq_len(state_dict, "latent")

        # scale != 1 builds a UniUpsample (registers MetaUpsample); scale == 1 is a
        # plain Conv2d with no buffer.
        if "dim_to_in.MetaUpsample" in state_dict:
            meta = state_dict["dim_to_in.MetaUpsample"]
            upsample = _SAMPLE_MODS[int(meta[1])]
            scale = int(meta[2])
            upsample_mid_dim = int(meta[5])
        else:
            upsample = "pixelshuffledirect"
            scale = 1
            upsample_mid_dim = 32

        model = GateRV3(
            in_ch=in_ch,
            dim=dim,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_latent=num_latent,
            scale=scale,
            upsample=upsample,
            upsample_mid_dim=upsample_mid_dim,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{num_latent}nl"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_ch,
            output_channels=in_ch,
        )


__all__ = ["GateRV3Arch", "GateRV3"]
