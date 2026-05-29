import math

from typing_extensions import override

from ...util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.DIS import DIS


class DISArch(Architecture[DIS]):
    def __init__(self) -> None:
        super().__init__(
            id="DIS",
            name="DIS",
            detect=KeyCondition.has_all(
                "head.weight",
                "head_act.weight",
                "fusion.weight",
                "tail.weight",
                KeyCondition.has_any(
                    "body.0.conv1.weight",  # FastResBlock
                    "body.0.dw_conv.depthwise.weight",  # LightBlock (depthwise)
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DIS]:
        in_channels = state_dict["head.weight"].shape[1]
        num_features = state_dict["head.weight"].shape[0]
        out_channels = state_dict["tail.weight"].shape[0]
        num_blocks = get_seq_len(state_dict, "body")
        use_depthwise = "body.0.dw_conv.depthwise.weight" in state_dict

        # Detect scale from upsampler structure
        if "upsampler.0.conv.weight" in state_dict:
            # nn.Sequential of two PixelShuffleUpsamplers -> scale=4
            scale = 4
        elif "upsampler.conv.weight" in state_dict:
            # Single PixelShuffleUpsampler
            # conv output is out_channels * (scale**2), input is num_features
            upsampler_out = state_dict["upsampler.conv.weight"].shape[0]
            scale = math.isqrt(upsampler_out // num_features)
        else:
            # nn.Identity -> scale=1
            scale = 1

        model = DIS(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            scale=scale,
            use_depthwise=use_depthwise,
        )

        if num_blocks >= 12:
            variant_tag = "Balanced"
        elif num_blocks >= 8:
            variant_tag = "Fast"
        else:
            variant_tag = "Tiny"

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[variant_tag, f"{num_features}nf", f"{num_blocks}nb"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
        )


__all__ = ["DISArch", "DIS"]
