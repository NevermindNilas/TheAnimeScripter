import torch
import torch.nn as nn
from typing import List, Optional, Set, Tuple, Union
from transformers import DPTForDepthEstimation, DPTPreTrainedModel

from transformers.modeling_outputs import DepthEstimatorOutput
from transformers.file_utils import replace_return_docstrings, add_start_docstrings_to_model_forward
from transformers.models.dpt.modeling_dpt import DPTNeck

from diffusers.models.lora import LoRACompatibleConv
from diffusers.utils import USE_PEFT_BACKEND
import torch.nn.functional as F

class DPTDepthEstimationHead(nn.Module):
    """
    Output head head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.projection = None
        features = config.fusion_hidden_size
        if config.add_projection:
            self.projection = nn.Conv2d(features, features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)

        predicted_depth = self.head(hidden_states)

        predicted_depth = predicted_depth.squeeze(dim=1)

        return predicted_depth

class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            # self.norm = RMSNorm(channels, eps, elementwise_affine)
            raise NotImplementedError
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.conv(hidden_states, scale)
                else:
                    hidden_states = self.conv(hidden_states)
            else:
                if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.Conv2d_0(hidden_states, scale)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class DPTForDepthEstimationElu(DPTForDepthEstimation):
    def __init__(self, config):
        super().__init__(config)

        self.head = DPTDepthEstimationHeadElu(config)

class DPTDepthEstimationHeadElu(nn.Module):
    """
    Output head head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.projection = None
        if config.add_projection:
            self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        features = config.fusion_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)

        predicted_depth = self.head(hidden_states) + 1 # range from [0, +inf]

        predicted_depth = predicted_depth.squeeze(dim=1)

        return predicted_depth
    
DPT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

_CONFIG_FOR_DOC = "DPTConfig"

class DPTNeckHeadForUnet(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.backbone = None
        # if config.backbone_config is not None and config.is_hybrid is False:
        #     self.backbone = load_backbone(config)
        # else:
        #     self.dpt = DPTModel(config, add_pooling_layer=False)

        self.feature_upsample_0 = Upsample2D(channels=config.neck_hidden_sizes[0], use_conv=True)
        self.feature_upsample_1 = Upsample2D(channels=config.neck_hidden_sizes[1], use_conv=True)
        self.feature_upsample_2 = Upsample2D(channels=config.neck_hidden_sizes[2], use_conv=True)
        self.feature_upsample_3 = Upsample2D(channels=config.neck_hidden_sizes[3], use_conv=True)

        # Neck
        self.neck = DPTNeck(config)
        self.neck.reassemble_stage = None

        # Depth estimation head
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # if self.backbone is not None:
        #     outputs = self.backbone.forward_with_filtered_kwargs(
        #         pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        #     )
        #     hidden_states = outputs.feature_maps
        # else:
        #     outputs = self.dpt(
        #         pixel_values,
        #         head_mask=head_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=True,  # we need the intermediate hidden states
        #         return_dict=return_dict,
        #     )
        #     hidden_states = outputs.hidden_states if return_dict else outputs[1]
        #     # only keep certain features based on config.backbone_out_indices
        #     # note that the hidden_states also include the initial embeddings
        #     if not self.config.is_hybrid:
        #         hidden_states = [
        #             feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
        #         ]
        #     else:
        #         backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
        #         backbone_hidden_states.extend(
        #             feature
        #             for idx, feature in enumerate(hidden_states[1:])
        #             if idx in self.config.backbone_out_indices[2:]
        #         )

        #         hidden_states = backbone_hidden_states


        assert len(hidden_states) == 4

        # upsample hidden_states for unet
        hidden_states = [getattr(self, "feature_upsample_%s" %i)(hidden_states[i]) for i in range(len(hidden_states))]

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = hidden_states[3].shape
            height *= 8; width *= 8
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=None,
            attentions=None,
        )



class DPTNeckHeadForUnetAfterUpsample(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.backbone = None
        # if config.backbone_config is not None and config.is_hybrid is False:
        #     self.backbone = load_backbone(config)
        # else:
        #     self.dpt = DPTModel(config, add_pooling_layer=False)

        self.feature_upsample_0 = Upsample2D(channels=config.neck_hidden_sizes[0], use_conv=True)
        # self.feature_upsample_1 = Upsample2D(channels=config.neck_hidden_sizes[1], use_conv=True)
        # self.feature_upsample_2 = Upsample2D(channels=config.neck_hidden_sizes[2], use_conv=True)
        # self.feature_upsample_3 = Upsample2D(channels=config.neck_hidden_sizes[3], use_conv=True)

        # Neck
        self.neck = DPTNeck(config)
        self.neck.reassemble_stage = None

        # Depth estimation head
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_depth_only: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # if self.backbone is not None:
        #     outputs = self.backbone.forward_with_filtered_kwargs(
        #         pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        #     )
        #     hidden_states = outputs.feature_maps
        # else:
        #     outputs = self.dpt(
        #         pixel_values,
        #         head_mask=head_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=True,  # we need the intermediate hidden states
        #         return_dict=return_dict,
        #     )
        #     hidden_states = outputs.hidden_states if return_dict else outputs[1]
        #     # only keep certain features based on config.backbone_out_indices
        #     # note that the hidden_states also include the initial embeddings
        #     if not self.config.is_hybrid:
        #         hidden_states = [
        #             feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
        #         ]
        #     else:
        #         backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
        #         backbone_hidden_states.extend(
        #             feature
        #             for idx, feature in enumerate(hidden_states[1:])
        #             if idx in self.config.backbone_out_indices[2:]
        #         )

        #         hidden_states = backbone_hidden_states


        assert len(hidden_states) == 4

        # upsample hidden_states for unet
        # hidden_states = [getattr(self, "feature_upsample_%s" %i)(hidden_states[i]) for i in range(len(hidden_states))]
        hidden_states[0] = self.feature_upsample_0(hidden_states[0])

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = hidden_states[3].shape
            height *= 8; width *= 8
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        # import pdb
        # pdb.set_trace()
        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if return_depth_only:
            return predicted_depth

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=None,
            attentions=None,
        )


class DPTNeckHeadForUnetAfterUpsampleWithVaeDecoderWithNeck(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.backbone = None
        # if config.backbone_config is not None and config.is_hybrid is False:
        #     self.backbone = load_backbone(config)
        # else:
        #     self.dpt = DPTModel(config, add_pooling_layer=False)

        # self.feature_upsample_0 = Upsample2D(channels=config.neck_hidden_sizes[0], use_conv=True)
        # self.feature_upsample_1 = Upsample2D(channels=config.neck_hidden_sizes[1], use_conv=True)
        # self.feature_upsample_2 = Upsample2D(channels=config.neck_hidden_sizes[2], use_conv=True)
        # self.feature_upsample_3 = Upsample2D(channels=config.neck_hidden_sizes[3], use_conv=True)

        # Neck
        self.neck = DPTNeck(config)
        self.neck.reassemble_stage = None

        # Depth estimation head
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # if self.backbone is not None:
        #     outputs = self.backbone.forward_with_filtered_kwargs(
        #         pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        #     )
        #     hidden_states = outputs.feature_maps
        # else:
        #     outputs = self.dpt(
        #         pixel_values,
        #         head_mask=head_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=True,  # we need the intermediate hidden states
        #         return_dict=return_dict,
        #     )
        #     hidden_states = outputs.hidden_states if return_dict else outputs[1]
        #     # only keep certain features based on config.backbone_out_indices
        #     # note that the hidden_states also include the initial embeddings
        #     if not self.config.is_hybrid:
        #         hidden_states = [
        #             feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
        #         ]
        #     else:
        #         backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
        #         backbone_hidden_states.extend(
        #             feature
        #             for idx, feature in enumerate(hidden_states[1:])
        #             if idx in self.config.backbone_out_indices[2:]
        #         )

        #         hidden_states = backbone_hidden_states


        assert len(hidden_states) == 4

        # upsample hidden_states for unet
        # hidden_states = [getattr(self, "feature_upsample_%s" %i)(hidden_states[i]) for i in range(len(hidden_states))]
        # hidden_states[0] = getattr(self, "feature_upsample_0")(hidden_states[0])

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = hidden_states[3].shape
            height *= 8; width *= 8
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        # import pdb
        # pdb.set_trace()
        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=None,
            attentions=None,
        )



class DPTNeckHeadForUnetAfterUpsampleWithVaeDecoderWithoutNeck(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.backbone = None
        # if config.backbone_config is not None and config.is_hybrid is False:
        #     self.backbone = load_backbone(config)
        # else:
        #     self.dpt = DPTModel(config, add_pooling_layer=False)

        # self.feature_upsample_0 = Upsample2D(channels=config.neck_hidden_sizes[0], use_conv=True)
        # self.feature_upsample_1 = Upsample2D(channels=config.neck_hidden_sizes[1], use_conv=True)
        # self.feature_upsample_2 = Upsample2D(channels=config.neck_hidden_sizes[2], use_conv=True)
        # self.feature_upsample_3 = Upsample2D(channels=config.neck_hidden_sizes[3], use_conv=True)

        self.feature_adapt_conv_0 = nn.Conv2d(config.neck_hidden_sizes[0], config.fusion_hidden_size, kernel_size=3, padding=1, bias=False)
        self.feature_adapt_conv_1 = nn.Conv2d(config.neck_hidden_sizes[1], config.fusion_hidden_size, kernel_size=3, padding=1, bias=False)
        self.feature_adapt_conv_2 = nn.Conv2d(config.neck_hidden_sizes[2], config.fusion_hidden_size, kernel_size=3, padding=1, bias=False)
        self.feature_adapt_conv_3 = nn.Conv2d(config.neck_hidden_sizes[3], config.fusion_hidden_size, kernel_size=3, padding=1, bias=False)

        # # Neck
        # self.neck = DPTNeck(config)
        # self.neck.reassemble_stage = None

        # Depth estimation head
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # if self.backbone is not None:
        #     outputs = self.backbone.forward_with_filtered_kwargs(
        #         pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        #     )
        #     hidden_states = outputs.feature_maps
        # else:
        #     outputs = self.dpt(
        #         pixel_values,
        #         head_mask=head_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=True,  # we need the intermediate hidden states
        #         return_dict=return_dict,
        #     )
        #     hidden_states = outputs.hidden_states if return_dict else outputs[1]
        #     # only keep certain features based on config.backbone_out_indices
        #     # note that the hidden_states also include the initial embeddings
        #     if not self.config.is_hybrid:
        #         hidden_states = [
        #             feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
        #         ]
        #     else:
        #         backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
        #         backbone_hidden_states.extend(
        #             feature
        #             for idx, feature in enumerate(hidden_states[1:])
        #             if idx in self.config.backbone_out_indices[2:]
        #         )

        #         hidden_states = backbone_hidden_states


        assert len(hidden_states) == 4

        # upsample hidden_states for unet
        hidden_states = [getattr(self, "feature_adapt_conv_%s" %i)(hidden_states[i]) for i in range(len(hidden_states))]
        # hidden_states[0] = getattr(self, "feature_upsample_0")(hidden_states[0])

        # patch_height, patch_width = None, None
        # if self.config.backbone_config is not None and self.config.is_hybrid is False:
        #     _, _, height, width = hidden_states[3].shape
        #     height *= 8; width *= 8
        #     patch_size = self.config.backbone_config.patch_size
        #     patch_height = height // patch_size
        #     patch_width = width // patch_size

        # # import pdb
        # # pdb.set_trace()
        # hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=None,
            attentions=None,
        )
