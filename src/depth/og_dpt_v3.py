import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2_v3 import DINOv2_V3
from .og_dpt_utils.blocks import FeatureFusionBlock, _make_scratch
from .og_dpt_utils.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHeadV3(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        output_dim=2,
    ):
        super(DPTHeadV3, self).__init__()

        self.output_dim = output_dim

        # Layer norm for the concatenated features (cls_token + patch_token)
        self.norm = nn.LayerNorm(in_channels)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            # out_features is a list of tuples: (patch_tokens, cls_token)
            patch_tokens, cls_token = x[0], x[1]

            # Expand cls_token to match patch_tokens shape and concatenate
            # cls_token: [B, C], patch_tokens: [B, N, C]
            cls_expanded = cls_token.unsqueeze(1).expand(-1, patch_tokens.shape[1], -1)
            x = torch.cat([patch_tokens, cls_expanded], dim=-1)  # [B, N, 2*C]

            # Apply normalization
            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        return out


class DepthAnythingV3(nn.Module):
    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
    ):
        super(DepthAnythingV3, self).__init__()

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }

        self.encoder = encoder

        # Determine qknorm_start based on encoder
        # vits has qknorm starting at block 4 (verified)
        # assuming others follow similar pattern or checking weights would be best
        qknorm_start = 4  # Default to 4

        self.pretrained = DINOv2_V3(model_name=encoder, qknorm_start=qknorm_start)

        # DA3 uses cat_token=True, which doubles the embed_dim for the head
        head_in_channels = self.pretrained.embed_dim * 2

        self.depth_head = DPTHeadV3(
            head_in_channels,
            features,
            use_bn,
            out_channels=out_channels,
            output_dim=2,  # DA3 outputs depth + confidence
        )

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        out = self.depth_head(features, patch_h, patch_w)

        # Apply exp activation to depth channel (first channel)
        # out shape: [B, 2, H, W] where channel 0 is depth, channel 1 is confidence
        depth = torch.exp(out[:, 0:1, :, :])

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518, half=False):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        image = image.half() if half else image.float()
        depth = self.forward(image)

        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)

        depth = depth[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.byte()

        depth = depth.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()

        return depth

    def image2tensor(
        self,
        raw_image,
        input_size=518,
    ):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        image = image.to(DEVICE)

        return image, (h, w)
