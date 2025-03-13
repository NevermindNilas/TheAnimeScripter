import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from distillanydepth.modeling.backbones.vit.ViT_DINO import vit_large, vit_giant2, vit_base
from distillanydepth.modeling.backbones.vit.ViT_DINO_reg import vit_large_reg, vit_giant2_reg
from timm.models.vision_transformer import vit_large_patch16_224, vit_large_patch14_224

def compute_depth_expectation(prob, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(prob * depth_values, 1)
    return depth


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class DPTHead(nn.Module):
    def __init__(
        self, 
        mode, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        head_out_channels=1,
        use_clstoken=False,
        num_depth_regressor_anchor=512,
        ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
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


        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        head_features_2 = 32

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, head_out_channels, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )


    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            # import pdb;pdb.set_trace()
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

        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)


        return out
        

class DepthAnything(nn.Module, PyTorchModelHubMixin):
    # @register_to_config
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        head_out_channels=1,
        wo_relu_1_2_channel=False,
        use_bn=False, 
        use_clstoken=False, 
        # localhub=None
        use_registers=False,
        max_depth=1.0,
        mode='disparity',
        num_depth_regressor_anchor=512,
        depth_normalize=(0.1, 150),
        pretrain_type='dinov2', # sam, imagenet
        del_mask_token=True,
    ):
        super(DepthAnything, self).__init__()
        
        self.pretrain_type = pretrain_type
        self.max_depth = max_depth
        self.mode = mode

        assert encoder in ['vits', 'vitb', 'vitl', "vitg"]

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        self.backbone_name = encoder

        if use_registers:
            if encoder == 'vitl':
                checkpoint='data/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth'
                self.backbone = vit_large_reg(checkpoint=checkpoint)
            elif encoder == 'vitg':
                checkpoint='data/weights/dinov2/dinov2_vitg14_reg4_pretrain.pth'
                self.backbone = vit_giant2_reg(checkpoint=checkpoint)
            else:
                raise NotImplementedError
        else:
            if encoder == 'vitl':
                if pretrain_type == 'dinov2':
                    self.backbone = vit_large(checkpoint=None, del_mask_token=del_mask_token)
                else:
                    raise NotImplementedError

                # import pdb;pdb.set_trace()
            elif encoder == 'vitb':
                self.backbone = vit_base(checkpoint=None, del_mask_token=del_mask_token)

            else:
                raise NotImplementedError


        # dim = self.backbone.blocks[0].attn.qkv.in_features
        dim = self.backbone.embed_dim

        self.min_depth = depth_normalize[0]
        self.max_depth = depth_normalize[1]
        self.num_depth_regressor_anchor = num_depth_regressor_anchor
        self.depth_head = DPTHead(mode, dim, features, use_bn, 
        out_channels=out_channels, 
        head_out_channels=head_out_channels,
        use_clstoken=use_clstoken,
        num_depth_regressor_anchor=num_depth_regressor_anchor,
        )


        self.wo_relu_1_2_channel = wo_relu_1_2_channel

    def get_bins(self, bins_num):
        depth_bins_vec = torch.linspace(math.log(self.min_depth), math.log(self.max_depth), bins_num, device='cuda')
        depth_bins_vec = torch.exp(depth_bins_vec)
        return depth_bins_vec

    
    def register_depth_expectation_anchor(self, bins_num, B):
        depth_bins_vec = self.get_bins(bins_num)
        depth_bins_vec = depth_bins_vec.unsqueeze(0).repeat(B, 1)
        self.register_buffer('depth_expectation_anchor', depth_bins_vec, persistent=False)

    
    def forward(self, x):

        bs, _, h, w = x.shape

        if self.pretrain_type=='dinov2':
            features = self.backbone.get_intermediate_layers(x, self.intermediate_layer_idx[self.backbone_name], return_class_token=True)
            patch_h, patch_w = h // 14, w // 14

        elif self.pretrain_type=='imagenet':
            features = self.backbone.get_intermediate_layers(x, self.intermediate_layer_idx[self.backbone_name], return_prefix_tokens=True)
            patch_h, patch_w = h // 16, w // 16
        else:
            raise NotImplementedError

    
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        
        if not self.wo_relu_1_2_channel:
            depth = F.relu(depth)
        else:
            depth[:, 2:] = F.relu(depth[:, 2:])

        return depth, features[3][0]
