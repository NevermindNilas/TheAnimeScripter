# Spandrel

[![PyPI package](https://img.shields.io/badge/pip%20install-spandrel-brightgreen)](https://pypi.org/project/spandrel/)
[![version number](https://img.shields.io/pypi/v/spandrel?color=green&label=version)](https://github.com/chaiNNer-org/spandrel/releases)
[![PyPi Downloads](https://img.shields.io/pypi/dw/spandrel)](https://pypi.org/project/spandrel/#files)
[![Python Version](https://img.shields.io/pypi/pyversions/spandrel)](https://pypi.org/project/spandrel/#files:~:text=Requires%3A%20Python%20%3C3.12%2C%20%3E%3D3.8)
[![Documentation](https://img.shields.io/badge/-documentation-blue)](https://chainner.app/spandrel/)

[![Actions Status](https://github.com/chaiNNer-org/spandrel/workflows/Test/badge.svg)](https://github.com/chaiNNer-org/spandrel/actions)
[![License](https://img.shields.io/github/license/chaiNNer-org/spandrel)](https://github.com/chaiNNer-org/spandrel/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors/chaiNNer-org/spandrel)](https://github.com/chaiNNer-org/spandrel/graphs/contributors)

Spandrel is a library for loading and running pre-trained PyTorch models. It automatically detects the model architecture and hyperparameters from model files, and provides a unified interface for running models.

After seeing many projects extract out [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)'s model support into their own projects, I decided to create this PyPi package for the architecture support and model loading functionality. I'm also hoping that by having a central package anyone can use, the community will be encouraged [to help add support for more models](CONTRIBUTING.md).

This package does not yet have easy inference code, but porting that code is planned as well.

## Installation

Spandrel is available through pip:

```shell
pip install spandrel
```

## Basic Usage

While Spandrel supports different kinds of models, this is how you would run a super resolution model (e.g. ESRGAN, SwinIR, HAT, etc.):

```python
from spandrel import ImageModelDescriptor, ModelLoader
import torch

# load a model from disk
model = ModelLoader().load_from_file(r"path/to/model.pth")

# make sure it's an image to image model
assert isinstance(model, ImageModelDescriptor)

# send it to the GPU and put it in inference mode
model.cuda().eval()

# use the model
def process(image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(image)
```

Note that `model` is a [`ModelDescriptor`](https://chainner.app/spandrel/#ModelDescriptor) object, which is a wrapper around the actual PyTorch model. This wrapper provides a unified interface for running models, and also contains metadata about the model. See [`ImageModelDescriptor`](https://chainner.app/spandrel/spandrel.ImageModelDescriptor.html) for more details about the metadata contained and how to call the model.

> **_NOTE: `ImageModelDescriptor` will NOT convert an image to a tensor for you. It is purely making the forward passes of these models more convenient to use, since the actual forward passes are not always as simple as image in/image out._**

### More architectures

If you are working on a non-commercial open-source project or a private project, you should use `spandrel` and `spandrel_extra_arches` to get everything spandrel has to offer. The `spandrel` package only contains architectures with [permissive and public domain licenses](https://en.wikipedia.org/wiki/Permissive_software_license) (MIT, Apache 2.0, public domain), so it is fit for every use case. Architectures with restrictive licenses (e.g. non-commercial) are implemented in the `spandrel_extra_arches` package.

```python
import spandrel
import spandrel_extra_arches

# add extra architectures before `ModelLoader` is used
spandrel_extra_arches.install()

# load a model from disk
model = spandrel.ModelLoader().load_from_file(r"path/to/model.pth")

... # use model
```

## Supported File Types

Spandrel mainly supports loading `.pth` files for all supported architectures. This is what you will typically find from official repos and community trained models. However, Spandrel also supports loading TorchScript traced models (`.pt`), certain types of `.ckpt` files, and `.safetensors` files for any supported architecture saved in one of these formats.

## Model Architecture Support

> **_NOTE: By its very nature, Spandrel will never be able to support every model architecture. The goal is just to support as many as is realistically possible._**

Spandrel currently supports a limited amount of network architectures. If the architecture you need is not supported, feel free to [request it](https://github.com/chaiNNer-org/spandrel/issues) or try [adding it](CONTRIBUTING.md).

#### Single Image Super Resolution

- [ESRGAN](https://github.com/xinntao/ESRGAN) (RRDBNet)
  - This includes regular [ESRGAN](https://github.com/xinntao/ESRGAN), [ESRGAN+](https://github.com/ncarraz/ESRGANplus), "new-arch ESRGAN" ([RealSR](https://github.com/jixiaozhong/RealSR), [BSRGAN](https://github.com/cszn/BSRGAN)), and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  - Models: [Community ESRGAN](https://openmodeldb.info) | [ESRGAN+](https://drive.google.com/drive/folders/1lNky9afqEP-qdxrAwDFPJ1g0ui4x7Sin) | [BSRGAN](https://github.com/cszn/BSRGAN/tree/main/model_zoo) | [RealSR](https://github.com/jixiaozhong/RealSR#pre-trained-models) | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md)
- [Real-ESRGAN Compact](https://github.com/xinntao/Real-ESRGAN) (SRVGGNet) | [Models](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md)
- [Swift-SRGAN](https://github.com/Koushik0901/Swift-SRGAN) | [Models](https://github.com/Koushik0901/Swift-SRGAN/releases/tag/v0.1)
- [SwinIR](https://github.com/JingyunLiang/SwinIR) | [Models](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)
- [Swin2SR](https://github.com/mv-lab/swin2sr) | [Models](https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1)
- [HAT](https://github.com/XPixelGroup/HAT) | [Models](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0)
- [Omni-SR](https://github.com/Francis0625/Omni-SR) | [Models](https://github.com/Francis0625/Omni-SR#preparation)
- [SRFormer](https://github.com/HVision-NKU/SRFormer) (+) | [Models](https://github.com/HVision-NKU/SRFormer#pretrain-models)
- [DAT](https://github.com/zhengchen1999/DAT) | [Models](https://github.com/zhengchen1999/DAT#testing)
- [FeMaSR](https://github.com/chaofengc/FeMaSR) (+) | [Models](https://github.com/chaofengc/FeMaSR/releases/tag/v0.1-pretrain_models)
- [GRL](https://github.com/ofsoundof/GRL-Image-Restoration) | [Models](https://github.com/ofsoundof/GRL-Image-Restoration/releases/tag/v1.0.0)
- [DITN](https://github.com/yongliuy/DITN) | [Models](https://drive.google.com/drive/folders/1XpHW27H5j2S4IH8t4lccgrgHkIjqrS-X)
- [MM-RealSR](https://github.com/TencentARC/MM-RealSR) | [Models](https://github.com/TencentARC/MM-RealSR/releases/tag/v1.0.0)
- [SPAN](https://github.com/hongyuanyu/SPAN) | [Models](https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view?usp=sharing)
- [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | [Models](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD), [Pro Models](https://drive.google.com/drive/folders/1hfT4WwnNUaS43ErrgXk0J1R5Ik8s5NVo)
- [CRAFT](https://github.com/AVC2-UESTC/CRAFT-SR) | [Models](https://drive.google.com/file/d/13wAmc93BPeBUBQ24zUZOuUpdBFG2aAY5/view?usp=sharing)
- [SAFMN](https://github.com/sunny2109/SAFMN) | [Models](https://drive.google.com/drive/folders/12O_xgwfgc76DsYbiClYnl6ErCDrsi_S9?usp=share_link), [JPEG model](https://github.com/sunny2109/SAFMN/releases/tag/v0.1.1)
- [RGT](https://github.com/zhengchen1999/RGT) | [RGT Models](https://drive.google.com/drive/folders/1zxrr31Kp2D_N9a-OUAPaJEn_yTaSXTfZ?usp=drive_link), [RGT-S Models](https://drive.google.com/drive/folders/1j46WHs1Gvyif1SsZXKy1Y1IrQH0gfIQ1?usp=drive_link)
- [DCTLSA](https://github.com/zengkun301/DCTLSA) | [Models](https://github.com/zengkun301/DCTLSA/tree/main/pretrained)
- [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary) | [Models](https://drive.google.com/drive/folders/1D3BvTS1xBcaU1mp50k3pBzUWb7qjRvmB?usp=sharing)
- [AdaCode](https://github.com/kechunl/AdaCode) | [Models](https://github.com/kechunl/AdaCode/releases/tag/v0-pretrain_models)
- [DRCT](https://github.com/ming053l/DRCT)
- [PLKSR](https://github.com/dslisleedh/PLKSR) and [RealPLKSR](https://github.com/muslll/neosr/blob/master/neosr/archs/realplksr_arch.py) | [Models](https://drive.google.com/drive/u/1/folders/1lIkZ00y9cRQpLU9qmCIB2XtS-2ZoqKq8)
- [SeemoRe](https://github.com/eduardzamfir/seemoredetails) | [Models](https://drive.google.com/drive/folders/15jtvcS4jL_6QqEwaRodEN8FBrqVPrO2u?usp=share_link)
- [MoSR](https://github.com/umzi2/MoSR) | [Models](https://drive.google.com/drive/u/0/folders/1HPy7M4Zzq8oxhdsQ2cnfqy73klmQWp_r) | [RealTime Models](https://drive.google.com/drive/u/1/folders/1Oe88y4qgHRY0dsv_0Q2P0a9hysd-rkvz)
- [SPANPlus](https://github.com/umzi2/spanplus) | [Models](https://drive.google.com/drive/u/0/folders/1UQbK4i5yU-41m5yrjP33V994msMvSEA2?hl=ru)
- [Sebica](https://github.com/idiosyncracies/Sebica/) | [Models](https://github.com/idiosyncracies/Sebica/tree/main/logs/ckpts)

#### Face Restoration

- [GFPGAN](https://github.com/TencentARC/GFPGAN) | [1.2](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth), [1.3](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth), [1.4](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth)
- [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) | [Model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth)

#### Inpainting

- [LaMa](https://github.com/advimman/lama) | [Model](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)
- [MAT](https://github.com/fenglinglwb/MAT) (+) | [Model](https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth)

#### Denoising

- [SCUNet](https://github.com/cszn/SCUNet) | [GAN Model](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth) | [PSNR Model](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth)
- [Uformer](https://github.com/ZhendongWang6/Uformer) | [Denoise SIDD Model](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/Ea7hMP82A0xFlOKPlQnBJy0B9gVP-1MJL75mR4QKBMGc2w?e=iOz0zz) | [Deblur GoPro Model](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/EfCPoTSEKJRAshoE6EAC_3YB7oNkbLUX6AUgWSCwoJe0oA?e=jai90x)
- [KBNet](https://github.com/zhangyi-3/KBNet) | [Models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/EofsV3eVcAxNlrW72JXqzRUBhkM1Mzw50pJ3BHlAyMYnVw?e=MeMB5H)
- [NAFNet](https://github.com/megvii-research/NAFNet) | [Models](https://github.com/megvii-research/NAFNet#results-and-pre-trained-models)
- [Restormer](https://github.com/swz30/Restormer) (+) | [Models](https://github.com/swz30/Restormer/releases/tag/v1.0)
- [FFTformer](https://github.com/kkkls/FFTformer) | [Models](https://github.com/kkkls/FFTformer/releases/tag/pretrain_model)
- [M3SNet](https://github.com/Tombs98/M3SNet) (+) | [Models](https://drive.google.com/drive/folders/1y4BEX7LagtXVO98ZItSbJJl7WWM3gnbD)
- [MPRNet](https://github.com/swz30/MPRNet) (+) | [Deblurring](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing), [Deraining](https://drive.google.com/file/d/1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM/view?usp=sharing), [Denoising](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing)
- [MIRNet2](https://github.com/swz30/MIRNetv2) (+) | [Models](https://github.com/swz30/MIRNetv2/releases/tag/v1.0.0) (SR not supported)
- [DnCNN, FDnCNN](https://github.com/cszn/DPIR) | [Models](https://github.com/cszn/KAIR/releases/tag/v1.0)
- [DRUNet](https://github.com/cszn/DPIR) | [Models](https://github.com/cszn/KAIR/releases/tag/v1.0)
- [IPT](https://github.com/huawei-noah/Pretrained-IPT) | [Models](https://drive.google.com/drive/folders/1MVSdUX0YBExauG0fFz4ANiWTrq9xZEj7?usp=sharing)

#### DeJPEG

- [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | [Models](https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0)

#### Colorization

- [DDColor](https://github.com/piddnad/DDColor) (+) | [Models](https://github.com/piddnad/DDColor/blob/master/MODEL_ZOO.md)

#### Dehazing

- [MixDehazeNet](https://github.com/AmeryXiong/MixDehazeNet) | [Models](https://drive.google.com/drive/folders/1ep6W4H3vNxshYjq71Tb3MzxrXGgaiM6C?usp=drive_link)

#### Low-light Enhancement

- [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer) | [Models](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link)
- [HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet) | [Models](https://github.com/Fediory/HVI-CIDNet/#weights-and-results-)

(All architectures marked with a `+` are only part of `spandrel_extra_arches`.)

## Security

Use `.safetensors` files for guaranteed security.

As you may know, loading `.pth` files [poses a security risk](https://github.com/pytorch/pytorch/issues/52596) due to python's `pickle` module being inherently unsafe and vulnerable to arbitrary code execution (ACE). To mitigate this, Spandrel only allows deserializing certain types of data. This helps to improve security, but it still doesn't fully solve the issue of ACE.

## Used By

Here are some cool projects that use Spandrel:

- [AUTOMATIC1111's SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)
- [imaginAIry](https://github.com/brycedrennan/imaginAIry)
- [dgenerate](https://github.com/Teriks/dgenerate)

## License

This repo is bounded by the MIT license. However, the code of implemented architectures (everything inside an `__arch/` directory) is bound by their original respective licenses (which are included in their respective `__arch/` directories).
