import torch
import torch.nn.functional as F

from src.vfimamba.config import *


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k and "attn_mask" not in k and "HW" not in k
    }


class Model:
    def __init__(self, local_rank):
        backbonetype, multiscaletype = MODEL_CONFIG["MODEL_TYPE"]
        backbonecfg, multiscalecfg = MODEL_CONFIG["MODEL_ARCH"]
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG["LOGNAME"]
        self.device()

        self.local = LOCAL

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))

    def load_model(self, name=None, rank=0, real=False):
        if rank <= 0:
            if name is None:
                name = self.name
            print(f"loading {name} ckpt")
            self.net.load_state_dict((torch.load(f"ckpt/{name}.pkl")), strict=True)

    def memory_format(self):
        """
        Convert the model to memory format
        """
        self.net = self.net.to(memory_format=torch.channels_last)

    @torch.no_grad()
    def hr_inference(
        self, img0, img1, local, TTA=False, down_scale=1.0, timestep=0.5, fast_TTA=False
    ):
        """
        Infer with down_scale flow
        Noting: return BxCxHxW
        """

        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(
                imgs, scale_factor=down_scale, mode="bilinear", align_corners=False
            )

            flow, mask = self.net.calculate_flow(imgs_down, timestep, local=local)

            flow = F.interpolate(
                flow, scale_factor=1 / down_scale, mode="bilinear", align_corners=False
            ) * (1 / down_scale)
            mask = F.interpolate(
                mask, scale_factor=1 / down_scale, mode="bilinear", align_corners=False
            )

            af = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.0

        if TTA is False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(
        self, img0, img1, local, TTA=False, timestep=0.5, scale=0, fast_TTA=False
    ):
        imgs = torch.cat((img0, img1), 1)
        """
        Noting: return BxCxHxW
        """
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.net(
                input, local=local, timestep=timestep, scale=scale
            )
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.0

        _, _, _, pred = self.net(imgs, timestep=timestep, scale=scale, local=local)
        if TTA is False:
            return pred
        else:
            _, _, _, pred2 = self.net(
                imgs.flip(2).flip(3), timestep=timestep, scale=scale, local=local
            )
            return (pred + pred2.flip(2).flip(3)) / 2
