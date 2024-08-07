
import pytorch_lightning as pl
import torch

from .model import ISNetDIS, ISNetGTEncoder, U2NET, U2NET_full2, U2NET_lite2, MODNet \
    , InSPyReNet, InSPyReNet_Res2Net50, InSPyReNet_SwinB


# warnings.filterwarnings("ignore")

net_names = ["isnet_is", "isnet", "isnet_gt", "u2net", "u2netl", "modnet", "inspyrnet_res", "inspyrnet_swin"]

def get_net(net_name, img_size):
    if net_name == "isnet":
        return ISNetDIS()
    elif net_name == "isnet_is":
        return ISNetDIS()
    elif net_name == "isnet_gt":
        return ISNetGTEncoder()
    elif net_name == "u2net":
        return U2NET_full2()
    elif net_name == "u2netl":
        return U2NET_lite2()
    elif net_name == "modnet":
        return MODNet()
    elif net_name == "inspyrnet_res":
        return InSPyReNet_Res2Net50(base_size=img_size)
    elif net_name == "inspyrnet_swin":
        return InSPyReNet_SwinB(base_size=img_size)
    raise NotImplementedError

class AnimeSegmentation(pl.LightningModule):

    def __init__(self, net_name, img_size=None, lr=1e-3):
        super().__init__()
        assert net_name in net_names
        self.img_size = img_size
        self.lr = lr
        self.net = get_net(net_name, img_size)
        if net_name == "isnet_is":
            self.gt_encoder = get_net("isnet_gt", img_size)
            self.gt_encoder.requires_grad_(False)
        else:
            self.gt_encoder = None

    @classmethod
    def try_load(cls, net_name, ckpt_path, map_location=None, img_size=None):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        if "epoch" in state_dict:
            return cls.load_from_checkpoint(ckpt_path, net_name=net_name, img_size=img_size, map_location=map_location)
        else:
            model = cls(net_name, img_size)
            if any([k.startswith("net.") for k, v in state_dict.items()]):
                model.load_state_dict(state_dict)
            else:
                model.net.load_state_dict(state_dict)
            return model

    def forward(self, x):
        if isinstance(self.net, ISNetDIS):
            return self.net(x)[0][0].sigmoid()
        if isinstance(self.net, ISNetGTEncoder):
            return self.net(x)[0][0].sigmoid()
        elif isinstance(self.net, U2NET):
            return self.net(x)[0].sigmoid()
        elif isinstance(self.net, MODNet):
            return self.net(x, True)[2]
        elif isinstance(self.net, InSPyReNet):
            return self.net.forward_inference(x)["pred"]
        raise NotImplementedError
