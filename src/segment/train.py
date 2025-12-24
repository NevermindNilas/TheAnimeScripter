import torch

from .model import (
    ISNetDIS,
    ISNetGTEncoder,
)


# warnings.filterwarnings("ignore")

net_names = [
    "isnet_is",
    "isnet",
    "isnet_gt",
    "modnet",
    "inspyrnet_res",
    "inspyrnet_swin",
]


def get_net(net_name, img_size):
    if net_name == "isnet":
        return ISNetDIS()
    elif net_name == "isnet_is":
        return ISNetDIS()
    elif net_name == "isnet_gt":
        return ISNetGTEncoder()

    raise NotImplementedError


class AnimeSegmentation(torch.nn.Module):
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
            return cls.load_from_checkpoint(
                ckpt_path,
                net_name=net_name,
                img_size=img_size,
                map_location=map_location,
            )
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
        elif isinstance(self.net, InSPyReNet):
            return self.net.forward_inference(x)["pred"]
        raise NotImplementedError
