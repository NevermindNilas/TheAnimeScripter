
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer

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
    raise NotImplemented

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

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
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetDIS):
            ds, dfs = self.net(images)
            loss_args = [ds, dfs, labels]
            if self.gt_encoder is not None:
                fs = self.gt_encoder(labels)[1]
                loss_args.append(fs)
        elif isinstance(self.net, ISNetGTEncoder):
            ds = self.net(labels)[0]
            loss_args = [ds, labels]
        elif isinstance(self.net, U2NET):
            ds = self.net(images)
            loss_args = [ds, labels]
        elif isinstance(self.net, MODNet):
            trimaps = batch["trimap"]
            pred_semantic, pred_detail, pred_matte = self.net(images, False)
            loss_args = [pred_semantic, pred_detail, pred_matte, images, trimaps, labels]
        elif isinstance(self.net, InSPyReNet):
            out = self.net.forward_train(images, labels)
            loss_args = out
        else:
            raise NotImplemented

        loss0, loss = self.net.compute_loss(loss_args)
        self.log_dict({"train/loss": loss, "train/loss_tar": loss0})
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetGTEncoder):
            preds = self.forward(labels)
        else:
            preds = self.forward(images)
        pre, rec, f1, = f1_torch(preds.nan_to_num(nan=0, posinf=1, neginf=0), labels)
        mae_m = F.l1_loss(preds, labels, reduction="mean")
        pre_m = pre.mean()
        rec_m = rec.mean()
        f1_m = f1.mean()
        self.log_dict({"val/precision": pre_m, "val/recall": rec_m, "val/f1": f1_m, "val/mae": mae_m}, sync_dist=True)


def get_gt_encoder(train_dataloader, val_dataloader, opt):
    print("---start train ground truth encoder---")
    gt_encoder = AnimeSegmentation("isnet_gt")
    trainer = Trainer(precision=32 if opt.fp32 else 16, accelerator=opt.accelerator,
                      devices=opt.devices, max_epochs=opt.gt_epoch,
                      benchmark=opt.benchmark, accumulate_grad_batches=opt.acc_step,
                      check_val_every_n_epoch=opt.val_epoch, log_every_n_steps=opt.log_step,
                      strategy="ddp_find_unused_parameters_false" if opt.devices > 1 else None,
                      )
    trainer.fit(gt_encoder, train_dataloader, val_dataloader)
    return gt_encoder.net


