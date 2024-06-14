import torch


class InputPadder:
    """Pads images such that dimensions are divisible by factor"""

    def __init__(self, size, divide=8, mode="center"):
        self.ht, self.wd = size[-2:]
        pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
        pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
        if mode == "center":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [0, pad_wd, 0, pad_ht]

    def _pad_(self, x):
        return torch.nn.functional.pad(x, self._pad, mode="constant")

    def pad(self, *inputs):
        return [self._pad_(x) for x in inputs]

    def _unpad_(self, x):
        return x[
            ...,
            self._pad[2] : self.ht + self._pad[2],
            self._pad[0] : self.wd + self._pad[0],
        ]

    def unpad(self, *inputs):
        return [self._unpad_(x) for x in inputs]


def build_flow_estimator(name, device="cuda", checkpoint=None):
    if name.lower() == "raft":
        import argparse

        from .raft.raft import RAFT

        args = argparse.Namespace(
            mixed_precision=True, alternate_corr=False, small=False
        )
        model = RAFT(args)

        if checkpoint is None:
            ckpt = "checkpoints/RAFT/raft-sintel.pth"
        else:
            ckpt = checkpoint # Path to the checkpoint .pth, modified from original arch for better comp

        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in torch.load(ckpt, map_location=device).items()}
        )
        model.to(device).eval()

        @torch.no_grad()
        def infer(I1, I2):
            I1 = I1.to(device) * 255.0
            I2 = I2.to(device) * 255.0
            padder = InputPadder(I1.shape, 8)
            I1, I2 = padder.pad(I1, I2)
            fflow = model(I1, I2, bidirection=False, iters=12)
            bflow = model(I2, I1, bidirection=False, iters=12)
            return padder.unpad(fflow, bflow)

    if name.lower() == "raft_small":
        import argparse

        from .raft.raft import RAFT

        args = argparse.Namespace(
            mixed_precision=True, alternate_corr=False, small=True
        )
        model = RAFT(args)
        ckpt = "checkpoints/RAFT/raft-small.pth"
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in torch.load(ckpt).items()}
        )
        model.to(device).eval()

        @torch.no_grad()
        def infer(I1, I2):
            I1 = I1.to(device) * 255.0
            I2 = I2.to(device) * 255.0
            padder = InputPadder(I1.shape, 8)
            I1, I2 = padder.pad(I1, I2)
            fflow = model(I1, I2, bidirection=False, iters=12)
            bflow = model(I2, I1, bidirection=False, iters=12)
            return padder.unpad(fflow, bflow)

    if name.lower() == "gma":
        import argparse

        from .gma.network import RAFTGMA

        args = argparse.Namespace(
            mixed_precision=True,
            num_heads=1,
            position_only=False,
            position_and_content=False,
        )
        model = RAFTGMA(args)
        ckpt = "checkpoints/GMA/gma-sintel.pth"
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in torch.load(ckpt).items()}
        )
        model.to(device).eval()

        @torch.no_grad()
        def infer(I1, I2):
            I1 = I1.to(device) * 255.0
            I2 = I2.to(device) * 255.0
            padder = InputPadder(I1.shape, 8)
            I1, I2 = padder.pad(I1, I2)
            _, fflow = model(I1, I2, test_mode=True, iters=20)
            _, bflow = model(I2, I1, test_mode=True, iters=20)
            return padder.unpad(fflow, bflow)

    if name.lower() == "gmflow":
        from .gmflow.gmflow import GMFlow

        model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )
        ckpt = "checkpoints/GMFlow/gmflow_sintel-0c07dcb3.pth"
        model.load_state_dict(torch.load(ckpt)["model"])
        model.to(device).eval()

        @torch.no_grad()
        def infer(I1, I2):
            I1 = I1.to(device) * 255.0
            I2 = I2.to(device) * 255.0
            padder = InputPadder(I1.shape, 16)
            I1, I2 = padder.pad(I1, I2)
            results_dict = model(
                I1,
                I2,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=False,
            )
            fflow = results_dict["flow_preds"][-1]
            results_dict = model(
                I2,
                I1,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=False,
            )
            bflow = results_dict["flow_preds"][-1]

            fflow, bflow = padder.unpad(fflow, bflow)
            return fflow, bflow

    return model, infer
