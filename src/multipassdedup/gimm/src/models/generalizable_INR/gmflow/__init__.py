from .gmflow import GMFlow
import argparse
import torch


def initialize_GMFlow(model_path="pretrained_ckpt/gmflow_sintel_with_refinement.pkl", device="cuda"):
    """Initializes the RAFT model."""

    model = GMFlow()
    ckpt = torch.load(model_path, map_location="cpu")

    # def convert(param):
    #     return {k.replace("module.", ""): v for k, v in param.items() if "module" in k}
    #
    # ckpt = convert(ckpt)
    model.load_state_dict(ckpt, strict=True)
    print("load gmflow from " + model_path)

    return model
