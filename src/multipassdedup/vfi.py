import torch
from models.IFNet_HDv3 import IFNet
from models.gimm.src.utils.setup import single_setup
from models.gimm.src.models import create_model
from models.model_pg104.GMFSS import Model as GMFSS
import argparse
from models.utils.tools import *


class VFI:
    def __init__(self, model_type='rife', weights='weights', scale=1.0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if model_type == 'rife':
            model = IFNet()
            model.load_state_dict(convert(torch.load(f'{weights}/rife48.pkl')))
        elif model_type == 'gmfss':
            model = GMFSS()
            model.load_model(f'{weights}/train_log_pg104', -1)
        else:
            args = argparse.Namespace(
                model_config=r"models/gimm/configs/gimmvfi/gimmvfi_r_arb.yaml",
                load_path=f"{weights}/gimmvfi_r_arb_lpips.pt",
                ds_factor=scale,
                eval=True,
                seed=0
            )
            config = single_setup(args)
            model, _ = create_model(config.arch)

            # Checkpoint loading
            if "ours" in args.load_path:
                ckpt = torch.load(args.load_path, map_location="cpu")

                def convert_gimm(param):
                    return {
                        k.replace("module.feature_bone", "frame_encoder"): v
                        for k, v in param.items()
                        if "feature_bone" in k
                    }

                ckpt = convert_gimm(ckpt)
                model.load_state_dict(ckpt, strict=False)
            else:
                ckpt = torch.load(args.load_path, map_location="cpu")
                model.load_state_dict(ckpt["state_dict"], strict=False)

        model.eval()
        if model_type == 'gmfss':
            model.device()
        else:
            model.to(device)

        self.model = model
        self.model_type = model_type
        base_pads = {
            'gimm': 64,
            'rife': 64,
            'gmfss': 128,
        }
        self.pad_size = base_pads[model_type] / scale
        self.device = device
        self.saved_result = {}
        self.scale = scale

    @torch.inference_mode()
    def gen_ts_frame(self, x, y, ts):
        _outputs = list()
        head = [x] if 0 in ts else []
        tail = [y] if 1 in ts else []
        if 0 in ts:
            ts.remove(0)
        if 1 in ts:
            ts.remove(1)
        with torch.autocast(str(self.device)):
            _reuse_things = self.model.reuse(x, y, self.scale) if self.model_type == 'gmfss' else None
            if self.model_type in ['rife', 'gmfss']:
                for t in ts:
                    if self.model_type == 'rife':
                        scale_list = [8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
                        _out = self.model(torch.cat((x, y), dim=1), t, scale_list)
                    elif self.model_type == 'gmfss':
                        _out = self.model.inference(x, y, _reuse_things, t)
                    _outputs.append(_out)
            elif self.model_type == 'gimm':
                xs = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim=2).to(
                    self.device, non_blocking=True
                )
                self.model.zero_grad()
                coord_inputs = [
                    (
                        self.model.sample_coord_input(
                            xs.shape[0],
                            xs.shape[-2:],
                            [t],
                            device=xs.device,
                            upsample_ratio=self.scale,
                        ),
                        None,
                    )
                    for t in ts
                ]
                timesteps = [
                    t * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                    for t in ts
                ]
                all_outputs = self.model(xs, coord_inputs, t=timesteps, ds_factor=self.scale)

                _outputs = all_outputs["imgt_pred"]

            _outputs = head + _outputs + tail

            return _outputs
