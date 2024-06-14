import torch

from .flow_estimators import build_flow_estimator
from .generators import build_generator_arch


def get_z(heat: float, img_size: tuple, batch: int, device: str):
    def calc_z_shapes(img_size, n_levels):
        h, w = img_size
        z_shapes = []
        channel = 3

        for _ in range(n_levels - 1):
            h //= 2
            w //= 2
            channel *= 2
            z_shapes.append((channel, h, w))
        h //= 2
        w //= 2
        z_shapes.append((channel * 4, h, w))
        return z_shapes

    z_list = []
    z_shapes = calc_z_shapes(img_size, 3)
    for z in z_shapes:
        z_new = torch.randn(batch, *z, device=device) * heat
        z_list.append(z_new)
    return z_list


class Pipeline_infer(torch.nn.Module):
    def __init__(self, flownet: str, generator: str, model_file: str, flowCheckpoint: str = None, device = "cuda"):
        super().__init__()
        if flownet is None:
            self.flownet = None
        else:
            self.flownet, self.compute_flow = build_flow_estimator(flownet, device=device, checkpoint=flowCheckpoint)
            self.flownet.to(device).eval()

        self.netG = build_generator_arch(generator)
        state_dict = {
            k.replace("module.", ""): v for k, v in torch.load(model_file, map_location=device).items()
        }
        self.netG.load_state_dict(state_dict)
        self.netG.to(device).eval()

    def forward(self, img0, img1, heat=0.3, time=0.5, flows=None):
        if isinstance(heat, float):
            zs = get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        else:
            zs = heat

        fflow, bflow = flows if self.flownet is None else self.compute_flow(img0, img1)
        conds = [img0, img1, fflow, bflow]
        pred, _ = self.netG(zs=zs, inps=conds, time=time, code="decode")
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def inference_rand_noise(self, img0, img1, heat=0.7, time=0.5, flows=None):
        zs = get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        fflow, bflow = flows if flows is not None else self.compute_flow(img0, img1)

        conds = [img0, img1, fflow, bflow]
        pred, _ = self.netG(zs=zs, inps=conds, time=time, code="decode")
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def inference_best_noise(self, img0, img1, gt, time=0.5, flows=None):
        fflow, bflow = flows if flows is not None else self.compute_flow(img0, img1)
        conds = [img0, img1, fflow, bflow]
        _, pred, _ = self.netG(gt=gt, inps=conds, code="encode_decode", time=time)
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def inference_spec_noise(self, img0, img1, zs: list, time=0.5, flows=None):
        fflow, bflow = flows if flows is not None else self.compute_flow(img0, img1)
        conds = [img0, img1, fflow, bflow]
        pred, _ = self.netG(zs=zs, inps=conds, code="decode", time=time)
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def generate_masks(self, img0, img1, time=0.5):
        zs = get_z(0.4, img0.shape[-2:], img0.shape[0], img0.device)
        fflow, bflow = self.compute_flow(img0, img1)

        conds = [img0, img1, fflow, bflow]
        pred, smasks = self.netG(zs=zs, inps=conds, time=time, code="decode")
        return torch.clamp(pred, 0.0, 1.0), smasks
