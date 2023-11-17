import itertools
import numpy as np
import torch
from .rife_arch import IFNet
#from .download import check_and_download


# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class RIFE:
    def __init__(self, scale, fastmode, ensemble, model_version, fp16, arch_ver, model_path):
        self.scale = scale
        self.fastmode = fastmode
        self.ensemble = ensemble
        self.model_version = model_version
        self.fp16 = fp16
        self.cache = False
        self.amount_input_img = 2
        self.arch_ver = arch_ver

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        #check_and_download(model_path)
        self.model = IFNet(arch_ver=arch_ver)
        self.model.load_state_dict(torch.load(model_path), False)

        self.model.eval().cuda()

        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.model.half()

    def execute(self, I0, I1, timestep):
        scale_list = [8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]

        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.inference_mode():
            middle = self.model(
                I0,
                I1,
                scale_list=scale_list,
                fastmode=self.fastmode,
                ensemble=self.ensemble,
                timestep=timestep,
            )

        middle = middle.detach().squeeze(0).cpu().numpy()
        return middle
