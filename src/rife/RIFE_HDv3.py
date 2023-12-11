import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet_HDv3 import *
from .loss import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 4.8
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(
                self.flownet, device_ids=[local_rank], output_device=local_rank
            )

    def find_flownet(self):
        self.abs_path = os.path.abspath(__file__)
        self.directory = os.path.dirname(self.abs_path)
        self.flownet_path = os.path.join(self.directory, "flownet.pkl")
        
    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        self.find_flownet()
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(
                    #convert(torch.load("{}/flownet.pkl".format(path))), False
                    convert(torch.load(self.flownet_path)), False
                )
            else:
                self.flownet.load_state_dict(
                    convert(
                        torch.load(self.flownet_path, map_location="cpu")
                    ),
                    False,
                )

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]

