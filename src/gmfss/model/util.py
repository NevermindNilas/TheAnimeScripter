import torch.nn as nn


# Fused-kernel drop-ins for the original hand-rolled implementations.
# State-dict keys match (both expose `weight` of shape (num_parameters,)).
MyPixelShuffle = nn.PixelShuffle


class MyPReLU(nn.PReLU):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__(num_parameters=num_parameters, init=init)
