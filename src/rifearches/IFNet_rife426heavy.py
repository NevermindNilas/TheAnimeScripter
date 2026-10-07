import torch.nn as nn

from .IFNet_rife425 import Head, IFBlock
from .IFNet_rife425 import IFNet as IFNet425


class Head16(Head):
    """4.25's Head with a 16-channel output instead of 4."""

    def __init__(self):
        super().__init__()
        self.cnn3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)


class IFNet(IFNet425):
    """RIFE 4.26-heavy: 4.25's block widths with a 16-channel feature encoder.

    Not to be confused with 4.25-heavy, which keeps the 4-channel encoder and
    doubles every block width instead. The 4.25 forward concatenates whatever
    the encoder returns, so only the encoder and the block input widths differ.
    """

    def __init__(
        self, ensemble=False, dynamicScale=False, scale=1, interpolateFactor=2
    ):
        super().__init__(ensemble, dynamicScale, scale, interpolateFactor)
        self.block0 = IFBlock(7 + 32, c=192)
        self.block1 = IFBlock(8 + 4 + 32 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 32 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 32 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 32 + 8, c=32)
        self.encode = Head16()
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
