from .Rife425_v3 import IFNet as IFNet425


class IFNet(IFNet425):
    """RIFE 4.26-heavy for TensorRT: 4.25's block widths, 16-channel encoder.

    The f0 input and f1 output of the exported engine are 16 channels wide, so
    RifeTensorRT sizes its feature bindings to match.
    """

    encodeChannels = 16
