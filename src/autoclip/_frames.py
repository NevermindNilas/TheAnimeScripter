"""Frame dtype shared by the autoclip readers."""

import torch


def eightBit(frames):
    """Decoded RGB frames as uint8, the input every autoclip model is written for.

    nelux decodes a >8-bit source (Hi10P, HEVC Main10, ProRes) to uint16:
    TransNetV2's forward asserts uint8 (and its CUDA graph's uint8 input
    wrapped the values), and MaxxVIT's preprocessing divides by 255. Torch has
    no uint16 shift, hence the int32 hop; nelux's uint16 white (65283..65535)
    lands on 255.
    """
    if frames.dtype == torch.uint8:
        return frames
    return (frames.to(torch.int32) >> 8).to(torch.uint8)
