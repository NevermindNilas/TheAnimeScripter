import torch
from typing import Callable


"""
An attempt at converting the lineDarken function from the VapourSynth plugin

Original Source:
    https://github.com/darealshinji/vapoursynth-plugins/blob/master/scripts/fastlinedarken.py


Still not finished yet. The original function is quite complex and I'm still trying to understand it.    
"""


def clamp(minimum: int, x: float, maximum: int) -> int:
    return int(max(minimum, min(round(x), maximum)))


class LineDarken:
    def __init__(self) -> None:
        self.lutRange: range = None
        self.max: int = 0
        self.mid: int = 0

    def mtLut(self, c1: torch.Tensor, expr: Callable[[int], int]) -> torch.Tensor:
        lut = torch.tensor(
            [clamp(0, expr(x), self.max) for x in self.lutRange], dtype=torch.float32
        )
        return lut

    def mtLutxy(
        self, c1: torch.Tensor, c2: torch.Tensor, expr: Callable[[int, int], int]
    ) -> torch.Tensor:
        lut = torch.zeros_like(c1, dtype=torch.float32)

        print(f"c1 dimensions: {c1.dim()}, shape: {c1.shape}")
        print(f"c2 dimensions: {c2.dim()}, shape: {c2.shape}")

        for y in range(c1.size(1)):
            for x in range(c1.size(2)):
                for c in range(c1.size(0)):
                    c1_value = c1[c, y, x].item() if c1.dim() == 3 else c1[y, x].item()
                    c2_value = c2[c, y, x].item() if c2.dim() == 3 else c2[y, x].item()
                    lut[c, y, x] = clamp(0, expr(c1_value, c2_value), self.max)
        return lut

    def expr1(self, x: int, y: int, lumaCap: int, threshold: int) -> int:
        return (
            (x - y if y < lumaCap else x - lumaCap)
            if (y if y < lumaCap else lumaCap) > (x + threshold)
            else 0
        ) + (self.mid - 1)

    def expr2(self, x: int, y: int, lumaCap: int, threshold: int, strf: float) -> int:
        return (
            (
                (x - y if y < lumaCap else x - lumaCap)
                if (y if y < lumaCap else lumaCap) > (x + threshold)
                else 0
            )
            * strf
        ) + x

    def expr3(self, x: int, y: int, strf: float) -> int:
        return x + ((y - (self.mid - 1)) * (strf + 1))

    def lineMaskExpr(self, x: int, thn: float) -> int:
        return ((x - (self.mid - 1)) * thn) + 255

    def lineDarken(
        self,
        frame: torch.Tensor,
        strength: int = 48,
        lumaCap: int = 191,
        threshold: int = 4,
        thinning: int = 1,
    ) -> torch.Tensor:
        strf = float(strength) / 128.0
        thn = float(thinning) / 16.0

        self.max = 2**frame.dtype.itemsize * 8 - 1
        self.mid = self.max // 2 + 1
        self.lutRange = range(self.max + 1)

        if frame.dim() == 3 and frame.size(2) == 3:
            frame = frame.permute(2, 0, 1)

        frame = frame.float()

        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
        elif frame.dim() == 3 and frame.size(0) != 1:
            frame = frame.unsqueeze(0)

        print(f"Shape after ensuring 3D: {frame.shape}")

        exin = torch.nn.functional.max_pool2d(
            frame, kernel_size=3, stride=1, padding=1
        ).squeeze()

        print(f"Shape after first max_pool2d: {exin.shape}")

        diff = self.mtLutxy(frame, exin, self.expr1Wrapper(lumaCap, threshold))

        print(f"Shape of diff: {diff.shape}")

        lineMask = torch.nn.functional.max_pool2d(
            diff.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze()

        print(f"Shape after second max_pool2d: {lineMask.shape}")

        lineMask = self.mtLut(lineMask, self.lineMaskExprWrapper(thn))

        thick = self.mtLutxy(frame, exin, self.expr2Wrapper(lumaCap, threshold, strf))

        print(f"Shape of thick: {thick.shape}")

        if thinning > 0:
            expa = torch.nn.functional.max_pool2d(
                frame, kernel_size=3, stride=1, padding=1
            ).squeeze()

            print(f"Shape after third max_pool2d: {expa.shape}")

            return self.mtLutxy(expa, diff, self.expr3Wrapper(strf))
        else:
            return thick

    def expr1Wrapper(self, lumaCap: int, threshold: int) -> Callable[[int, int], int]:
        def wrappedExpr1(x: int, y: int) -> int:
            return self.expr1(x, y, lumaCap, threshold)

        return wrappedExpr1

    def expr2Wrapper(
        self, lumaCap: int, threshold: int, strf: float
    ) -> Callable[[int, int], int]:
        def wrappedExpr2(x: int, y: int) -> int:
            return self.expr2(x, y, lumaCap, threshold, strf)

        return wrappedExpr2

    def expr3Wrapper(self, strf: float) -> Callable[[int, int], int]:
        def wrappedExpr3(x: int, y: int) -> int:
            return self.expr3(x, y, strf)

        return wrappedExpr3

    def lineMaskExprWrapper(self, thn: float) -> Callable[[int], int]:
        def wrappedLineMaskExpr(x: int) -> int:
            return self.lineMaskExpr(x, thn)

        return wrappedLineMaskExpr
