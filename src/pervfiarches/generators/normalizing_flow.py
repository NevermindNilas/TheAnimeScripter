"""Basic layers and funcs for Normalizing Flow Network
"""
from math import log, pi

import numpy as np
import torch

from scipy import linalg as la
from torch import nn
from torch.nn import functional as F

from . import thops


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.logs = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            logs = torch.log(1 / (std + 1e-6) + 1e-6)
            self.logs.data.copy_(logs.data)

    def forward(self, input, reverse=False):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        if reverse:
            return input * (torch.exp(-self.logs)) - self.loc
        else:
            logdet = thops.pixels(input) * torch.sum(self.logs)
            return (torch.exp(self.logs) + 1e-8) * (input + self.loc), logdet


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q[:, :, None, None]
        self.weight = nn.Parameter(weight)

    def forward(self, input, reverse=False):
        _, _, height, width = input.shape

        if reverse:
            return F.conv2d(input, self.weight.squeeze().inverse()[:, :, None, None])

        else:
            logdet = (
                thops.pixels(input)
                * torch.slogdet(self.weight.squeeze().double())[1].float()
            )
            return F.conv2d(input, self.weight), logdet


class InvConv2dLU(nn.Module):
    """invertible conv2d with LU decompose"""

    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s) + 1e-6))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input, reverse=False):
        weight = self.calc_weight()

        if reverse:
            return F.conv2d(input, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
        else:
            logdet = thops.pixels(input) * torch.sum(self.w_s)
            return F.conv2d(input, weight), logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)


class ZeroConv2d(nn.Module):
    """The 3x3 convolution in which weight and bias are initialized with zero.
    The output is then scaled with a positive learnable param.
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out


class condAffineCouplingBN(nn.Module):
    """The conditional affine coupling layer with BN layer."""

    def __init__(self, cin, ccond):
        super().__init__()

        class condAffineNet(nn.Module):
            def __init__(self, cin, ccond, cout) -> None:
                super().__init__()
                self.tail = nn.Sequential(
                    nn.Conv2d(cin + ccond, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 1, 1, 0),
                    nn.ReLU(),
                    ZeroConv2d(64, cout),
                )

            def forward(self, x, cond):
                return self.tail(torch.cat([x, cond], dim=1))

        self.affine = condAffineNet(cin // 2, ccond, cin)
        self.bn = nn.BatchNorm2d(cin // 2, affine=False)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input, cond, reverse=False):
        if not reverse:
            return self.encode(input, cond)
        else:
            return self.decode(input, cond)

    def encode(self, x, cond):
        # split into two folds, 'off' is to generate shift and log_rescale
        # FIXME BN layer is not revertable yet.
        on, off = x.chunk(2, 1)
        shift, log_rescale = self.affine(off, cond).chunk(2, 1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        # affine
        on = on * torch.exp(log_rescale) + shift
        logdet = thops.sum(log_rescale, dim=[1, 2, 3])
        # bn
        if self.training:
            mean = torch.mean(on, dim=(1, 2, 3))
            var = torch.mean((on - mean) ** 2, dim=(1, 2, 3))
        else:
            var = self.bn.running_var
        on = self.bn(on)
        # mean, var = self.bn.running_mean, self.bn.running_var
        print("encode mean: ", torch.mean(mean), "var: ", torch.mean(var))
        logdet = logdet - 0.5 * torch.log(torch.mean(var) + 1e-5)

        output = torch.cat([on, off], 1)
        return output, logdet

    def decode(self, x, cond):
        on, off = x.chunk(2, 1)
        shift, log_rescale = self.affine(off, cond).chunk(2, 1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift

        mean, var = self.bn.running_mean, self.bn.running_var
        print("decode mean: ", torch.mean(mean), "var: ", torch.mean(var))
        mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
        var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
        on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
        on = (on - shift) * torch.exp(-log_rescale)
        x = torch.cat([on, off], 1)
        return x


class condAffineCoupling(nn.Module):
    """The conditional affine coupling layer"""

    def __init__(self, cin, ccond):
        class condAffineNet(nn.Module):
            def __init__(self, cin, ccond, cout) -> None:
                super().__init__()
                self.tail = nn.Sequential(
                    nn.Conv2d(cin + ccond, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 1, 1, 0),
                    nn.ReLU(),
                    ZeroConv2d(64, cout),
                )

            def forward(self, x, cond):
                return self.tail(torch.cat([x, cond], dim=1))

        super().__init__()
        self.affine = condAffineNet(cin // 2, ccond, cin)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input, cond, reverse=False):
        if not reverse:
            return self.encode(input, cond)
        else:
            return self.decode(input, cond)

    def encode(self, x, cond):
        on, off = x.chunk(2, 1)
        shift, log_rescale = self.affine(off, cond).chunk(2, 1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        on = on * torch.exp(log_rescale) + shift
        output = torch.cat([on, off], 1)
        logdet = thops.sum(log_rescale, dim=[1, 2, 3])

        return output, logdet

    def decode(self, x, cond):
        on, off = x.chunk(2, 1)
        shift, log_rescale = self.affine(off, cond).chunk(2, 1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        on = (on - shift) * torch.exp(-log_rescale)
        x = torch.cat([on, off], 1)
        return x


class Flow(nn.Module):
    """Flow step, contains actnorm -> invconv -> condAffineCouple"""

    def __init__(self, in_channel, cond_channel, with_bn=False, train_1x1=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        self.invconv = InvConv2d(in_channel)
        if not train_1x1:
            for p in self.invconv.parameters():
                p.requires_grad = False  # no need to train

        if with_bn:
            self.coupling = condAffineCouplingBN(in_channel, cond_channel)
        else:
            self.coupling = condAffineCoupling(in_channel, cond_channel)

    def forward(self, input, cond, reverse=False):
        if not reverse:
            x, logdet = self.actnorm(input)
            x, det1 = self.invconv(x)
            out, det2 = self.coupling(x, cond)
            logdet = logdet + det1 + det2
            return out, logdet
        else:
            x = self.coupling(input, cond, reverse)
            x = self.invconv(x, reverse)
            out = self.actnorm(x, reverse)
            return out


class Block(nn.Module):
    """Each block contains: squeeze -> flowstep ... flowstep -> split"""

    def __init__(
        self, K, in_channel, cond_channel, split: bool, with_bn: bool, train_1x1: bool
    ):
        super().__init__()

        self.K = K  # number of flow steps

        squeeze_dim = in_channel * 4
        self.split = split

        # layers
        self.actnorm = ActNorm(squeeze_dim)
        self.invconv = InvConv2d(squeeze_dim)
        self.flows = nn.ModuleList(
            [Flow(squeeze_dim, cond_channel, with_bn, train_1x1) for _ in range(self.K)]
        )
        if not train_1x1:
            for p in self.invconv.parameters():
                p.requires_grad = False  # no need to train

        if self.split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input, cond, eps=None, reverse=False):
        if not reverse:
            out, logdet, log_p, z_new = self.encode(input, cond)
            return out, logdet, log_p, z_new
        else:
            out = self.decode(input, cond, eps)
            return out

    def encode(self, input, cond):
        b_size = input.shape[0]
        out = squeeze2d(input, 2)
        out, logdet = self.actnorm(out)
        out, det = self.invconv(out)
        logdet += det

        for flow in self.flows:
            out, det = flow(out, cond)
            logdet = logdet + det
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def decode(self, output, cond, eps=None):
        # eps: noise
        input = output

        if self.split:
            mean, log_sd = self.prior(input).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = torch.cat([output, z], 1)
        else:
            zero = torch.zeros_like(input)
            # zero = F.pad(zero, [1, 1, 1, 1], value=1)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = z

        for flow in self.flows[::-1]:
            input = flow(input, cond, reverse=True)

        input = self.invconv(input, reverse=True)
        input = self.actnorm(input, reverse=True)

        unsqueezed = unsqueeze2d(input)
        return unsqueezed


class CondFlowNet(nn.Module):
    """Conditional Normalizing Flow Network"""

    def __init__(self, cins: list, with_bn: bool, train_1x1: bool, K=4, **kwargs):
        super().__init__()
        self.L = 3  # block number
        # three blocks at three scales, each has 4,4,4 flowsteps.
        self.blocks = nn.ModuleList()
        conf = dict(with_bn=with_bn, train_1x1=train_1x1)
        self.blocks.append(Block(K, 3, cins[0], split=True, **conf))
        self.blocks.append(Block(K, 3 * 2, cins[1], split=True, **conf))
        self.blocks.append(Block(K, 3 * 4, cins[2], split=False, **conf))
        # logger.info(
        #     f"Parameter of condflownet: {sum(p.numel() for p in self.parameters())}"
        # )

    def forward(self, input, conds: list, reverse=False):
        if not reverse:
            log_p_sum, logdet, z_outs = self.encode(input, conds)
            return log_p_sum, logdet, z_outs
        else:
            z_list = input
            out = self.decode(z_list, conds[::-1])
            return out

    def encode(self, input: torch.Tensor, conds: list):
        log_p = 0
        logdet = 0
        z_outs = []

        for i, block in enumerate(self.blocks):
            input, det, logp, z_new = block(input, conds[i])
            z_outs.append(z_new)
            logdet = logdet + det
            log_p = log_p + logp

        return log_p, logdet, z_outs

    def decode(self, z_list: list, conds: list):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block(z_list[-1], conds[0], z_list[-1], reverse=True)
            else:
                input = block(input, conds[i], z_list[-(i + 1)], reverse=True)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 * torch.exp(-2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    B, C, H, W = input.shape
    factor2 = factor**2
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor2, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    B, C, H, W = input.shape
    factor2 = factor**2
    assert C % (factor2) == 0, "{}".format(C)

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x
