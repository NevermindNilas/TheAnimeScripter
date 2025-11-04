import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import warp
from .dynamic_scale import dynamicScale


def get_drm_t(drm, t, precision=1e-3):
    """
    DRM is a tensor with dimensions b, 1, h, w, where for any value x (0 < x < 1).
    We define the timestep of the entire DRM tensor as 0.5, want the entire DRM approach t,
    but during this process, all values in the tensor must maintain their original proportions.

    Example:
        Input:
            drm = [0.1, 0.7, 0.4, 0.2, ...]
            init_t = 0.5
            target_t = 0.8

        Iteration Outputs:
            [0.1900(0.1 + (1 - 0.1) * 0.1), 0.9100, ...]  t = 0.5 + (1 - 0.5) * 0.5 = 0.75\n
            [0.2710(0.19 + (1 - 0.19) * 0.1), 0.9730, ...]  t = 0.75 + (1 - 0.75) * 0.5 = 0.875\n
            [0.2629(0.271 - (0.271 - 0.19) * 0.1), 0.9289, ...]  t = 0.875 - (0.875 - 0.75) * 0.5 = 0.8125\n
            [0.2556(0.2629 - (0.2629 - 0.19) * 0.1), 0.9157, ...]  t = 0.8125 - (0.8125 - 0.75) * 0.5 = 0.78125\n
            [0.2563(0.2556 + (0.2629 - 0.2556) * 0.1), 0.9249, ...]  t = 0.78125 + (0.8125 - 0.78125) * 0.5 = 0.796875\n
            [0.2570(0.2563 + (0.2629 - 0.2563) * 0.1), 0.9277, ...]  t = 0.796875 + (0.8125 - 0.796875) * 0.5 = 0.8046875\n
            ...

        Final Output:
            drm_t = [0.2569, 0.9258, 0.7106, 0.4486, ...]  t = 0.80078125
    """
    dtype = drm.dtype

    _x, b = 0.5, 0.5
    l, r = 0, 1

    # float is suggested for drm calculation to avoid overflow
    x_drm, b_drm = drm.float().clone(), drm.float().clone()
    l_drm, r_drm = x_drm.clone() * 0, x_drm.clone() * 0 + 1

    while abs(_x - t) > precision:
        if _x > t:
            r = _x
            # print(f"{_x} - ({_x} - {l}) * {b}")
            _x = _x - (_x - l) * b

            r_drm = x_drm.clone()
            x_drm = x_drm - (x_drm - l_drm) * b_drm
            # print(_x, x_drm)

        if _x < t:
            l = _x
            # print(f"{_x} + ({r} - {_x}) * {b}")
            _x = _x + (r - _x) * b

            l_drm = x_drm.clone()
            x_drm = x_drm + (r_drm - x_drm) * b_drm
            # print(_x, x_drm)

    return x_drm.to(dtype)


def calc_drm_rife(t, flow10, flow12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    if linear:
        drm_t0_unaligned = drm10 * t * 2
        drm_t1_unaligned = drm12 * t * 2
    else:
        drm_t0_unaligned = get_drm_t(drm10, t)
        drm_t1_unaligned = get_drm_t(drm12, t)

    warp_method = "avg"
    # When using RIFE to generate intermediate frames between I0 and I1,
    # if the input image order is I0, I1, you need to use drm_t_I0_t01.
    # Conversely, if the order is reversed, you should use drm_t_I1_t01.
    # The same rule applies when processing intermediate frames between I1 and I2.

    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    # drm_t0_t01 = warp(drm_t0_unaligned, flow10 * drm_t0_unaligned, None, warp_method)
    drm_t1_t01 = warp(drm_t1_unaligned, flow10 * drm_t1_unaligned, None, warp_method)
    drm_t1_t12 = warp(drm_t0_unaligned, flow12 * drm_t0_unaligned, None, warp_method)
    # drm_t2_t12 = warp(drm_t1_unaligned, flow12 * drm_t1_unaligned, None, warp_method)

    ones_mask = drm10.clone() * 0 + 1

    mask_t1_t01 = warp(ones_mask, flow10 * drm_t1_unaligned, None, warp_method)
    mask_t1_t12 = warp(ones_mask, flow12 * drm_t0_unaligned, None, warp_method)

    gap_t1_t01 = mask_t1_t01 < 0.999
    gap_t1_t12 = mask_t1_t12 < 0.999

    drm_t1_t01[gap_t1_t01] = drm_t1_unaligned[gap_t1_t01]
    drm_t1_t12[gap_t1_t12] = drm_t0_unaligned[gap_t1_t12]

    return {"drm_t1_t01": drm_t1_t01, "drm_t1_t12": drm_t1_t12}


def calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10)
    d12 = distance_calculator(flow12)

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    warp_method = "soft" if (metric10 is not None and metric12 is not None) else "avg"

    if linear:
        drm1t_t01 = drm12 * t * 2
        drm1t_t12 = drm10 * t * 2
        drm0t_t01_unaligned = 1 - drm1t_t01
        drm2t_t12_unaligned = 1 - drm1t_t12
    else:
        drm1t_t01 = get_drm_t(drm12, t)
        drm1t_t12 = get_drm_t(drm10, t)
        drm0t_t01_unaligned = 1 - drm1t_t01
        drm2t_t12_unaligned = 1 - drm1t_t12

    drm0t_t01 = warp(drm0t_t01_unaligned, flow10, metric10, warp_method)
    drm2t_t12 = warp(drm2t_t12_unaligned, flow12, metric12, warp_method)

    # Create a mask with all ones to identify the holes in the warped drm maps
    ones_mask = drm0t_t01.clone() * 0 + 1

    # Warp the ones mask
    warped_ones_mask0t_t01 = warp(ones_mask, flow10, metric10, warp_method)
    warped_ones_mask2t_t12 = warp(ones_mask, flow12, metric12, warp_method)

    # Identify holes in warped drm map
    gap_0t_t01 = warped_ones_mask0t_t01 < 0.999
    gap_2t_t12 = warped_ones_mask2t_t12 < 0.999

    # Fill the holes in the warped drm maps with the inverse of the original drm maps
    drm0t_t01[gap_0t_t01] = drm0t_t01_unaligned[gap_0t_t01]
    drm2t_t12[gap_2t_t12] = drm2t_t12_unaligned[gap_2t_t12]

    return {
        "drm0t_t01": drm0t_t01,
        "drm1t_t01": drm1t_t01,
        "drm1t_t12": drm1t_t12,
        "drm2t_t12": drm2t_t12,
    }


def calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    if linear:
        drm_t0_unaligned = drm10 * t * 2
        drm_t1_unaligned = drm12 * t * 2
    else:
        drm_t0_unaligned = get_drm_t(drm10, t)
        drm_t1_unaligned = get_drm_t(drm12, t)

    warp_method = "soft" if (metric10 is not None and metric12 is not None) else "avg"

    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    drm_t1_t01 = warp(
        drm_t1_unaligned, flow10 * drm_t1_unaligned, metric10, warp_method
    )
    drm_t1_t12 = warp(
        drm_t0_unaligned, flow12 * drm_t0_unaligned, metric12, warp_method
    )

    ones_mask = drm10.clone() * 0 + 1

    mask_t1_t01 = warp(ones_mask, flow10 * drm_t1_unaligned, metric10, warp_method)
    mask_t1_t12 = warp(ones_mask, flow12 * drm_t0_unaligned, metric12, warp_method)

    gap_t1_t01 = mask_t1_t01 < 0.999
    gap_t1_t12 = mask_t1_t12 < 0.999

    drm_t1_t01[gap_t1_t01] = drm_t1_unaligned[gap_t1_t01]
    drm_t1_t12[gap_t1_t12] = drm_t0_unaligned[gap_t1_t12]

    # why use drm0t1 not drm1t0, because rife use backward warp not forward warp.
    return {"drm_t1_t01": drm_t1_t01, "drm_t1_t12": drm_t1_t12}


def distance_calculator(_x):
    dtype = _x.dtype
    u, v = _x[:, 0:1].float(), _x[:, 1:].float()
    return torch.sqrt(u**2 + v**2).to(dtype)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    def __init__(
        self, ensemble=False, dynamicScale=False, scale=1, interpolateFactor=2
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

        self.f0 = None
        self.f1 = None
        self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.dynamicScale = dynamicScale
        self.counter = 1
        self.interpolateFactor = interpolateFactor
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

    def cache(self):
        self.f0.copy_(self.f1, non_blocking=True)

    def cacheReset(self, frame):
        self.f0 = self.encode(frame[:, :3])

    def forward(
        self,
        img0,
        img1,
        timestep=0.5,
    ):
        if self.interpolateFactor == 2:
            if self.f0 is None:
                self.f0 = self.encode(img0[:, :3])
            self.f1 = self.encode(img1[:, :3])
        else:
            if self.counter == self.interpolateFactor:
                self.counter = 1
                if self.f0 is None:
                    self.f0 = self.encode(img0[:, :3])
                self.f1 = self.encode(img1[:, :3])
            else:
                if self.f0 is None or self.f1 is None:
                    self.f0 = self.encode(img0[:, :3])
                    self.f1 = self.encode(img1[:, :3])
            self.counter += 1

        if self.dynamicScale:
            scale = dynamicScale(img0, img1)
            self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

        merged = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        for i in range(5):
            if flow is None:
                flow, mask, feat = self.blocks[i](
                    torch.cat(
                        (img0[:, :3], img1[:, :3], self.f0, self.f1, timestep), 1
                    ),
                    None,
                    scale=self.scale_list[i],
                )

            else:
                wf0 = warp(self.f0, flow[:, :2])
                wf1 = warp(self.f1, flow[:, 2:4])
                fd, m0, feat = self.blocks[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=self.scale_list[i],
                )

                mask = m0
                flow = flow + fd
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, I0, I1, ts):
        output = []
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            else:
                output.append(
                    self.ifnet(
                        torch.cat((I0, I1), 1), timestep=t, scale_list=self.scale_list
                    )[0]
                )

        return output

    def calc_flow(self, a, b, f0=None, f1=None):
        # calc flow at the lowest resolution (significantly faster with almost no quality loss).
        timestep = (a[:, :1].clone() * 0 + 1) * 0.5
        f0 = self.ifnet.encode(a[:, :3]) if f0 is None else f0
        f1 = self.ifnet.encode(b[:, :3]) if f1 is None else f1
        flow, _, _ = self.ifnet.block0(
            torch.cat((a[:, :3], b[:, :3], f0, f1, timestep), 1),
            None,
            scale=self.scale_list[0],
        )

        # get flow flow0.5 -> 0/1
        flow50, flow51 = flow[:, :2], flow[:, 2:]

        warp_method = "avg"

        # qvi
        # flow05, norm2 = fwarp(flow50, flow50)
        # flow05[norm2]...
        # flow05 = -flow05

        flow05 = -1 * warp(flow50, flow50, None, warp_method)
        flow15 = -1 * warp(flow51, flow51, None, warp_method)

        ones_mask = flow05.clone() * 0 + 1
        mask05 = warp(ones_mask, flow50, None, warp_method)
        mask15 = warp(ones_mask, flow51, None, warp_method)

        gap05 = mask05 < 0.999
        gap15 = mask15 < 0.999

        flow05[gap05] = (ones_mask * max(flow05.shape[2], flow05.shape[3]))[gap05]
        flow15[gap15] = (ones_mask * max(flow15.shape[2], flow15.shape[3]))[gap15]

        flow01 = flow05 * 2
        flow10 = flow15 * 2

        return flow01, flow10, f0, f1

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, I0, I1, I2, ts, reuse=None, linear=False):
        flow10, flow01, f1, f0 = self.calc_flow(I1, I0) if not reuse else reuse
        if reuse is None:
            flow12, flow21, f1, f2 = self.calc_flow(I1, I2)
        else:
            flow12, flow21, f1, f2 = self.calc_flow(I1, I2, f0=reuse[2])

        output = list()
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            elif t == 2:
                output.append(I2)
            elif 0 < t < 1:
                t = 1 - t
                drm = calc_drm_rife(t, flow10, flow12, linear)
                inp = torch.cat((I1, I0), 1)
                out = self.ifnet(
                    inp,
                    timestep=drm["drm_t1_t01"],
                    scale_list=self.scale_list,
                    f0=f1,
                    f1=f0,
                )[0]
                output.append(out)
            elif 1 < t < 2:
                t = t - 1
                drm = calc_drm_rife(t, flow10, flow12, linear)
                inp = torch.cat((I1, I2), 1)
                out = self.ifnet(
                    inp,
                    timestep=drm["drm_t1_t12"],
                    scale_list=self.scale_list,
                    f0=f1,
                    f1=f2,
                )[0]
                output.append(out)

        # next flow10, flow01 = reverse(current flow12, flow21)
        return output, (flow21, flow12, f2, f1)
