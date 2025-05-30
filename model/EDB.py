from thop import profile, clever_format

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
import pywt
from typing import Sequence, Tuple, Union, List
from einops import rearrange
import torch.nn.functional as F


def _as_wavelet(wavelet):
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def construct_2d_filt(lo, hi) -> torch.Tensor:
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    # filt = filt.unsqueeze(1)
    return filt


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


# global count
# count = 1
class LWN(nn.Module):
    def __init__(self, dim, wavelet='haar'):
        super(LWN, self).__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)
        self.conv1 = nn.Conv2d(dim * 4, dim * 6, 1)
        self.conv2 = nn.Conv2d(dim * 6, dim * 6, 7, padding=3, groups=dim * 6)  # dw
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim * 6, dim * 4, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        ya, (yh, yv, yd) = self.wavedec(x)
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv1(dec_x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)], None)
        return y


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel = torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class WaveletBlock(nn.Module):
    def __init__(self, c, DW_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = LWN(c // 2, wavelet='haar')
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=c,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=c // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.sig = nn.Sigmoid()

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv4(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv5(x1)
        x2 = self.sig(self.wavelet_block1(x2))
        x = x1 * x2
        x = self.conv6(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = x * self.sca(x)
        return y + x * self.gamma


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same", stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            WaveletBlock(out_channels),
            nn.BatchNorm2d(out_channels),
            EMA(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        x = self.down_sample(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, c: int) -> None:
        super().__init__()
        n = 0 if c == 0 else int(math.log(c, 2))

        self.upsample = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels, in_channels, 2, 2) for i in range(n)]
        )
        self.conv_3 = nn.Conv2d(in_channels, out_channels, 3, padding="same", stride=1)

    def forward(self, x):
        for layer in self.upsample:
            x = layer(x)
        return self.conv_3(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n: int = 16) -> None:
        super().__init__()
        self.in_conv = DoubleConv(in_channels, n)
        self.down_1 = DownSample(n, 2 * n)
        self.down_2 = DownSample(2 * n, 4 * n)
        self.down_3 = DownSample(4 * n, 8 * n)
        self.down_4 = DownSample(8 * n, 16 * n)
        self.up_1024_512 = UpSample(16 * n, 8 * n, 2)
        self.up_512_64 = UpSample(8 * n, n, 8)
        self.up_512_128 = UpSample(8 * n, 2 * n, 4)
        self.up_512_256 = UpSample(8 * n, 4 * n, 2)
        self.up_512_512 = UpSample(8 * n, 8 * n, 0)
        self.up_256_64 = UpSample(4 * n, n, 4)
        self.up_256_128 = UpSample(4 * n, 2 * n, 2)
        self.up_256_256 = UpSample(4 * n, 4 * n, 0)
        self.up_128_64 = UpSample(2 * n, n, 2)
        self.up_128_128 = UpSample(2 * n, 2 * n, 0)
        self.up_64_64 = UpSample(n, n, 0)
        self.dec_4 = DoubleConv(2 * 8 * n, 8 * n)
        self.dec_3 = DoubleConv(3 * 4 * n, 4 * n)
        self.dec_2 = DoubleConv(4 * 2 * n, 2 * n)
        self.dec_1 = DoubleConv(5 * n, n)
        self.out_conv = OutConv(n, out_channels)

    def forward(self, x):
        x = self.in_conv(x)
        x_enc_1 = self.down_1(x)
        x_enc_2 = self.down_2(x_enc_1)
        x_enc_3 = self.down_3(x_enc_2)
        x_enc_4 = self.down_4(x_enc_3)
        x_up_1 = self.up_1024_512(x_enc_4)
        x_dec_4 = self.dec_4(torch.cat([x_up_1, self.up_512_512(x_enc_3)], dim=1))
        x_up_2 = self.up_512_256(x_dec_4)
        x_dec_3 = self.dec_3(torch.cat([x_up_2,self.up_512_256(x_enc_3), self.up_256_256(x_enc_2)], dim=1))
        x_up_3 = self.up_256_128(x_dec_3)
        x_dec_2 = self.dec_2(torch.cat([x_up_3,self.up_512_128(x_enc_3),self.up_256_128(x_enc_2),self.up_128_128(x_enc_1)], dim=1))
        x_up_4 = self.up_128_64(x_dec_2)
        x_dec_1 = self.dec_1(torch.cat([x_up_4,self.up_512_64(x_enc_3),self.up_256_64(x_enc_2),self.up_128_64(x_enc_1),self.up_64_64(x)], dim=1))
        return torch.sigmoid(self.out_conv(x_dec_1))


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    flops, params = profile(model, inputs=(x,))
    print(f"：{flops}, ：{params}")
    flops, params = clever_format(flops, "%.3f"), clever_format(params, "%.3f")
    print(f"：{flops}, ：{params}")
