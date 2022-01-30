"""Ref: https://bit.ly/3qWEhgq"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def gem1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1.0 / p)


def gem2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
        1.0 / p
    )


class GeMPool1d(nn.Module):
    def __init__(self, p=3, inference_p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.inference_p = inference_p
        self.eps = eps

    def forward(self, x):
        if self.training:
            return gem1d(x, p=self.p, eps=self.eps)
        else:
            return gem1d(x, p=self.inference_p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GeMPool2d(nn.Module):
    def __init__(self, p=3, inference_p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.inference_p = inference_p
        self.eps = eps

    def forward(self, x):
        if self.training:
            return gem2d(x, p=self.p, eps=self.eps)
        else:
            return gem2d(x, p=self.inference_p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor(
        [2, 3, 4, 5, 6, 7]
    )  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)

    b = (max(H, W) - w) / (steps - 1)
    (_, idx) = torch.min(
        torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0
    )  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2
        )  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2
        )  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[
                    :,
                    :,
                    (int(i_) + torch.Tensor(range(wl)).long()).tolist(),
                    :,
                ]
                R = R[
                    :,
                    :,
                    :,
                    (int(j_) + torch.Tensor(range(wl)).long()).tolist(),
                ]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (
                    torch.norm(vt, p=2, dim=1, keepdim=True) + eps
                ).expand_as(vt)
                v += vt

    return v


class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "L=" + "{}".format(self.L) + ")"


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor(
        [2, 3, 4, 5, 6, 7]
    )  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)

    b = (max(H, W) - w) / (steps - 1)
    _, idx = torch.min(
        torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0
    )  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b).int()
            - wl2
        )  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b).int()
            - wl2
        )  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(
                    rpool(x.narrow(2, i_, wl).narrow(3, j_, wl)).unsqueeze(1)
                )

    return torch.cat(vecs, dim=1)


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "eps=" + str(self.eps) + ")"


class Rpool(nn.Module):
    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super().__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = roipool(x, self.rpool, self.L, self.eps)  # size: #im, #reg, D, 1, 1

        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0] * s[1], s[2], s[3], s[4])  # size: #im x #reg, D, 1, 1

        # rvecs -> norm
        o = self.norm(o)

        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4])  # size: #im, #reg, D, 1, 1

        # aggregate regions into a single global vector per image
        if aggregate:
            # rvecs -> sumpool -> norm
            o = self.norm(o.sum(1, keepdim=False))  # size: #im, D, 1, 1

        return o

    def __repr__(self):
        return (
            super(Rpool, self).__repr__()
            + "("
            + "L="
            + "{}".format(self.L)
            + ")"
        )
