"""Ref: https://bit.ly/32oEgry"""
import math

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # ------------------ cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # ----------------- convert label to one-hot ---------------------------
        if label.shape == cosine.shape:
            one_hot = label
        else:
            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (
                1 - self.ls_eps
            ) * one_hot + self.ls_eps / self.out_features
        # -------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # ------------------ cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # ----------------- convert label to one-hot ---------------------------
        if label.shape == cosine.shape:
            one_hot = label
        else:
            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ")"
        )


class AdaCos(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        m=0.50,
        ls_eps=0,
        theta_zero=math.pi / 4,
    ):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        if label.shape == logits.shape:
            one_hot = label
        else:
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (
                1 - self.ls_eps
            ) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1,
                torch.exp(self.s * logits),
                torch.zeros_like(logits),
            )
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(
                torch.min(
                    self.theta_zero * torch.ones_like(theta_med), theta_med
                )
            )
        output *= self.s

        return output


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(
            self.LambdaMin,
            self.base * (1 + self.gamma * self.iter) ** (-1 * self.power),
        )

        # ------------------ cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # ----------------- convert label to one-hot ---------------------------
        if label.shape == cos_theta.shape:
            one_hot = label
        else:
            one_hot = torch.zeros(cos_theta.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1), 1)

        # ------------------------- Calculate output ---------------------------
        output = (
            one_hot * (phi_theta - cos_theta) / (1 + self.lamb)
        ) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", m="
            + str(self.m)
            + ")"
        )


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class CurricularFace(nn.Module):
    """Ref: https://github.com/HuangYG123/CurricularFace"""

    def __init__(self, in_features, out_features, m=0.5, s=64.0):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        # if not onehot
        if label.shape[-1] > 1:
            label = torch.argmax(label, dim=1)

        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        target_logit = cos_theta[
            torch.arange(0, embbedings.size(0)), label
        ].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (
            target_logit * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm
        )

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output
