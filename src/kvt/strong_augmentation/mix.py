"""
Ref: https://www.kaggle.com/virajbagal/mixup-cutmix-fmix-visualisations
"""
import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_spm, rand_bbox, rand_bbox_withwh, sample_mask


def mixup(data, target, alpha=1.0, apply_or=False):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    if apply_or:
        lam = 0.5
    else:
        lam = np.random.beta(alpha, alpha)

    data = lam * data + (1 - lam) * shuffled_data

    if apply_or:
        target = target + shuffled_target
        target = target.clamp(min=0, max=1.0)
        shuffled_target = target

    return data, target, shuffled_target, lam, indices


def cutmix(data, target, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    bbx1_p, bby1_p, bbx2_p, bby2_p = rand_bbox_withwh(
        data.size(), bbx2 - bbx1, bby2 - bby1
    )

    data[:, :, bby1_p:bby2_p, bbx1_p:bbx2_p] = data[
        indices, :, bby1:bby2, bbx1:bbx2
    ]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - (
        (bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2])
    )

    return data, target, shuffled_target, lam, indices


def fmix(
    data, target, alpha=1.0, decay_power=3, max_soft=0.0, reformulate=False
):
    device = data.device
    shape = (data.size(2), data.size(3))
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    mask = torch.from_numpy(mask).to(device)
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    x1 = mask * data
    x2 = (1 - mask) * shuffled_data
    return (x1 + x2), target, shuffled_target, lam, indices


def resizemix(data, target, alpha=1.0, mode="bilinear"):
    indices = torch.randperm(data.size(0))
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bby1:bby2, bbx1:bbx2] = F.interpolate(
        data[indices, :, :, :],
        size=(bbx2 - bbx1, bby2 - bby1),
        mode=mode,
        align_corners=False,
    )

    # adjust lambda to exactly match pixel ratio
    lam = 1 - (
        (bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2])
    )

    return data, target, shuffled_target, lam, indices


def snapmix(data, target, model, snapmix_id, beta=1.0):
    device = data.device
    wfmaps, _ = get_spm(data, target, model, snapmix_id)
    bs = data.size(0)
    lam = np.random.beta(beta, beta)
    lam1 = np.random.beta(beta, beta)
    indices = torch.randperm(bs).cuda()
    wfmaps_b = wfmaps[indices, :, :]
    target_b = target[indices]

    size = data.size()

    same_label = target == target_b
    same_label = same_label.squeeze()
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
    bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(size, lam1)

    area = (bby2 - bby1) * (bbx2 - bbx1)
    area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

    if area1 > 0 and area > 0:
        ncont = data[indices, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
        ncont = F.interpolate(
            ncont,
            size=(bbx2 - bbx1, bby2 - bby1),
            mode="bilinear",
            align_corners=True,
        )
        data[:, :, bbx1:bbx2, bby1:bby2] = ncont
        lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (
            wfmaps.sum(2).sum(1) + 1e-8
        )
        lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (
            wfmaps_b.sum(2).sum(1) + 1e-8
        )
        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[-1] * size[-2]))
        lam_a[torch.isnan(lam_a)] = lam
    else:
        lam_a = torch.ones(data.size(0)).to(device)

    return data, target, target_b, lam_a, indices
