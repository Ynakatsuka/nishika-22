import torch
import torchmetrics


def torch_rmse(
    pred, target,
):
    score = torch.sqrt(torchmetrics.functional.mean_squared_error(pred, target))
    return score


def torch_rocauc(
    pred, target,
):
    score = torch.sqrt(torchmetrics.functional.auroc(pred, target.int()))
    return score
