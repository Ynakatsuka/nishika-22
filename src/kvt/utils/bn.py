import torch.nn as nn
from kvt.models.layers import Identity


def judge_bn(layer):
    if (
        isinstance(layer, nn.BatchNorm1d)
        or isinstance(layer, nn.BatchNorm2d)
        or isinstance(layer, nn.BatchNorm3d)
        or isinstance(layer, nn.SyncBatchNorm)
    ):
        return True
    else:
        return False


def replace_bn(model):
    for name, layer in model.named_children():
        if judge_bn(layer):
            setattr(getattr(model, name), Identity())
        return model


def disable_bn(model):
    def _disable(module):
        if (
            judge_bn(module)
            and hasattr(module, "backup_momentum")
            and hasattr(module, "momentum")
        ):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_bn(model):
    def _enable(module):
        if (
            judge_bn(module)
            and hasattr(module, "backup_momentum")
            and hasattr(module, "momentum")
        ):
            module.momentum = module.backup_momentum

    model.apply(_enable)
