import numpy as np
import pytorch_lightning as pl
import torch


class AutoClip(pl.callbacks.Callback):
    def __init__(self, percentile=0.25):
        self.grad_history = []
        self.percentile = percentile

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        return total_norm

    def on_after_backward(self, trainer, pl_module):
        grad_norm = self.compute_grad_norm(pl_module.model)
        self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, self.percentile)
        torch.nn.utils.clip_grad_norm_(pl_module.model.parameters(), clip_value)
