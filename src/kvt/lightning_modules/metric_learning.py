import numpy as np
import torch
from pytorch_metric_learning.utils import distributed as pml_dist

from .base import LightningModuleBase


class LightningModuleForFaceLoss(LightningModuleBase):
    def _calculate_loss(self, x, y, aux_x, aux_y):
        if (
            (self.strong_transform is not None)
            and (self.trainer.max_epochs is not None)
            and (
                (
                    self.current_epoch
                    <= self.trainer.max_epochs
                    - self.disable_strong_transform_in_last_epochs
                )
                or (
                    self.current_epoch
                    > self.disable_strong_transform_in_first_epochs
                )
            )
            and (np.random.rand() < self.storong_transform_p)
        ):
            x, y_a, y_b, lam, idx = self.strong_transform(x, y)
            y_hat = self.model.full_forward(x, y, **aux_x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
        else:
            y_hat = self.model.full_forward(x, y, **aux_x)
            loss = self.hooks.loss_fn(y_hat, y, **aux_y)
        return loss

    def validation_step(self, batch, batch_nb):
        outputs = {}

        x, aux_x, y, aux_y = self.extract_inputs_from_batch(batch)
        if self.valid_transform is not None:
            x = self.valid_transform(x)

        y_hat = self.model.full_forward(x, y, **aux_x)

        if self.calculate_val_loss:
            val_loss = self.hooks.loss_fn(y_hat, y, **aux_y)
            outputs["val_loss"] = val_loss

        if self.hooks.metric_fn is not None:
            y_hat = self.hooks.post_forward_fn(y_hat)
            y_hat, y = y_hat.detach(), y.detach()

            if self.enable_overall_evaluation:
                outputs["y_hat"] = y_hat
                outputs["y"] = y
            else:
                metric_outputs = self._calculate_evaluation(
                    y_hat, y, suffix="_batch_mean"
                )
                outputs.update(metric_outputs)
        return outputs


class LightningModuleForPyTorchMetricLearning(LightningModuleBase):
    """LightningModule for PyTorch Metric Learning"""

    calculate_val_loss = False

    def __init__(
        self,
        model=None,
        miner=None,
        optimizer=None,
        scheduler=None,
        hooks=None,
        transform=None,
        strong_transform=None,
        strong_transform_p=None,
        disable_strong_transform_in_first_epochs=-1,
        disable_strong_transform_in_last_epochs=5,
        enable_overall_evaluation=False,
        enable_numpy_evaluation=True,
        monitor="val_loss",
        main_input_keys=("x", "input_ids"),
        main_target_keys=("y"),
        enable_pml_dist=False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hooks=hooks,
            transform=transform,
            strong_transform=strong_transform,
            strong_transform_p=strong_transform_p,
            disable_strong_transform_in_first_epochs=disable_strong_transform_in_first_epochs,
            disable_strong_transform_in_last_epochs=disable_strong_transform_in_last_epochs,
            enable_overall_evaluation=enable_overall_evaluation,
            enable_numpy_evaluation=enable_numpy_evaluation,
            monitor=monitor,
            main_input_keys=main_input_keys,
            main_target_keys=main_target_keys,
        )
        self.miner = miner
        if enable_pml_dist and hasattr(self.hooks, "loss_fn"):
            self.hooks.loss_fn = pml_dist.DistributedLossWrapper(
                loss=self.hooks.loss_fn
            )

    def training_step(self, batch, batch_nb):
        # batch: x_anchor, x_pos, x_neg, y_anchor, y_pos, y_neg
        # x_neg, y_neg: list (length: num_negatives)
        x = torch.cat(
            [batch["x_anchor"], batch["x_pos"]] + batch["x_neg"], dim=0
        )
        y = torch.cat(
            [batch["y_anchor"], batch["y_pos"]] + batch["y_neg"], dim=0
        )
        aux_x, aux_y = {}, None

        if self.train_transform is not None:
            x = self.train_transform(x)

        loss = self._calculate_loss(x, y, aux_x, aux_y)

        return {"loss": loss}

    def _calculate_loss(self, x, y, aux_x, aux_y):
        if (y.ndim == 2) and (y.shape[1] == 1):
            y = y.squeeze()

        if (
            (self.strong_transform is not None)
            and (self.trainer.max_epochs is not None)
            and (
                (
                    self.current_epoch
                    <= self.trainer.max_epochs
                    - self.disable_strong_transform_in_last_epochs
                )
                or (
                    self.current_epoch
                    > self.disable_strong_transform_in_first_epochs
                )
            )
            and (np.random.rand() < self.storong_transform_p)
        ):
            x, y_a, y_b, lam, idx = self.strong_transform(x, y)
            embeddings = self.forward(x, **aux_x)
            if self.miner is not None:
                mined_outputs = self.miner(embeddings, y)
                loss = lam * self.hooks.loss_fn(
                    embeddings, y_a, mined_outputs
                ) + (1 - lam) * self.hooks.loss_fn(
                    embeddings, y_b, mined_outputs
                )
            else:
                loss = lam * self.hooks.loss_fn(embeddings, y_a) + (
                    1 - lam
                ) * self.hooks.loss_fn(embeddings, y_b)
        else:
            embeddings = self.forward(x, **aux_x)
            if self.miner is not None:
                mined_outputs = self.miner(embeddings, y)
                loss = self.hooks.loss_fn(embeddings, y, mined_outputs)
            else:
                loss = self.hooks.loss_fn(embeddings, y)
        return loss
