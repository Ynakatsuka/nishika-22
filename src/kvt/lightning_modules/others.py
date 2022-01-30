import inspect

import numpy as np

from .base import LightningModuleBase


class LightningModuleNode2Vec(LightningModuleBase):
    def training_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch[0], batch[1]
        loss = self.model.loss(pos_rw, neg_rw)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        outputs = {}
        return outputs


class LightningModuleLightlySSL(LightningModuleBase):
    calculate_val_loss = False

    def forward(self, x):
        return self.model.backbone(x)

    def training_step(self, batch, batch_nb):
        (x0, x1) = batch[0]
        y0, y1 = self.model.forward(x0, x1)

        loss_args = inspect.getfullargspec(self.hooks.loss_fn).args
        if "epoch" in loss_args:
            loss = self.hooks.loss_fn(y0, y1, epoch=self.current_epoch)
        else:
            loss = self.hooks.loss_fn(y0, y1)

        return {"loss": loss}


class LightningModuleManifoldMixUp(LightningModuleBase):
    def _calculate_loss(self, x, y, aux_x, aux_y):
        if (
            (self.strong_transform is not None)
            and (self.trainer.max_epochs is not None)
            and (
                self.current_epoch
                <= self.trainer.max_epochs
                - self.disable_strong_transform_in_last_epochs
            )
            and (np.random.rand() < self.storong_transform_p)
        ):
            _, y_a, y_b, lam, idx = self.strong_transform(x, y)
            y_hat = self.forward(x, mixup_lambda=lam, mixup_index=idx, **aux_x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
        else:
            y_hat = self.forward(x, **aux_x)
            loss = self.hooks.loss_fn(y_hat, y, **aux_y)

        return {"loss": loss}
