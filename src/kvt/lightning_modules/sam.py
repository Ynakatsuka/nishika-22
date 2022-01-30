from kvt.utils import disable_bn, enable_bn

from .base import LightningModuleBase


class LightningModuleSAM(LightningModuleBase):
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, aux_x, y, aux_y = self.extract_inputs_from_batch(batch)

        if self.train_transform is not None:
            x = self.train_transform(x)

        optimizer = self.optimizers()

        enable_bn(self.model)
        loss = self._calculate_loss(x, y, aux_x, aux_y)
        with self.model.no_sync():
            self.manual_backward(loss, optimizer)
        optimizer.first_step(zero_grad=True)

        disable_bn(self.model)
        _loss = self._calculate_loss(x, y, aux_x, aux_y)
        self.manual_backward(_loss)
        optimizer.second_step(zero_grad=True)

        return {"loss": loss}
