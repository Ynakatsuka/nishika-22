import inspect
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb

logger = logging.getLogger(__name__)


class LightningModuleBase(pl.LightningModule):
    calculate_val_loss = True

    def __init__(
        self,
        model=None,
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
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hooks = hooks
        self.strong_transform = strong_transform
        self.strong_transform_p = strong_transform_p
        self.disable_strong_transform_in_first_epochs = (
            disable_strong_transform_in_first_epochs
        )
        self.disable_strong_transform_in_last_epochs = (
            disable_strong_transform_in_last_epochs
        )
        self.enable_overall_evaluation = enable_overall_evaluation
        self.enable_numpy_evaluation = enable_numpy_evaluation
        self.monitor = monitor
        self.main_input_keys = main_input_keys
        self.main_target_keys = main_target_keys

        # batch transform
        if transform is None:
            self.train_transform = None
            self.valid_transform = None
            self.test_transform = None
        elif isinstance(transform, dict):
            keys = transform.keys()
            self.train_transform = (
                transform["train"] if "train" in keys else None
            )
            self.valid_transform = (
                transform["validation"] if "validation" in keys else None
            )
            self.test_transform = transform["test"] if "test" in keys else None
        else:
            raise TypeError

        if (
            hasattr(self.hooks, "")
            and (self.hooks.metric_fn is not None)
            and (not isinstance(self.hooks.metric_fn, dict))
        ):
            raise ValueError("metric_fn must be dict.")

        # for metric learning model
        self.is_metric_learning = (
            "label" in inspect.getfullargspec(model.forward).args
        )

        # for tta
        self.tta_enabled = False

    def enable_tta(self, tta_wrapper=None):
        if (tta_wrapper is not None) and (not self.tta_enabled):
            logger.info(f"Enabled TTA wrapper: {tta_wrapper}")
            tta_wrapper.model = self.model
            self.model = tta_wrapper
            self.tta_enabled = True

    def run_visualization(self, predictions, targets, dataloader=None):
        visualization_result = {}
        if hasattr(self.hooks, "visualization") and (
            self.hooks.visualization is not None
        ):
            funcs = self.hooks.visualization
            if isinstance(funcs, list):
                for func in funcs:
                    r = func(
                        self.model,
                        dataloader,
                        predictions,
                        targets,
                        self.logger,
                    )
                    visualization_result.update(r)
            else:
                r = funcs(
                    self.model, dataloader, predictions, targets, self.logger
                )
                visualization_result.update(r)

            # log
            if isinstance(self.logger, pl.loggers.WandbLogger):
                for key, value in visualization_result.items():
                    if os.path.exists(value):
                        self.logger.experiment.log({key: [wandb.Image(value)]})
            else:
                logger.info(
                    f"Log artifacts is not supported for {type(self.logger)}"
                )

        return visualization_result

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            return self.model(**x, **kwargs)
        return self.model(x, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, aux_x, y, _ = self.extract_inputs_from_batch(batch)
        if self.test_transform is not None:
            x = self.test_transform(x)

        y_hat = self.forward(x, **aux_x)
        y_hat = self.hooks.post_forward_fn(y_hat)
        y_hat = y_hat.detach().cpu().numpy()
        if y is not None:
            y = y.detach().cpu().numpy()
        return {"y_hat": y_hat, "y": y}

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
            y_hat = self.forward(x, **aux_x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
        else:
            y_hat = self.forward(x, **aux_x)
            loss = self.hooks.loss_fn(y_hat, y, **aux_y)
        return loss

    def extract_inputs_from_batch(self, batch):
        if "y" in batch.keys():
            y = batch["y"]  # x, y: dict or tensor
            if isinstance(y, dict):
                main_target_key = list(
                    set(y.keys()) & set(self.main_target_keys)
                )
                assert len(main_target_key) == 1, main_target_key
                main_target_key = main_target_key[0]

                aux_y = {k: v for k, v in y.items() if k != main_target_key}
                y = y[main_target_key]
            else:
                aux_y = {
                    k: v
                    for k, v in batch.items()
                    if (k != "y") and (k[0] == "y")
                }
        else:
            y, aux_y = None, None

        if "x" in batch.keys():
            x = batch["x"]  # x, y: dict or tensor
            if isinstance(x, dict):
                main_input_key = list(set(x.keys()) & set(self.main_input_keys))
                assert len(main_input_key) == 1, main_input_key
                main_input_key = main_input_key[0]

                aux_x = {k: v for k, v in x.items() if k != main_input_key}
                x = x[main_input_key]
            else:
                aux_x = {
                    k: v
                    for k, v in batch.items()
                    if (k != "x") and (k[0] == "x")
                }
        else:
            main_input_key = list(set(batch.keys()) & set(self.main_input_keys))
            assert len(main_input_key) == 1, main_input_key
            main_input_key = main_input_key[0]

            aux_x = {
                k: v
                for k, v in batch.items()
                if k not in main_input_key + self.main_target_keys
            }
            x = batch[main_input_key]

        if (y is not None) and self.is_metric_learning:
            aux_x["label"] = y

        return x, aux_x, y, aux_y

    def training_step(self, batch, batch_nb):
        x, aux_x, y, aux_y = self.extract_inputs_from_batch(batch)

        if self.train_transform is not None:
            x = self.train_transform(x)

        loss = self._calculate_loss(x, y, aux_x, aux_y)

        return {"loss": loss}

    # for dp, ddp
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/4073
    def training_step_end(self, training_step_outputs):
        outputs = {
            name: val.sum() for name, val in training_step_outputs.items()
        }

        self.log_dict(outputs, rank_zero_only=True)

        return outputs

    def _calculate_evaluation(self, y_hat, y, suffix=""):
        outputs = {}
        device = y.device

        if self.enable_numpy_evaluation:
            y_hat = y_hat.cpu().numpy()
            y = y.cpu().numpy()

        for name, func in self.hooks.metric_fn.items():
            result = func(y_hat, y)
            if isinstance(result, tuple):
                # discard detail score metrics
                result, _ = result[0], result[1]

            if isinstance(result, np.ndarray):
                result = torch.tensor(result).to(device)

            outputs[f"val_{name}{suffix}"] = result

        return outputs

    def validation_step(self, batch, batch_nb):
        outputs = {}

        x, aux_x, y, aux_y = self.extract_inputs_from_batch(batch)
        if self.valid_transform is not None:
            x = self.valid_transform(x)

        y_hat = self.forward(x, **aux_x)

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

    def gather_outputs(self, outputs):
        """for DDP. outputs: list of dict that have key: tensor"""
        gathered_outputs = {}

        if len(outputs) == 0:
            return gathered_outputs

        keys = outputs[0].keys()
        for key in keys:
            if len(outputs[0][key].shape):  # tensor
                value = torch.cat([o[key] for o in outputs], dim=0)
            else:  # scaler
                value = torch.stack([o[key] for o in outputs])

            if torch.distributed.is_initialized():
                gathered_value = [
                    torch.zeros_like(value)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_value, value)
                gathered_outputs[key] = torch.cat(gathered_value)
            else:
                gathered_outputs[key] = value

        return gathered_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = {}
        gathered_outputs = self.gather_outputs(outputs)

        for key, value in gathered_outputs.items():
            if key not in ("y", "y_hat"):
                avg_outputs[key] = value.mean()

        if self.enable_overall_evaluation:
            metric_outputs = self._calculate_evaluation(
                gathered_outputs["y_hat"],
                gathered_outputs["y"],
                suffix="_overall",
            )
            avg_outputs.update(metric_outputs)

        if not self.trainer.sanity_checking:
            self.log_dict(
                avg_outputs,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                rank_zero_only=True,
            )

        return avg_outputs

    def configure_optimizers(self):
        opt = {"optimizer": self.optimizer}

        if self.scheduler is not None:
            lr_dict = {
                "scheduler": self.scheduler,
                "monitor": self.monitor,
            }
            opt["lr_scheduler"] = lr_dict

        return opt
