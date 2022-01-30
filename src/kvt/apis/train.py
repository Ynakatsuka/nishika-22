import glob
import logging
import os
import subprocess

import kvt.utils
import pytorch_lightning as pl
import torch
from kvt.builder import (
    build_batch_transform,
    build_callbacks,
    build_dataloaders,
    build_hooks,
    build_lightning_module,
    build_logger,
    build_loss,
    build_metrics,
    build_miner,
    build_model,
    build_optimizer,
    build_plugins,
    build_scheduler,
    build_strong_transform,
    build_tta_wrapper,
)
from kvt.utils import check_attr, concatenate, save_predictions

try:
    from pytorch_lightning.callbacks import RichProgressBar
except ImportError:
    RichProgressBar = None
try:
    from transformers.debug_utils import DebugUnderflowOverflow
except ImportError:
    DebugUnderflowOverflow = None

log = logging.getLogger(__name__)


def run(config):
    # ------------------------------
    # Parameters
    # ------------------------------
    is_ddp = hasattr(config.trainer.trainer, "accelerator") and (
        config.trainer.trainer.accelerator in ("ddp", "ddp2")
    )

    # ------------------------------
    # Building
    # ------------------------------
    # build hooks
    loss_fn = build_loss(config)
    metric_fn = build_metrics(config)
    hooks = build_hooks(config)
    hooks.update({"loss_fn": loss_fn, "metric_fn": metric_fn})

    # build model
    model = build_model(config)

    # build callbacks
    callbacks = build_callbacks(config)

    # build miner
    miner = build_miner(config)

    # build logger
    logger = build_logger(config)

    # debug overflow
    if (
        check_attr(config.trainer, "enable_debug_overflow")
        and hasattr(config.trainer.trainer, "precision")
        and config.trainer.trainer.precision == 16
    ):
        DebugUnderflowOverflow(model)

    # build optimizer
    optimizer = build_optimizer(config, model=model, params=model.parameters())

    # build scheduler
    scheduler = build_scheduler(config, optimizer=optimizer)

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build strong transform (e.g., mixup)
    strong_transform, strong_transform_p = build_strong_transform(config)

    # build batch transform (torchvision or kornia)
    transform = build_batch_transform(config)

    # build tta wrapper
    tta_wrapper = build_tta_wrapper(config)

    # add progress bar callback
    if RichProgressBar is not None:
        callbacks.append(RichProgressBar())

    # build lightning module
    lightning_module = build_lightning_module(
        config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        hooks=hooks,
        transform=transform,
        strong_transform=strong_transform,
        strong_transform_p=strong_transform_p,
        miner=miner,
    )

    # build plugins
    plugins = build_plugins(config, is_ddp)

    # ------------------------------
    # Logging
    # ------------------------------
    # debug
    if config.debug:
        logger = None
        config.trainer.trainer.max_epochs = None
        config.trainer.trainer.max_steps = 10

    # logging for wandb or mlflow
    if hasattr(logger, "log_hyperparams"):
        for k, v in config.trainer.items():
            if k not in ("metrics", "inference"):
                logger.log_hyperparams(params=v)
        logger.log_hyperparams(params=config.augmentation)
        logger.log_hyperparams(params=config.dataset)
        logger.log_hyperparams(params=config.fold)
        logger.log_hyperparams(params=config.lightning_module)
        logger.log_hyperparams(params=config.model)
        logger.log_hyperparams(params=config.optimizer)
        if hasattr(config, "comment"):
            logger.log_hyperparams(params={"comment": config.comment})

    log.info(f"strong_transform: {strong_transform}")
    log.info(f"batch transform: {transform}")

    # ------------------------------
    # Checkpoint
    # ------------------------------
    # best model path
    dir_path = os.path.join(config.save_dir, "models", config.experiment_name)
    filename = f"fold_{config.trainer.idx_fold}_best.ckpt"
    best_model_path = os.path.join(dir_path, filename)

    # auto resume_from_checkpoint (load latest checkpoint in directory)
    if config.trainer.auto_resume_from_checkpoint:
        ckpts = os.path.join(dir_path, f"fold_{config.trainer.idx_fold}_*.ckpt")
        ckpts = glob.glob(ckpts)
        if len(ckpts):
            latest_ckpt = max(ckpts, key=os.path.getctime)
            log.info(f"Auto Loading: {latest_ckpt}")
            config.trainer.trainer.resume_from_checkpoint = latest_ckpt

    # ------------------------------
    # Training
    # ------------------------------
    # train loop
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        plugins=plugins,
        **config.trainer.trainer,
    )
    if not config.trainer.skip_training:
        trainer.fit(
            lightning_module,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"],
        )

    is_master = (
        (not is_ddp)
        or (config.trainer.skip_training)
        or (torch.distributed.get_rank() == 0)
    )

    # checkpoint
    path = trainer.checkpoint_callback.best_model_path
    if is_master and path:
        log.info(f"Best model: {path}")
        # copy best model
        subprocess.run(f"cp {path} {best_model_path}", shell=True)
    # if there is no best_model_path
    # e.g., no validation dataloader
    elif path is None:
        trainer.save_checkpoint(best_model_path)

    # ------------------------------
    # Checkpoint
    # ------------------------------
    # log best model
    if is_master and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(params={"original_best_model_path": path})
        logger.log_hyperparams(params={"best_model_path": best_model_path})

    # load best checkpoint
    if os.path.exists(best_model_path):
        log.info(f"Loading best model: {best_model_path}")
        state_dict = torch.load(best_model_path)["state_dict"]
        state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)
        lightning_module.model.load_state_dict(state_dict)
    else:
        log.info(f"Loaded Best model {best_model_path} does not exist.")

    # ------------------------------
    # Evaluate
    # ------------------------------
    if check_attr(config.trainer, "enable_final_evaluation") and (
        tta_wrapper is not None
    ):
        log.info("Enabled TTA wrapper")
        lightning_module.enable_tta(tta_wrapper)
        trainer.validate(
            model=lightning_module,
            dataloaders=dataloaders["validation"],
            ckpt_path=None,
            verbose=True,
        )

    # ------------------------------
    # Inference
    # ------------------------------
    # inference on validation (DDP seems to destroy the order of samples)
    if is_master:
        lightning_module.enable_tta(tta_wrapper)
        predictor = pl.Trainer(logger=logger, gpus=1)
        outputs = predictor.predict(
            model=lightning_module,
            dataloaders=dataloaders["validation"],
            return_predictions=True,
        )
        predictions = concatenate([o["y_hat"] for o in outputs])
        targets = concatenate([o["y"] for o in outputs])

        # save
        if config.trainer.evaluation.save_predictions:
            save_predictions(
                predictions,
                config.trainer.evaluation.dirpath,
                config.trainer.evaluation.filename,
                split="validation",
            )

        # visualization
        lightning_module.run_visualization(
            predictions, targets, dataloaders["validation"]
        )
