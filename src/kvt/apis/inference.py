import os

import kvt.utils
import pytorch_lightning as pl
import torch
from kvt.builder import (
    build_batch_transform,
    build_dataloaders,
    build_hooks,
    build_lightning_module,
    build_model,
    build_tta_wrapper,
)
from kvt.models.layers import Identity
from kvt.utils import (
    check_attr,
    combine_model_parts,
    concatenate,
    save_predictions,
)


def run(config):
    # ------------------------------
    # Building
    # ------------------------------
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, is_inference=True)

    # build datasets
    dataloaders = build_dataloaders(config, is_inference=True)

    # build torch transform (kornia)
    transform = build_batch_transform(config)

    # build tta wrapper
    tta_wrapper = build_tta_wrapper(config)

    # build lightning module
    lightning_module = build_lightning_module(
        config, model=model, hooks=hooks, transform=transform
    )

    # ------------------------------
    # Checkpoint
    # ------------------------------
    # load best checkpoint
    dir_path = os.path.join(config.save_dir, "models", config.experiment_name)
    filename = f"fold_{config.trainer.idx_fold}_best.ckpt"
    best_model_path = os.path.join(dir_path, filename)

    state_dict = torch.load(best_model_path)["state_dict"]

    # if using dp, it is necessary to fix state dict keys
    is_parallel = hasattr(config.trainer.trainer, "accelerator") and (
        config.trainer.trainer.accelerator in ("ddp", "ddp2", "dp")
    )
    if is_parallel:
        state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

    lightning_module.model.load_state_dict(state_dict)
    lightning_module.enable_tta(tta_wrapper)

    # ------------------------------
    # Update model for inference
    # ------------------------------
    # if using feature extracter
    if check_attr(config.trainer, "feature_extraction"):
        lightning_module.model = combine_model_parts(
            lightning_module.model, Identity(), Identity()
        )

    # ------------------------------
    # Inference
    # ------------------------------
    # inference on test
    predictor = pl.Trainer(gpus=1)
    outputs = predictor.predict(
        model=lightning_module,
        dataloaders=dataloaders["test"],
        return_predictions=True,
    )
    predictions = concatenate([o["y_hat"] for o in outputs])

    if check_attr(config.trainer, "feature_extraction"):
        predictions = predictions.squeeze()

    # save
    if config.trainer.inference.save_predictions:
        save_predictions(
            predictions,
            config.trainer.inference.dirpath,
            config.trainer.inference.filename,
            split="test",
        )
