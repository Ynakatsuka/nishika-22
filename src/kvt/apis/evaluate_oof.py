import glob
import os

import numpy as np
import pandas as pd
import torch
from kvt.builder import build_logger, build_metrics


def run(config):
    # build hooks
    metric_fn = build_metrics(config)

    # build logger
    logger = build_logger(config)

    # variables
    fold_column = config.fold.fold_column
    target_column = config.competition.target_column
    num_fold = config.fold.fold.n_splits
    save_dir = config.save_dir
    csv_filename = config.fold.csv_filename

    # load train DataFrame
    load_train_path = os.path.join(save_dir, csv_filename)
    train = pd.read_csv(load_train_path)
    y_train = train[target_column]

    # load oof predictions
    load_oof_paths = sorted(
        glob.glob(f"{config.trainer.evaluation.dirpath}/*.npy")
    )
    assert len(load_oof_paths) == num_fold

    y_pred = np.zeros_like(y_train, dtype=float)
    for fold, load_oof_path in enumerate(load_oof_paths):
        valid_idx = train[fold_column] == fold
        loaded_object = np.load(load_oof_path)
        if (len(y_pred.shape) == 1) and len(loaded_object.shape) == 2:
            loaded_object = loaded_object.flatten()
        y_pred[valid_idx] = loaded_object

    if hasattr(
        config.lightning_module.lightning_module.params,
        "enable_numpy_evaluation",
    ) and (
        not config.lightning_module.lightning_module.params.enable_numpy_evaluation
    ):
        y_train = torch.tensor(y_train)
        y_pred = torch.tensor(y_pred)

    # evaluate
    results = {}
    for key, fn in metric_fn.items():
        results[f"{key}_across_folds"] = fn(y_pred, y_train)

        for fold in range(num_fold):
            valid_idx = train[fold_column] == fold
            results[f"{key}_fold_{fold}"] = fn(
                y_pred[valid_idx], y_train[valid_idx]
            )

    if hasattr(logger, "log_metrics"):
        logger.log_metrics(results)
