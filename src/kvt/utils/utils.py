import logging
import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import psutil
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict

logger = logging.getLogger(__name__)


def seed_torch(seed=None, random_seed=True):
    if random_seed or seed is None:
        seed = np.random.randint(0, 1000000)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def trace(title, logger=None):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2.0 ** 30
    yield
    m1 = p.memory_info()[0] / 2.0 ** 30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    message = (
        f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} "
    )
    print(message)
    if logger is not None:
        logger.info(message)


def check_attr(config, name):
    if hasattr(config, name):
        return config[name]
    else:
        return False


@contextmanager
def timer(name, logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time()-t0:.0f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def update_experiment_name(config):
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.experiment_name = ",".join(
            [
                e
                for e in config.experiment_name.split(",")
                # ignore parameters
                if ("trainer.idx_fold=" not in e)
                and ("mode=" not in e)
                and ("trainer.logger.group=" not in e)
                and ("trainer.feature_extraction=" not in e)
                and ("trainer.auto_resume_from_checkpoint=" not in e)
            ]
        )
        if not config.experiment_name:
            config.experiment_name = "default"

        if hasattr(config.trainer, "logger") and (
            not config.trainer.logger.name
        ):
            config.trainer.logger.name = "default"

    return config


def concatenate(results):
    if len(results) == 0:
        return results

    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(results[0], torch.Tensor):
        return torch.vstack([r.detach() for r in results])
    else:
        raise ValueError(f"Invalid result type: {type(results[0])}")


def save_predictions(predictions, dirpath, filename, split="validation"):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    path = os.path.join(dirpath, f"{split}_{filename}",)
    np.save(path, predictions)


def update_input_layer(model, in_channels):
    for l in model.children():
        if isinstance(l, nn.Sequential):
            for ll in l.children():
                assert ll.bias is None
                data = torch.mean(ll.weight, axis=1).unsqueeze(1)
                data = data.repeat((1, in_channels, 1, 1))
                ll.weight.data = data
                break
        else:
            assert l.bias is None
            data = torch.mean(l.weight, axis=1).unsqueeze(1)
            data = data.repeat((1, in_channels, 1, 1))
            l.weight.data = data
        break
    return model


def combine_model_parts(backbone, neck, head):
    if backbone is not None:
        # replace neck
        if neck is not None:
            replaced = False
            for layer_name in ["avgpool", "global_pool", "pooler"]:
                if hasattr(backbone, layer_name):
                    setattr(backbone, layer_name, neck)
                    replaced = True
                elif hasattr(backbone, "head") and hasattr(
                    backbone.head, layer_name
                ):
                    setattr(backbone.head, layer_name, neck)
                    replaced = True
            if replaced:
                logger.info(f"Replace Neck: {neck}")

        # replace head
        if head is not None:
            if hasattr(backbone, "classifier"):
                backbone.classifier = head
            elif hasattr(backbone, "fc"):
                backbone.fc = head
            elif hasattr(backbone, "last_linear"):
                backbone.last_linear = head
            elif hasattr(backbone, "head") and (hasattr(backbone.head, "fc")):
                backbone.head.fc = head
            elif hasattr(backbone, "head"):
                backbone.head = head
            logger.info(f"Replace Head: {head}")

    return backbone


def analyze_in_features(model):
    in_features = None
    # timm, torchvision, etc...
    if hasattr(model, "classifier"):
        if hasattr(model.classifier, "dense"):
            in_features = model.classifier.dense.in_features
        else:
            in_features = model.classifier.in_features
    elif hasattr(model, "classif"):
        in_features = model.classif.in_features
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
    elif hasattr(model, "last_linear"):
        in_features = model.last_linear.in_features
    elif hasattr(model, "head"):
        if hasattr(model.head, "fc"):
            if hasattr(model.head.fc, "in_features"):
                in_features = model.head.fc.in_features
            else:
                in_features = model.head.fc.in_channels
        else:
            in_features = model.head.in_features
    # transformers
    elif hasattr(model, "config"):
        if hasattr(model.config, "dim"):
            in_features = model.config.dim
        elif hasattr(model.config, "d_model"):
            in_features = model.config.d_model
        elif hasattr(model.config, "hidden_size"):
            in_features = model.config.hidden_size
        elif hasattr(model.config, "embedding_size"):
            in_features = model.config.embedding_size
    else:
        logger.info("Failed to analyze in_features")

    return in_features
