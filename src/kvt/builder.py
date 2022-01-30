import logging
from functools import partial

import hydra
import kvt
import kvt.augmentation
import torch.nn as nn
import torchvision.transforms as T
from easydict import EasyDict as edict
from kvt.registry import (
    COLLATE_FNS,
    DATASETS,
    HOOKS,
    LIGHTNING_MODULES,
    LOSSES,
    METRICS,
    MINERS,
    OPTIMIZERS,
    SAMPLERS,
    SCHEDULERS,
    TRANSFORMS,
)
from kvt.utils import build_from_config
from omegaconf import OmegaConf
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

try:
    import torch_optimizer as optim
except ImportError:
    optim = None
try:
    import kornia.augmentation as K
except ImportError:
    K = None
try:
    import torch_audiomentations as TA
except ImportError:
    TA = None

logger = logging.getLogger(__name__)


def build_collate_fn(config, dataset, split, **kwargs):
    collate_fn = None
    if hasattr(config.dataset, "collate_fn"):
        cfg = config.dataset.collate_fn
        if hasattr(cfg, split):
            # additional paramerters
            if hasattr(dataset, "tokenizer"):
                kwargs["tokenizer"] = dataset.tokenizer

            collate_fn = build_from_config(
                getattr(cfg, split), COLLATE_FNS, default_args=kwargs
            )

    return collate_fn


def build_sampler(config, dataset, split, **kwargs):
    kwargs["dataset"] = dataset
    sampler = None
    if hasattr(config.dataset, "sampler"):
        cfg = config.dataset.sampler
        if hasattr(cfg, split):
            sampler = build_from_config(
                getattr(cfg, split), SAMPLERS, default_args=kwargs
            )

    return sampler


def build_dataloaders(config, drop_last=None, shuffle=None, is_inference=False):
    dataloaders = {}
    dataset_configs = config.dataset.dataset

    if not isinstance(dataset_configs, list):
        dataset_configs = [dataset_configs]

    for dataset_config in dataset_configs:
        dataset_config = edict(dataset_config)
        for split_config in dataset_config.splits:
            cfg = edict(
                {"name": dataset_config.name, "params": dataset_config.params}
            )
            cfg.params.update(split_config)

            split = cfg.params.split
            is_train = split == "train"

            # if inference mode, skip loading train or validation dataset
            if is_inference and (split != "test"):
                continue

            if config.print_config:
                print("-" * 100)
                print(f"dataset config: \n {cfg}")

            if is_train:
                batch_size = config.trainer.train.batch_size
            else:
                batch_size = config.trainer.evaluation.batch_size

            # build transform
            transform_cfg = {
                "split": split,
                "aug_cfg": config.augmentation.transform.get(split),
            }
            for hw in ["height", "width"]:
                if hasattr(config.augmentation, hw):
                    transform_cfg[hw] = getattr(config.augmentation, hw)

            transform = build_from_config(
                config.dataset.transform,
                TRANSFORMS,
                default_args=transform_cfg,
            )

            # build dataset
            dataset = build_from_config(
                cfg,
                DATASETS,
                default_args={"transform": transform},
                match_object_args=True,
            )
            logger.info(f"Dataset split: {split}, n_samples: {len(dataset)}")
            logger.info(f"{split} transform: {transform}")

            # build dataloader parameters
            collate_fn = build_collate_fn(config, dataset, split=split)
            sampler = build_sampler(config, dataset, split=split)

            _shuffle = is_train if sampler is None else False
            _shuffle = _shuffle if shuffle is None else shuffle
            _drop_last = is_train if drop_last is None else drop_last

            dataloader = DataLoader(
                dataset,
                shuffle=_shuffle,
                batch_size=batch_size,
                drop_last=_drop_last,
                num_workers=config.dataset.transform.num_preprocessor,
                pin_memory=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            dataloaders[cfg.params.split] = dataloader

    return dataloaders


def build_model(config, **kwargs):
    build_model_hook_config = {"name": "DefaultModelBuilderHook"}
    hooks = config.hook.hooks
    if (hooks is not None) and ("build_model" in hooks):
        build_model_hook_config.update(hooks.build_model)

    build_model_fn = build_from_config(build_model_hook_config, HOOKS)
    return build_model_fn(config.model.model, **kwargs)


def build_optimizer(config, model=None, **kwargs):
    # for specific optimizers that needs "base optimizer"
    cfg = config.optimizer.optimizer
    if cfg.name == "AGC":
        # optimizer: instance
        base_optimizer = build_from_config(
            cfg.params.base, OPTIMIZERS, default_args=kwargs
        )
        optimizer = getattr(kvt.optimizers, cfg.name)(
            model.parameters(),
            base_optimizer,
            model=model,
            **{k: v for k, v in cfg.params.items() if k != "base"},
        )
    elif cfg.name == "Lookahead":
        # optimizer: instance
        base_optimizer = build_from_config(
            cfg.params.base, OPTIMIZERS, default_args=kwargs
        )
        optimizer = getattr(optim, cfg.name)(
            base_optimizer,
            **{k: v for k, v in cfg.params.items() if k != "base"},
        )
    elif cfg.name == "SAM":
        # optimizer: class
        base_optimizer = OPTIMIZERS.get(cfg.params.base.name)
        optimizer = getattr(kvt.optimizers, cfg.name)(
            model.parameters(),
            base_optimizer,
            **{k: v for k, v in cfg.params.items() if k != "base"},
        )
    else:
        optimizer = build_from_config(cfg, OPTIMIZERS, default_args=kwargs)

    return optimizer


def build_scheduler(config, **kwargs):
    if config.optimizer.scheduler is None:
        return None

    scheduler = build_from_config(
        config.optimizer.scheduler, SCHEDULERS, default_args=kwargs
    )

    return scheduler


def build_loss(config, **kwargs):
    # for pytorch metric learning
    if hasattr(config.model.loss, "params") and hasattr(
        config.model.loss.params, "loss"
    ):
        base_loss = build_from_config(config.model.loss.params.loss, LOSSES)
        config.model.loss.params.loss = base_loss
    if hasattr(config.model.loss, "params") and hasattr(
        config.model.loss.params, "distance"
    ):
        base_distance = build_from_config(
            config.model.loss.params.distance, LOSSES
        )
        config.model.loss.params.distance = base_distance
    return build_from_config(config.model.loss, LOSSES, default_args=kwargs)


def build_lightning_module(config, **kwargs):
    return build_from_config(
        config.lightning_module.lightning_module,
        LIGHTNING_MODULES,
        default_args=kwargs,
    )


def build_callbacks(config):
    """pytorch_lightning callbacks"""
    callbacks = []
    if config.trainer.callbacks is not None:
        cfgs = config.trainer.callbacks
        if not isinstance(cfgs, list):
            cfgs = [cfgs]
        for cfg in cfgs:  # cfgs: list of dict
            for key in cfg:
                callback = hydra.utils.instantiate(
                    OmegaConf.create(dict(cfg[key]))
                )
                callbacks.append(callback)
    return callbacks


def build_logger(config):
    """pytorch_lightning logger"""
    logger = None
    cfg = config.trainer.logger
    if cfg is not None:
        cfg = OmegaConf.create(dict(cfg))
        logger = hydra.utils.instantiate(cfg)
    return logger


def build_metrics(config):
    """pytorch_lightning metrics"""
    metrics = {}
    if config.model.metrics is not None:
        for name, cfg in config.model.metrics.items():
            metrics[name] = build_from_config(cfg, METRICS)
    return edict(metrics)


def build_hooks(config):
    if "hooks" in config.hook:
        hooks = config.hook.hooks
    else:
        hooks = None

    # build default hooks
    post_forward_hook_config = {"name": "DefaultPostForwardHook"}
    if (
        (hooks is not None)
        and ("post_forward" in hooks)
        and (hooks.post_forward is not None)
    ):
        post_forward_hook_config.update(hooks.post_forward)

    visualization_hook_configs = []
    if (
        (hooks is not None)
        and ("visualizations" in hooks)
        and (hooks.visualizations is not None)
    ):
        if isinstance(hooks.visualizations, list):
            visualization_hook_configs.extend(hooks.visualizations)
        else:
            visualization_hook_configs.append(hooks.visualizations)

    hooks_dict = {}
    hooks_dict["post_forward_fn"] = build_from_config(
        post_forward_hook_config, HOOKS
    )
    hooks_dict["visualization"] = [
        build_from_config(cfg, HOOKS) for cfg in visualization_hook_configs
    ]
    hooks = edict(hooks_dict)

    return hooks


def build_strong_transform(config):
    """strong transform (e.g. mixup)"""
    strong_transform, p = None, None
    if hasattr(config.augmentation, "strong_transform") and (
        config.augmentation.strong_transform is not None
    ):
        strong_cfg = config.augmentation.get("strong_transform")
        p = strong_cfg.params.p
        params = {k: v for k, v in strong_cfg.params.items() if k != "p"}
        if hasattr(kvt.augmentation, strong_cfg.name):
            strong_transform = partial(
                getattr(kvt.augmentation, strong_cfg.name), **params
            )
    return strong_transform, p


def build_batch_transform(config):
    """transform on torch.Tensor (torchvision, kornia)"""
    transforms = {}
    if hasattr(config.augmentation, "batch_transform") and (
        config.augmentation.batch_transform is not None
    ):
        cfgs = config.augmentation.get("batch_transform")
        source = cfgs["source"] if "source" in cfgs.keys() else None
        for split in cfgs.keys():
            if split == "source":
                continue

            assert split in ("train", "validation", "test")
            cfg = cfgs[split]
            if cfg is None:
                continue

            if source is None:
                trans = []
            elif source == "torchvision":
                trans = [
                    getattr(T, trans.name)(**trans.params) for trans in cfg
                ]
            elif source == "kornia":
                trans = [
                    getattr(K, trans.name)(**trans.params) for trans in cfg
                ]
            elif source == "torch_audiomentations":
                trans = [
                    getattr(TA, trans.name)(**trans.params) for trans in cfg
                ]
            else:
                raise ValueError(f"Invalid source: {source}")

            transforms[split] = nn.Sequential(*trans)

    return transforms


def build_tta_wrapper(config):
    """ttach"""
    wrapper = None
    if hasattr(config.augmentation, "tta_transform") and (
        config.augmentation.tta_transform is not None
    ):
        # build transforms
        cfg = config.augmentation.tta_transform
        transforms = hydra.utils.instantiate(
            OmegaConf.create(dict(cfg.transforms))
        )

        # build wrapper
        cfg.transforms = None
        cfg.model = None
        wrapper = hydra.utils.instantiate(OmegaConf.create(dict(cfg)))
        wrapper.transforms = transforms

    return wrapper


def build_plugins(config, is_ddp=True):
    # fix this issue
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6219
    plugins = []
    if is_ddp:
        if hasattr(config.trainer, "find_unused_parameters"):
            plugins.append(
                DDPPlugin(
                    find_unused_parameters=config.trainer.find_unused_parameters
                ),
            )
        else:
            plugins.append(DDPPlugin(find_unused_parameters=False))

    return plugins


def build_miner(config):
    """for PyTorchMetricLearning"""
    miner = None
    if hasattr(config.model, "miner") and (config.model.miner is not None):
        miner = build_from_config(config.model.miner, MINERS)
    return miner
