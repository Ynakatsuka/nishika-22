import abc
import logging
import os

import kvt.registry
import torch
import torch.nn as nn
from kvt.models.layers import Identity, MixLinear
from kvt.models.wrappers import MetricLearningModelWrapper
from kvt.registry import BACKBONES, HEADS, MODELS, NECKS
from kvt.utils import (
    analyze_in_features,
    build_from_config,
    combine_model_parts,
    replace_bn,
)

logger = logging.getLogger(__name__)


class ModelBuilderHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, config):
        pass


class DefaultModelBuilderHook(ModelBuilderHookBase):
    def __call__(self, config, is_inference=False, **kwargs):
        """
        Args:
            config (EasyDict): model config that has below keys:
                - name: model type (e.g., classification, lightly, UNet, SED)
                - params: model parameters
                - backbone: backbone config
                - head: head config
                - neck: neck config

        Config Example:
            model:
            name: "classification"
            params:
            backbone:
                name: "resnet34d"
                params:
                num_classes: 6
                pretrained: True
                drop_rate: 0.1
                # attn_drop_rate: 0.0
                drop_path_rate: 0.1
            neck:
                - name: "GeMPool2d"
                params:
            head:
                - name: "Flatten"
                params:
                - name: "BatchNorm1d"
                params:
                    num_features: "auto"
                - name: "Linear"
                params:
                    in_features: "auto"
                    out_features: "auto"
                - name: "Dropout"
                params:
                    p: 0.3
                - name: "ReLU"
                params:
                - name: "Linear"
                params:
                    in_features: "auto"
                    out_features: 6
            others:
                disable_bn: False
                mixout: False
        """
        # build backbone
        # state_dict is not None if you set pretrained path
        backbone, state_dict = self.build_backbone(
            config.backbone, is_inference
        )
        in_features = analyze_in_features(backbone)

        # build neck
        neck = self.build_sequential_from_config(
            config.neck, NECKS, in_features=in_features
        )

        # build head
        head = self.build_sequential_from_config(
            config.head, HEADS, in_features=in_features
        )

        # classification models
        if config.name == "classification":
            model = self.build_classification_model(backbone, neck, head)
        # UNet
        elif "UNet" in config.name:
            model = build_from_config(config, MODELS)
        # SED models, transformers or others
        else:
            backbone = combine_model_parts(backbone, neck, head)
            model = self.build_wrapper_model(
                config, backbone, neck, head, in_features=in_features
            )

        # Load state_dict
        if state_dict is not None:
            model = kvt.utils.load_state_dict_on_same_size(
                model, state_dict, infer_key=True
            )

        if hasattr(config, "others"):
            # mixout
            if hasattr(config.others, "mixout") and (config.others.mixout > 0):
                logger.info("Apply mixout")
                model = self.apply_mixout(model, config.others.mixout)

            # replace bn
            if (
                hasattr(config.others, "replace_bn")
                and config.others.replace_bn
            ):
                logger.info("Replace BatchNorm to Identity")
                model = replace_bn(model)

        return model

    def build_sequential_from_config(self, configs, registry, in_features=None):
        layers = []
        configs = [] if configs is None else configs

        for config in configs:
            # overwrite "auto"
            if (config["params"] is not None) and (in_features is not None):
                for key, value in config["params"].items():
                    if "auto" in str(value):
                        config["params"][key] = eval(
                            value.replace("auto", str(in_features))
                        )

            layers.append(build_from_config(config, registry))

        if len(layers) > 1:
            layers = nn.Sequential(*layers)
        elif len(layers) == 1:
            layers = layers[0]
        else:
            layers = None

        return layers

    def build_backbone(self, config, is_inference):
        state_dict = None

        if config is None:
            backbone = None
        # if inference mode, don't load pretrained model
        # (will be loaded after model builder)
        elif is_inference:
            logger.info("Skip loading pretrained model")
            config.params.pretrained = False
            backbone = build_from_config(config, BACKBONES)

        # load pretrained model from local path (e.g., trained on external data)
        # state_dict will be loaded after combining model parts
        elif hasattr(config.params, "pretrained") and isinstance(
            config.params.pretrained, str
        ):
            path = config.params.pretrained
            logger.info(f"Loaded pretrained model: {path}")

            config.params.pretrained = False
            backbone = build_from_config(config, BACKBONES)

            if os.path.exists(path):
                loaded_object = torch.load(path)
                if "state_dict" in loaded_object.keys():
                    state_dict = loaded_object["state_dict"]
                else:
                    state_dict = loaded_object
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    path, progress=True
                )["state_dict"]

            # fix state_dict: local model trained on dp
            state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

            # fix state_dict: SSL
            if hasattr(config, "fix_state_dict"):
                if config.fix_state_dict == "mocov2":
                    state_dict = kvt.utils.fix_mocov2_state_dict(state_dict)
                elif config.fix_state_dict == "transformers":
                    state_dict = kvt.utils.fix_transformers_state_dict(
                        state_dict
                    )
                else:
                    raise KeyError

        # regular loading
        else:
            backbone = build_from_config(config, BACKBONES)

        return backbone, state_dict

    def build_classification_model(self, backbone, neck, head):
        """combine backbone, neck and head"""
        metric_learnings = kvt.models.heads.metric_learning.__dict__.keys()
        is_metric_learning = (
            True if head.__class__.__name__ in metric_learnings else False
        )

        if is_metric_learning:
            logger.info("Updated model for metric learning")
            backbone = combine_model_parts(backbone, neck, Identity())
            model = MetricLearningModelWrapper(backbone, head)
        else:
            model = combine_model_parts(backbone, neck, head)

        return model

    def build_wrapper_model(
        self, config, backbone, neck, head, in_features=None
    ):
        if config.backbone.name == "resnest50":
            layers = list(backbone.children())[:-2]
            backbone = nn.Sequential(*layers)

        args = {"backbone": backbone, "neck": neck, "head": head}

        # auto detecting "num_ftrs" for lightly models
        if (
            (config.params is not None)
            and ("num_ftrs" in config.params.keys())
            and (config.params.num_ftrs != in_features)
            and (in_features is not None)
        ):
            logger.info(
                f"Replace num_ftrs: {config.params.num_ftrs} -> {in_features}"
            )
            config.params.num_ftrs = in_features

        model = build_from_config(
            config, MODELS, default_args=args, match_object_args=True
        )

        return model

    def apply_mixout(self, model, p):
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features,
                        module.out_features,
                        bias,
                        target_state_dict["weight"],
                        p,
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
        return model
