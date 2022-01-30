import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def fix_dp_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key
        if key.startswith("model."):
            name = key[6:]
        new_state_dict[name] = value

    return new_state_dict


def fix_transformers_state_dict(state_dict):
    to_replace_keys = ["model.roberta"]
    replace_key = "net"

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key
        for to_replace_key in to_replace_keys:
            if key.startswith(to_replace_key):
                name = replace_key + key[len(to_replace_key) :]
        new_state_dict[name] = value

    return new_state_dict


def fix_mocov2_state_dict(state_dict):
    """
    Ref: https://bit.ly/3cDfGVA
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.encoder_q."):
            k = k.replace("model.encoder_q.", "")
            new_state_dict[k] = v

    return new_state_dict


def fix_state_dict_shallow(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = ".".join(k.split(".")[1:])
        new_state_dict[new_k] = v

    return new_state_dict


def load_state_dict_on_same_size(
    model, state_dict, threshold=0.5, infer_key=False
):
    new_state_dict = OrderedDict()
    overall_count, load_count = 0, 0
    skipping_layers = []

    for name, param in model.named_parameters():
        overall_count += 1
        if (name in state_dict.keys()) and (
            param.shape == state_dict[name].shape
        ):
            load_count += 1
            new_state_dict[name] = state_dict[name]
        else:
            skipping_layers.append(name)

    for name, param in model.named_buffers():
        overall_count += 1
        if (name in state_dict.keys()) and (
            param.shape == state_dict[name].shape
        ):
            load_count += 1
            new_state_dict[name] = state_dict[name]
        else:
            skipping_layers.append(name)

    load_ratio = load_count / overall_count

    if load_ratio < threshold:
        if infer_key:
            state_dict = fix_state_dict_shallow(state_dict)
            return load_state_dict_on_same_size(
                model, state_dict, threshold, infer_key=False
            )
        else:
            logger.info(f"Loading weights: {load_count} / {overall_count}")
            logger.info("State dict keys: ", state_dict.keys())
            logger.info("Model state dict keys: ", model.state_dict().keys())
            raise ValueError("Failed loading pretrained model.")
    else:
        logger.info(f"Skipping pretrained layers: {skipping_layers}")
        logger.info(f"Loading weights: {load_count} / {overall_count}")
        model.load_state_dict(new_state_dict, strict=False)
        return model
