import logging

logger = logging.getLogger(__name__)


def initialize_model(
    config, model, backbone_lr_ratio=None, encoder_lr_ratio=None
):
    params = model.parameters()

    # set learning rate
    if backbone_lr_ratio is None:
        backbone_lr_ratio = config.trainer.train.backbone_lr_ratio
    if encoder_lr_ratio is None:
        encoder_lr_ratio = config.trainer.train.encoder_lr_ratio

    if backbone_lr_ratio is not None:
        if backbone_lr_ratio == 0:
            logger.info("Freezed Layers: ")
            for child in list(model.children())[:-1]:
                logger.info(child)
                for param in child.parameters():
                    param.requires_grad = False
        else:
            logger.info("Layer-wise Learning Rate:")
            base_lr = config.trainer.optimizer.params.lr
            params = []  # replace params
            for child in list(model.children())[:-1]:
                params.append(
                    {
                        "params": list(child.parameters()),
                        "lr": base_lr * backbone_lr_ratio,
                    }
                )
                logger.info(child, " lr: ", base_lr * backbone_lr_ratio)

    elif encoder_lr_ratio is not None:
        raise NotImplementedError

    return model, params


def reinitialize_model(model):
    params = model.parameters()

    for child in list(model.children())[:-1]:
        for param in child.parameters():
            param.requires_grad = True

    return model, params


def initialize_transformer_models(
    model, learning_rate, weight_decay, layerwise_learning_rate_decay=0.9,
):
    """
    Ref: https://bit.ly/3FwAYR6
    """
    model_type = model.model_name

    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "classifier" in n or "pooler" in n
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    layers = [getattr(model, model_type).embeddings] + list(
        getattr(model, model_type).encoder.layer
    )
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters
