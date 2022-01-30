import logging

import transformers
from transformers import AutoConfig

logger = logging.getLogger(__name__)


def get_transformers_auto_model(
    model_name, num_classes=None, model_type="AutoModel", **kwargs
):
    config = AutoConfig.from_pretrained(
        model_name, num_labels=num_classes, **kwargs
    )
    logger.info(f"Model Config: \n {config}")
    return getattr(transformers, model_type).from_pretrained(
        model_name, config=config
    )
