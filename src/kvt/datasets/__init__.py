# flake8: noqa
from .base import (
    BaseClassificationDataset,
    BaseDataset,
    BaseEncodedClassificationDataset,
    BaseImageDataset,
    BaseImageEncodedClassificationDataset,
    BaseImageMultiClassificationDataset,
    BaseJpegImageDataset,
    BaseTextDataset,
)
from .contrastive import PairOfAnchorPositivieNegativeDataset
from .huggingface import MLMTextDataset
