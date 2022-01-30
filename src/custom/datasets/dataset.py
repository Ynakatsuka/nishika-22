import kvt
from kvt.datasets import BaseImageDataset


@kvt.DATASETS.register
class SampleDataset(BaseImageDataset):
    pass
