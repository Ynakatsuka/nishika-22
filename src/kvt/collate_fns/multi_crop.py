import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

try:
    from lightly.transforms import GaussianBlur, RandomSolarization
except ImportError:
    GaussianBlur, RandomSolarization = None, None

imagenet_normalize = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class DinoCollateFunction(nn.Module):
    """
    Based on BaseCollateFunction
    Ref: https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        local_crops_size=96,
        input_size=224,
        cj_prob=0.8,
        cj_bright=0.4,
        cj_contrast=0.4,
        cj_sat=0.2,
        cj_hue=0.1,
        random_gray_scale=0.2,
        random_solarization=0.2,
        random_gaussian_blur=0.1,
        random_gaussian_blur_local=0.5,
        hf_prob=0.5,
        normalize=imagenet_normalize,
    ):
        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomApply(
                    [T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)],
                    p=cj_prob,
                ),
                T.RandomGrayscale(p=random_gray_scale),
            ]
        )
        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        # first global crop
        self.global_transfo1 = T.Compose(
            [
                T.RandomResizedCrop(
                    input_size_,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = T.Compose(
            [
                T.RandomResizedCrop(
                    input_size_,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(random_gaussian_blur),
                RandomSolarization(random_solarization),
                normalize,
            ]
        )
        transforms = [self.global_transfo1, self.global_transfo2]

        # transformation for the local small crops
        for _ in range(local_crops_number):
            transforms.append(
                T.Compose(
                    [
                        T.RandomResizedCrop(
                            local_crops_size,
                            scale=local_crops_scale,
                            interpolation=Image.BICUBIC,
                        ),
                        flip_and_color_jitter,
                        GaussianBlur(p=random_gaussian_blur_local),
                        normalize,
                    ]
                )
            )
        self.transforms = transforms

    def forward(self, batch):
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = [trns(batch[:][0]) for trns in self.transforms]

        # only the 2 global views pass through the teacher
        transforms = (transforms[:2], transforms)

        return transforms, labels, fnames
