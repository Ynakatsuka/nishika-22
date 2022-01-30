import random

import albumentations.augmentations.functional as F
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class ChoiceRotate(ImageOnlyTransform):
    def __init__(
        self,
        limit=(120, 240),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.rotate(img, angle, interpolation, self.border_mode, self.value)

    def get_params(self):
        return {"angle": random.choice(self.limit)}
