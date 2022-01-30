import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image


class PILResize(ImageOnlyTransform):
    def __init__(
        self,
        width,
        height,
        resample_method="LANCZOS",
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.width = width
        self.height = height
        self.resample_method = getattr(Image, resample_method)

    def apply(self, image, **params):
        image = Image.fromarray(image)
        image = image.resize((self.width, self.height), self.resample_method)
        image = np.array(image, dtype=np.uint8)

        return image
