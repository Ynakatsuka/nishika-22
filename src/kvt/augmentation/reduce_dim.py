from albumentations.core.transforms_interface import ImageOnlyTransform


class KeepFirstChannel(ImageOnlyTransform):
    def __init__(
        self, always_apply=False, p=0.5,
    ):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return image[:, :, :1]
