import albumentations as albu
import kvt
import kvt.augmentation
import numpy as np

try:
    import audiomentations as audi
except ImportError:
    audi = None


def get_audio_transform(cfg):
    def get_object(trans):
        params = trans.params if trans.params is not None else {}

        if trans.name in {"Compose", "OneOf"}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(kvt.augmentation, trans.name)(augs_tmp, **params)

        if hasattr(audi, trans.name):
            return getattr(audi, trans.name)(**params)
        elif hasattr(kvt.augmentation, trans.name):
            return getattr(kvt.augmentation, trans.name)(**params)
        else:
            return eval(trans.name)(**params)

    augs = [get_object(t) for t in cfg]

    return audi.Compose(augs)


def get_image_transform(cfg):
    def get_object(trans):
        params = trans.params if trans.params is not None else {}

        if trans.name in {"Compose", "OneOf"}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(albu, trans.name)(augs_tmp, **params)

        if hasattr(albu, trans.name):
            return getattr(albu, trans.name)(**params)
        elif hasattr(kvt.augmentation, trans.name):
            return getattr(kvt.augmentation, trans.name)(**params)
        else:
            return eval(trans.name)(**params)

    augs = [get_object(t) for t in cfg]

    return albu.Compose(augs)


def base_transform(aug_cfg=None, **_):
    if aug_cfg is not None:
        raise ValueError(
            "When using base_transform, augmentation must be None."
        )
    return lambda x: x


def base_audio_transform(split, aug_cfg=None, **_):
    if aug_cfg is not None:
        aug = get_audio_transform(aug_cfg)
    else:
        aug = None
    return aug


def base_image_transform(split, aug_cfg=None, **_):
    if aug_cfg is not None:
        aug = get_image_transform(aug_cfg)
    else:
        return None

    def transform(image, mask=None):
        def _transform(image):
            augmented = aug(image=image)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1))
            return image

        image = _transform(image)

        if mask is not None:
            mask = _transform(mask)
            return {"image": image, "mask": mask}

        return image

    return transform
