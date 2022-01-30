#


def sample_transform(_split, **params):
    def transform(image, mask=None):
        def _transform(image):
            return image

        image = _transform(image)
        return image

    return transform
