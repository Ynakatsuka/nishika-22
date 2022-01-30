import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


def crop_image(img, use_otsu=False):
    thres = cv2.THRESH_OTSU if use_otsu else cv2.THRESH_BINARY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.threshold(gray, 127, 255, thres)[1]
    contours = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    x1, y1, x2, y2 = [], [], [], []
    for i in range(1, len(contours)):
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    if x1:
        x1_min = min(x1)
        y1_min = min(y1)
        x2_max = max(x2)
        y2_max = max(y2)
        img = img[y1_min:y2_max, x1_min:x2_max]

    return img


class CropMargin(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5, use_otsu=False):
        super().__init__(always_apply, p)
        self.use_otsu = use_otsu

    def apply(self, img, **params):
        return crop_image(img, self.use_otsu)
