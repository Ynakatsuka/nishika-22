import os

import cv2
import easyocr
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

reader = easyocr.Reader(["ja", "en"])


def crop_image(img, min_size=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)[
        0
    ]
    x1, y1, x2, y2 = [], [], [], []
    for i in range(1, len(contours)):
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    if (
        x1
        and ((max(x2) - min(x1)) >= min_size)
        and ((max(y2) - min(y1)) >= min_size)
    ):
        img = img[min(y1) : max(y2), min(x1) : max(x2)]
    return img


def inpaint_white(img, x1, x2, y1, y2):
    result = img.copy()
    result[x1:x2, y1:y2] = 255
    return result


def inpaint_characters(img, threshold=0.3, whete_threshold=253):
    result = reader.readtext(img)
    for r in result:
        vertex = r[0]
        characters = r[1]
        score = r[2]
        if len(characters) == 1:
            continue
        if (score > threshold) or (len(characters) >= 8 and score >= 0.2):
            try:
                painted_img = inpaint_white(
                    img, vertex[0][1], vertex[2][1], vertex[0][0], vertex[1][0]
                )
                cropped = crop_image(painted_img)
                if cropped.mean() <= whete_threshold:
                    img = painted_img
            except TypeError:
                pass
    return img


def preprocess_pipeline(img):
    img = inpaint_characters(img)
    img = crop_image(img)
    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1)
    return img


def main():
    train = pd.read_csv("data/input/train.csv")
    test = pd.read_csv("data/input/test.csv")
    cite = pd.read_csv("data/input/cite_v2.csv")

    cite["path"] = cite["path"].apply(lambda x: f"cite_images/{x}")
    train["path"] = train["path"].apply(lambda x: f"apply_images/{x}")
    test["path"] = test["path"].apply(lambda x: f"apply_images/{x}")

    DIRNAME = "/home/working/data/input/"
    OUTPUT_DIRNAME = "/home/working/data/input/preprocessed_images/"
    os.makedirs(OUTPUT_DIRNAME, exist_ok=True)

    paths = train.path.tolist() + test.path.tolist() + cite.path.tolist()
    for path in tqdm(paths):
        filename = path.split("/")[-1]
        if os.path.exists(OUTPUT_DIRNAME + filename):
            continue

        img = cv2.imread(DIRNAME + path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_pipeline(img)
        cv2.imwrite(OUTPUT_DIRNAME + filename, img)


if __name__ == "__main__":
    main()
