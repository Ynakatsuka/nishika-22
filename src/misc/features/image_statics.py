import multiprocessing
import os
import sys
import warnings

import hydra
import pandas as pd
from base import BaseFeatureEngineering, BaseFeatureEngineeringDataset
from omegaconf import DictConfig


class StatFeatures(BaseFeatureEngineeringDataset):
    def _engineer_features(self, image):
        height, width, _ = image.shape
        area = height * width
        aspect_ratio = height / width
        filesize = sys.getsizeof(image)
        filesize_per_pixel = filesize / area
        result = {
            "height": height,
            "width": width,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "filesize": filesize,
            "filesize_per_pixel": filesize_per_pixel,
        }
        return result


@hydra.main(config_path="../../../config", config_name="default")
def main(config: DictConfig) -> None:
    filename = __file__.split("/")[-1][:-3]
    input_dir = config.input_dir
    features_dir = config.features_dir
    os.makedirs(features_dir, exist_ok=True)

    df = pd.read_csv(config.competition.train_path)
    input_paths = df["Id"].apply(lambda x: f"{input_dir}/train/{x}.jpg")

    num_workers = multiprocessing.cpu_count()
    transformer = BaseFeatureEngineering(
        StatFeatures, batch_size=num_workers, num_workers=num_workers
    )

    X = transformer.fit_transform(input_paths)
    print(X.info())

    pd.to_pickle(
        transformer, os.path.join(features_dir, f"{filename}_transformer.pkl")
    )
    X.to_pickle(os.path.join(features_dir, f"{filename}_train.pkl"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
