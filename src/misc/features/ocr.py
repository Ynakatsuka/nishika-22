import multiprocessing
import os
import warnings

import easyocr
import hydra
import pandas as pd
from base import BaseFeatureEngineering, BaseFeatureEngineeringDataset
from omegaconf import DictConfig


class OCRFeatures(BaseFeatureEngineeringDataset):
    reader = easyocr.Reader(["en"], gpu=True)

    def _engineer_features(self, image):
        result = self.reader.readtext(image)
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
        OCRFeatures, batch_size=num_workers, num_workers=0
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
