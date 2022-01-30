import multiprocessing
import os
import warnings

import hydra
import pandas as pd
import timm
import torch
from base import BaseFeatureEngineering, BaseFeatureEngineeringDataset
from omegaconf import DictConfig


class EmbbedingFeatures(BaseFeatureEngineeringDataset):
    model = timm.create_model("resnet34", pretrained=True, num_classes=0)

    def _engineer_features(self, image):
        image = (image / 255).astype("float32")
        inputs = torch.tensor([image]).permute(0, 3, 1, 2)
        embeddings = self.model(inputs)
        embeddings = embeddings.detach().cpu().numpy().flatten()
        result = {
            f"resnet34_embedding_{i:03}": v for i, v in enumerate(embeddings)
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
        EmbbedingFeatures, batch_size=num_workers, num_workers=num_workers
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
