import os
import pprint

import hydra
import nmslib
import numpy as np
import pandas as pd
from kvt.utils import update_experiment_name
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    # fix experiment name
    config = update_experiment_name(config)

    # variables
    sample_submission_path = config.competition.sample_submission_path
    save_dir = config.save_dir
    embedding_path = os.path.join(
        config.trainer.inference.dirpath,
        "test_" + config.trainer.inference.filename,
    )

    # load
    sub = pd.read_csv(sample_submission_path)
    test = pd.read_csv(f"{save_dir}/test_and_cite.csv")
    embeddings = np.load(embedding_path).astype("float32")

    n_query = len(sub)
    query_embeddings = embeddings[:n_query]
    reference_embeddings = embeddings[n_query:]
    query_ids = test["gid"].values[:n_query]
    reference_ids = test["gid"].values[n_query:]

    # get neighbors
    index = nmslib.init(method="hnsw", space="cosinesimil")
    index.addDataPointBatch(reference_embeddings)
    index.createIndex({"post": 2}, print_progress=True)
    results = {}
    for i in tqdm(range(query_embeddings.shape[0])):
        indices, _ = index.knnQuery(query_embeddings[i], k=20)
        ids = [reference_ids[idx] for idx in indices]
        query_id = query_ids[i]
        results[query_id] = ids

    sub["cite_gid"] = results.values()
    sub["cite_gid"] = sub["cite_gid"].apply(lambda x: " ".join(map(str, x)))

    # save submission
    dirpath = os.path.join(config.save_dir, "submission")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sub.to_csv(
        os.path.join(dirpath, f"{config.experiment_name}.csv"), index=False
    )


if __name__ == "__main__":
    main()
