import os

import hydra
import numpy as np
import pandas as pd
from kvt.utils import QueryExpansion, update_experiment_name
from omegaconf import DictConfig


def normalize(embeddings):
    embeddings /= np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    # fix experiment name
    config = update_experiment_name(config)

    # variables
    sample_submission_path = config.competition.sample_submission_path
    save_dir = config.save_dir

    # load
    sub = pd.read_csv(sample_submission_path)
    test = pd.read_csv(f"{save_dir}/test_and_cite.csv")
    embedding_names = [
        "experiment=exp016,model.model.backbone.name=convnext_large_in22ft1k",
        "experiment=exp012,model.model.backbone.name=swin_base_patch4_window7_224",
        "experiment=exp032",
    ]
    embedding_paths = [
        f"/home/working/data/output/predictions/test/{name}/test_fold_0.npy"
        for name in embedding_names
    ]
    embeddings = np.concatenate(
        [normalize(np.load(path)) for path in embedding_paths], axis=1
    ).astype("float32")

    n_query = len(sub)
    query_embeddings = embeddings[:n_query]
    reference_embeddings = embeddings[n_query:]
    reference_ids = test["gid"].values[n_query:]

    # get neighbors
    qe = QueryExpansion(
        alpha=1,
        k=50,
        similarity_threshold=0.7,
        normalize_similarity=True,
        strategy_to_deal_original="add",
        n_query_update_iter=1,
        n_reference_update_iter=0,
        batch_size=20,
    )
    query_embeddings, reference_embeddings = qe(
        query_embeddings, reference_embeddings
    )
    ids, sims = qe.search_index(
        reference_embeddings,
        query_embeddings,
        reference_ids=reference_ids,
        k=20,
        batch_size=10,
    )

    sub["cite_gid"] = [" ".join(_ids.astype(str)) for _ids in ids]

    # save submission
    dirpath = os.path.join(config.save_dir, "submission")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sub.to_csv(os.path.join(dirpath, f"ensemble.csv"), index=False)


if __name__ == "__main__":
    main()
