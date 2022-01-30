import os

import hydra
import numpy as np
import pandas as pd
from kvt.utils import QueryExpansion, update_experiment_name
from omegaconf import DictConfig
from tqdm.auto import tqdm


def search_index(index, query_embeddings, reference_ids=None, k=20):
    results_id, results_sim = [], []
    for i in tqdm(range(query_embeddings.shape[0])):
        D, indices = index.search(np.expand_dims(query_embeddings[i], 0), k)
        if reference_ids is not None:
            ids = [reference_ids[idx] for idx in indices[0]]
        else:
            ids = indices[0]
        results_id.append(ids)
        results_sim.append(D[0])
    return np.vstack(results_id), np.vstack(results_sim)


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
    reference_ids = test["gid"].values[n_query:]

    # get neighbors
    qe = QueryExpansion(
        alpha=1,
        k=3,
        similarity_threshold=None,
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
    sub.to_csv(
        os.path.join(dirpath, f"{config.experiment_name}.csv"), index=False
    )


if __name__ == "__main__":
    main()
