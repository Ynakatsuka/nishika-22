import os
import time
import warnings

import custom  # noqa
import faiss
import jpeg4py as jpeg
import kvt.utils
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from hydra import compose, initialize
from kvt.builder import (
    build_batch_transform,
    build_hooks,
    build_lightning_module,
    build_model,
    build_tta_wrapper,
)
from kvt.initialization import initialize as kvt_initialize
from kvt.models.layers import Identity
from kvt.registry import TRANSFORMS
from kvt.utils import build_from_config, check_attr, combine_model_parts
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# set available gpus
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model names
EXPERIMENT_NAMES = [
    "experiment=exp016,model.model.backbone.name=convnext_large_in22ft1k",
    "experiment=exp012,model.model.backbone.name=swin_base_patch4_window7_224",
    "experiment=exp032",
]


def load_lightning_module(config):
    # ------------------------------
    # Building
    # ------------------------------
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, is_inference=True)

    # build torch transform (kornia)
    transform = build_batch_transform(config)

    # build tta wrapper
    tta_wrapper = build_tta_wrapper(config)

    # build lightning module
    lightning_module = build_lightning_module(
        config, model=model, hooks=hooks, transform=transform
    )

    # ------------------------------
    # Checkpoint
    # ------------------------------
    # load best checkpoint
    dir_path = os.path.join(config.save_dir, "models", config.experiment_name)
    filename = f"fold_{config.trainer.idx_fold}_best.ckpt"
    best_model_path = os.path.join(dir_path, filename)

    state_dict = torch.load(best_model_path)["state_dict"]

    # if using dp, it is necessary to fix state dict keys
    is_parallel = hasattr(config.trainer.trainer, "accelerator") and (
        config.trainer.trainer.accelerator in ("ddp", "ddp2", "dp")
    )
    if is_parallel:
        state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

    lightning_module.model.load_state_dict(state_dict)
    lightning_module.enable_tta(tta_wrapper)

    # ------------------------------
    # Update model for inference
    # ------------------------------
    # if using feature extracter
    if check_attr(config.trainer, "feature_extraction"):
        lightning_module.model = combine_model_parts(
            lightning_module.model, Identity(), Identity()
        )

    lightning_module.eval().cuda()
    return lightning_module


def load_augmentation(config, split="test"):
    transform = None
    dataset_configs = config.dataset.dataset

    if not isinstance(dataset_configs, list):
        dataset_configs = [dataset_configs]

    for dataset_config in dataset_configs:
        for split_config in dataset_config.splits:
            cfg = edict(
                {"name": dataset_config.name, "params": dataset_config.params}
            )
            cfg.params.update(split_config)

            split = cfg.params.split
            if split != "test":
                continue

            # build transform
            transform_cfg = {
                "split": split,
                "aug_cfg": config.augmentation.transform.get(split),
            }
            for hw in ["height", "width"]:
                if hasattr(config.augmentation, hw):
                    transform_cfg[hw] = getattr(config.augmentation, hw)

            transform = build_from_config(
                config.dataset.transform,
                TRANSFORMS,
                default_args=transform_cfg,
            )

    return transform


def normalize(embeddings):
    embeddings /= np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings


def create_index(reference_embeddings, use_cuda=True):
    # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    index = faiss.IndexFlatIP(reference_embeddings.shape[1])  # inner product
    if use_cuda:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    return index


def search_index(
    index, query_embeddings, reference_ids=None, k=20, apply_normalize=True
):
    if apply_normalize:
        query_embeddings = normalize(query_embeddings)

    D, I = index.search(query_embeddings, k)
    if reference_ids is not None:
        I = np.apply_along_axis(lambda x: reference_ids[x], 1, I)

    return D, I


def get_result(
    path_or_image, preprocessors, transforms, models, index, reference_ids, k=20
):
    if isinstance(path_or_image, str):
        img = jpeg.JPEG(path_or_image).decode()
    else:
        img = path_or_image

    embeddings = []
    for preprocess, transform, model in zip(preprocessors, transforms, models):
        x = preprocess(img)
        x = transform(x)
        x = torch.tensor([x]).cuda()
        with torch.inference_mode():
            embeddings.append(normalize(model(x).cpu().detach().numpy()))
        torch.cuda.empty_cache()
    embeddings = np.hstack(embeddings)
    embeddings = normalize(embeddings)
    D, I = search_index(index, embeddings, reference_ids=reference_ids, k=k)
    return D, I, embeddings


def load_config(experiment_name="", overrides=()):
    with initialize(config_path="../../config/"):
        config = compose(
            config_name="default",
            return_hydra_config=True,
            overrides=overrides,
        )

    # update parameters for jupyter
    del config.hydra
    config.work_dir = "/home/working"
    config.experiment_name = experiment_name
    config = edict(OmegaConf.to_container(config, resolve=True))
    return config


def main():
    # config
    config = load_config()

    # variables
    sample_submission_path = config.competition.sample_submission_path
    save_dir = config.save_dir

    # load reference
    sub = pd.read_csv(sample_submission_path)
    test = pd.read_csv(config.competition.test_path)
    cite = pd.read_csv(config.competition.cite_path)
    embedding_paths = [
        f"{save_dir}/predictions/test/{name}/test_fold_0.npy"
        for name in EXPERIMENT_NAMES
    ]
    embeddings = np.concatenate(
        [normalize(np.load(path)) for path in embedding_paths], axis=1
    ).astype("float32")
    embeddings = normalize(embeddings)
    n_query = len(sub)
    test_embeddings = embeddings[:n_query]
    reference_embeddings = embeddings[n_query:]
    reference_ids = cite["gid"].values

    # load models
    models, transforms, preprocessors = [], [], []
    for name in EXPERIMENT_NAMES:
        overrides = name.split(",")
        config = load_config(name, overrides=overrides)
        models.append(load_lightning_module(config))
        transforms.append(load_augmentation(config))
        preprocessors.append(lambda x: x)

    # create index
    index = create_index(reference_embeddings, use_cuda=True)

    # benchmark
    results = []
    for i, filename in tqdm(enumerate(test["path"].values)):
        start = time.time()
        path = f"{config.input_dir}/apply_images/{filename}"
        D, I, embeddings = get_result(
            path, preprocessors, transforms, models, index, reference_ids, k=20
        )
        end = time.time()
        assert embeddings == test_embeddings[i]
        results.append(end - start)
    print(pd.Series(results).describe())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    kvt_initialize()
    main()
