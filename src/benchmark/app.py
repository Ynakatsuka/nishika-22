import warnings

import numpy as np
import pandas as pd
import streamlit as st
from benchmark import (
    EXPERIMENT_NAMES,
    create_index,
    get_result,
    load_augmentation,
    load_config,
    load_lightning_module,
    normalize,
)
from kvt.initialization import initialize as kvt_initialize
from PIL import Image


@st.cache(allow_output_mutation=True)
def load():
    # config
    config = load_config()

    # variables
    sample_submission_path = config.competition.sample_submission_path
    save_dir = config.save_dir

    # load reference
    sub = pd.read_csv(sample_submission_path)
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

    return config, preprocessors, transforms, models, index, reference_ids


def main(config, preprocessors, transforms, models, index, reference_ids):
    # draw the page
    st.title("Similar Trade Mark Image Search")

    k = 20
    n_cols, n_rows = 5, 4
    assert n_cols * n_rows == k

    # search
    uploaded_file = st.sidebar.file_uploader("Upload Image File", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="Query Image", use_column_width=True)

        D, I, _embeddings = get_result(
            np.array(image),
            preprocessors,
            transforms,
            models,
            index,
            reference_ids,
            k=k,
        )
        assert len(D) == 1

        # draw image
        st.header("Found Images:")
        col = st.columns(n_cols)
        for i, (sim, ref_id) in enumerate(zip(D[0], I[0])):
            if (i > 0) and (i % n_cols == 0):
                col = st.columns(n_cols)

            with col[i % n_cols]:
                path = f"{config.input_dir}/cite_images/{ref_id}/{ref_id}.jpg"
                image = Image.open(path)
                st.image(
                    image,
                    caption=f"#{i+1}: Similarity: {sim:.3f}",
                    use_column_width=True,
                )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    kvt_initialize()
    config, preprocessors, transforms, models, index, reference_ids = load()
    main(config, preprocessors, transforms, models, index, reference_ids)
