import cv2
import imagehash
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pandarallel import pandarallel
from PIL import Image


class DuplicatedImageFinder:
    def __init__(self, threshold=0.8, use_cuda=True):
        pandarallel.initialize(progress_bar=True)

        self.threshold = threshold
        self.use_cuda = use_cuda
        self.funcs = [
            imagehash.average_hash,
            imagehash.phash,
            imagehash.dhash,
            imagehash.whash,
        ]

    def to_hashes(self, image):
        return (
            np.array([f(image).hash for f in self.funcs])
            .flatten()
            .astype(np.uint8)
        )

    def path2similarities(self, paths):
        hashes = paths.parallel_apply(lambda x: self.to_hashes(Image.open(x)))
        x = torch.from_numpy(np.array(hashes.to_list()))
        if self.use_cuda:
            x = x.cuda()

        if len(x) <= 10000:
            similarity_matrix = (
                1 - torch.cdist(x.float(), x.float(), p=1) / x.shape[1]
            )
            similarity_matrix = similarity_matrix.cpu().numpy()
        else:
            similarity_matrix = (
                np.array(
                    [
                        (x[i] == x).sum(dim=1).cpu().numpy()
                        for i in range(x.shape[0])
                    ]
                )
                / x.shape[1]
            )

        similarities = (
            pd.DataFrame(
                similarity_matrix, index=paths.values, columns=paths.values
            )
            .unstack()
            .rename("similarity")
            .reset_index()
        )
        similarities = similarities.query(
            f"level_0 != level_1 and similarity >= {self.threshold}"
        )
        similarities = similarities.sort_values("similarity", ascending=False)

        return similarities

    def group_similar_paths(self, paths, similarities):
        G = nx.from_pandas_edgelist(similarities, "level_0", "level_1")
        components = nx.connected_components(G)
        N = len(paths)
        groups = []
        for i, c in enumerate(components, N):
            groups.extend([(_c, i) for _c in c])
        groups = pd.DataFrame(groups, columns=["path", "group"])
        groups = paths.reset_index().merge(groups, how="left")
        groups["group"] = groups["group"].fillna(groups["index"]).astype(int)
        del groups["index"]

        return groups

    def __call__(self, paths):
        paths = pd.Series(paths, name="path")
        similarities = self.path2similarities(paths)
        groups = self.group_similar_paths(paths, similarities)

        return similarities, groups

    def _compare(self, path1, path2, similarity=None, figsize=(20, 20)):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for ax, path in zip(axs, (path1, path2)):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"{path}")
        if similarity is not None:
            fig.suptitle(f"Similarity: {similarity}", fontweight="bold")
            fig.tight_layout()
        plt.tight_layout()
        plt.show()

    def compare(self, similarities):
        for (path1, path2, similarity) in similarities.values:
            self._compare(path1, path2, similarity)
