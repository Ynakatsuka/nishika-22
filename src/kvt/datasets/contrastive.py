import os
from copy import deepcopy

import numpy as np
import pandas as pd

from .base import BaseJpegImageDataset


class PairOfAnchorPositivieNegativeDataset(BaseJpegImageDataset):
    def __init__(
        self,
        csv_filename,
        input_column,
        target_column=None,
        input_dir="../data/input",
        extension=".jpg",
        target_unique_values=None,
        num_classes=None,
        enable_load=True,
        images_dir="",
        split="train",
        transform=None,
        fold_column="Fold",
        num_fold=5,
        idx_fold=0,
        label_smoothing=0,
        return_input_as_x=True,
        csv_input_dir=None,
        # for contrastive learning
        num_negatives=1,
        **params,
    ):
        super().__init__(
            csv_filename=csv_filename,
            input_column=input_column,
            target_column=target_column,
            input_dir=input_dir,
            extension=extension,
            target_unique_values=target_unique_values,
            num_classes=num_classes,
            enable_load=enable_load,
            images_dir=images_dir,
            split=split,
            transform=transform,
            fold_column=fold_column,
            num_fold=num_fold,
            idx_fold=idx_fold,
            label_smoothing=label_smoothing,
            return_input_as_x=return_input_as_x,
            csv_input_dir=csv_input_dir,
        )

        self.num_negatives = num_negatives

        # Inputs which the target appears only once are not suitable
        # for train data because positive samples cannot be sampled.
        self.all_inputs = deepcopy(self.inputs)
        self.all_targets = deepcopy(self.targets)

        self.inputs = pd.Series(self.inputs)
        self.targets = pd.Series(self.targets)
        value_counts = self.targets.value_counts()
        valid_targets = value_counts[value_counts >= 2].index
        valid_idx = self.targets.isin(valid_targets)
        self.inputs = self.inputs[valid_idx].tolist()
        self.targets = self.targets[valid_idx].tolist()

    def __getitem__(self, idx):
        path = self.inputs[idx]
        x_anchor = self._load(path)
        y_anchor = self.targets[idx]

        # positive sample
        pos_indices = np.where(np.array(self.all_targets) == y_anchor)[0]
        pos_indices = list(set(pos_indices) - set([idx]))
        pos_idx = np.random.choice(pos_indices, size=1, replace=False)[0]
        x_pos = self._load(self.all_inputs[pos_idx])
        y_pos = self.all_targets[pos_idx]

        # negative sample
        neg_indices = np.where(np.array(self.all_targets) != y_anchor)[0]
        neg_idx = np.random.choice(
            neg_indices, size=self.num_negatives, replace=False
        )
        x_neg = [self._load(self.all_inputs[idx]) for idx in neg_idx]
        y_neg = [self.all_targets[idx] for idx in neg_idx]

        x_anchor = self._preprocess_input(x_anchor)
        x_pos = self._preprocess_input(x_pos)
        x_neg = [self._preprocess_input(x) for x in x_neg]
        if self.transform is not None:
            x_anchor = self.transform(x_anchor)
            x_pos = self.transform(x_pos)
            x_neg = [self.transform(x) for x in x_neg]

        inputs = {
            "x_anchor": x_anchor,
            "x_pos": x_pos,
            "x_neg": x_neg,
            "y_anchor": y_anchor,
            "y_pos": y_pos,
            "y_neg": y_neg,
        }

        return inputs
