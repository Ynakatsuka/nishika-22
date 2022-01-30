from functools import partial

import kvt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


@kvt.METRICS.register
def recall20(pred, target, num_classes=1905):
    target = target.int()
    return torchmetrics.functional.recall(
        pred,
        target,
        average="samples",
        num_classes=num_classes,
        mdmc_average="global",
        top_k=20,
    )


@kvt.METRICS.register
class Recall20WithLogisticRegression:
    def __init__(
        self,
        test_size=0.2,
        random_state=42,
        apply_normalize=True,
        is_multiclass=False,
        **kwargs,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.apply_normalize = apply_normalize
        self.is_multiclass = is_multiclass

    def _get_model(self, **kwargs):
        return KNeighborsClassifier(n_jobs=-1)

    def __call__(self, embeddings, target):
        if self.apply_normalize:
            embeddings = normalize(embeddings)
        if self.is_multiclass:
            target = np.argmax(target, axis=1)

        # y_trainに全種類のラベルが含まれるようにする
        train_size = int(len(embeddings) * (1 - self.test_size))
        indices = list(range(len(embeddings)))
        unique_indices = list(
            pd.Series(target.flatten()).drop_duplicates().index
        )
        remain_indices = list(set(indices) - set(unique_indices))
        sample_size = train_size - len(unique_indices)
        if len(remain_indices) and (sample_size > 0):
            random_indices = list(
                np.random.choice(
                    remain_indices, size=sample_size, replace=False
                )
            )
            train_indices = unique_indices + random_indices
        else:
            train_indices = unique_indices
        test_indices = sorted(list(set(indices) - set(train_indices)))
        X_train = embeddings[train_indices]
        y_train = target[train_indices]
        X_test = embeddings[test_indices]
        y_test = target[test_indices]

        model = self._get_model(**self.model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred = torch.tensor(y_pred)
        y_test = torch.tensor(y_test)

        score = recall20(y_pred, y_test)

        return score
