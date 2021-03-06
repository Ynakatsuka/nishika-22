import random

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _BaseKFold


class MultilabelStratifiedGroupKFold(_BaseKFold):
    """
    create multi-label stratified group kfold indexs.

    reference: https://bit.ly/3HEGinw
    """

    def __init__(self, n_splits=5, random_state=None, shuffle=False):
        super().__init__(
            n_splits=n_splits, random_state=random_state, shuffle=shuffle,
        )

    def _iter_test_indices(self, X=None, y=None, groups=None):
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        n_train, n_class = y.shape
        gid_unique = sorted(set(groups))
        n_group = len(gid_unique)

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        # aid_arr: (n_train,), indicates alternative id for group id.
        # generally, group ids are not 0-index and continuous or not integer.
        gid2aid = dict(zip(gid_unique, range(n_group)))
        aid_arr = np.vectorize(lambda x: gid2aid[x])(groups)

        # count labels by class
        cnts_by_class = y.sum(axis=0)  # (n_class, )

        # count labels by group id.
        col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
        cnts_by_group = (
            coo_matrix((np.ones(len(y)), (row, col)))
            .dot(coo_matrix(y))
            .toarray()
            .astype(int)
        )
        cnts_by_fold = np.zeros((self.n_splits, n_class), int)

        groups_by_fold = [[] for fid in range(self.n_splits)]
        group_and_cnts = list(enumerate(cnts_by_group))
        np.random.shuffle(group_and_cnts)
        for aid, cnt_by_g in sorted(
            group_and_cnts, key=lambda x: -np.std(x[1])
        ):
            best_fold = None
            min_eval = None
            for fid in range(self.n_splits):
                # eval assignment.
                cnts_by_fold[fid] += cnt_by_g
                fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
                cnts_by_fold[fid] -= cnt_by_g

                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fid

            cnts_by_fold[best_fold] += cnt_by_g
            groups_by_fold[best_fold].append(aid)

        idx_arr = np.arange(n_train)
        for fid in range(self.n_splits):
            val_groups = groups_by_fold[fid]

            val_indexs_bool = np.isin(aid_arr, val_groups)
            val_indexs = idx_arr[val_indexs_bool]

            yield val_indexs


class RegressionStratifiedKFold(_BaseKFold):
    """
    Ref: https://www.kaggle.com/abhishek/step-1-create-folds
    """

    def __init__(self, n_splits=5, random_state=None, shuffle=False):
        super().__init__(
            n_splits=n_splits, random_state=random_state, shuffle=shuffle,
        )

    def split(self, X=None, y=None, groups=None):
        # calculate number of bins by Sturge's rule
        # I take the floor of the value, you can also
        # just round it
        num_bins = int(np.floor(1 + np.log2(len(y))))

        # bin targets
        bins = pd.cut(y, bins=num_bins, labels=False).values

        # initiate the kfold class from model_selection module
        kf = StratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for train_index, valid_index in kf.split(X=X, y=bins, groups=groups):
            yield train_index, valid_index
