import more_itertools
import numpy as np
import sklearn
import torch


def divide_chunks(l, n):
    if n == len(l):
        yield np.arange(len(l), dtype=np.int32), l
    else:
        # looping till length l
        for i in range(0, len(l), n):
            data = l[i : i + n]
            yield np.arange(i, i + len(data), dtype=np.int32), data


def prepare_buckets(lens, bucket_size, batch_size, shuffle=True, indices=None):
    lens = -lens
    assert bucket_size % batch_size == 0 or bucket_size == len(lens)
    if indices is None:
        if shuffle:
            indices = sklearn.utils.shuffle(
                np.arange(len(lens), dtype=np.int32)
            )
            lens = lens[indices]
        else:
            indices = np.arange(len(lens), dtype=np.int32)
    new_indices = []
    extra_batch = None
    for chunk_index, chunk in divide_chunks(lens, bucket_size):
        # sort indices in bucket by descending order of length
        indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
        batches = []
        for _, batch in divide_chunks(indices_sorted, batch_size):
            if len(batch) == batch_size:
                batches.append(batch.tolist())
            else:
                assert extra_batch is None
                assert batch is not None
                extra_batch = batch
        # shuffling batches within buckets
        if shuffle:
            batches = sklearn.utils.shuffle(batches)
        for batch in batches:
            new_indices.extend(batch)

    if extra_batch is not None:
        new_indices.extend(extra_batch)
    return indices[new_indices]


class BucketSampler(torch.utils.data.Sampler):
    """
    Ref: https://www.kaggle.com/yaroshevskiy/bert-base-2-epochs
    """

    def __init__(
        self, dataset, bucket_size=None, batch_size=1536, shuffle=True,
    ):
        super().__init__(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sort_keys = np.array(dataset.lengths)
        self.bucket_size = (
            bucket_size if bucket_size is not None else len(self.sort_keys)
        )
        if not shuffle:
            self.index = prepare_buckets(
                self.sort_keys,
                bucket_size=self.bucket_size,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        else:
            self.index = None
        self.weights = None

    def set_weights(self, w):
        assert w >= 0
        total = np.sum(w)
        if total != 1:
            w = w / total
        self.weights = w

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_keys)

            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = prepare_buckets(
                self.sort_keys,
                bucket_size=self.bucket_size,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                indices=indices,
            )

        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_keys)


class SmartBatchingSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None

    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds
