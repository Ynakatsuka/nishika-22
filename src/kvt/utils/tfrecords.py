"""Ref: https://bit.ly/3KSSvGG"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm.auto import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tf_feature(value):
    if isinstance(value, int):
        return _int64_feature(value)
    elif isinstance(value, float):
        return _float_feature(value)
    else:
        return _bytes_feature(value)


def serialize_example(row, image_columns=[]):
    feature = {}
    for c in image_columns:
        image = np.array(Image.open(row[c])).tostring()
        feature[c] = _bytes_feature(image)
    for c in row.index:
        if c in image_columns:
            continue
        feature[c] = _tf_feature(row[c])
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


def write_tfrecords(
    df,
    filename,
    features_to_write=None,
    image_columns=[],
    compression_type="GZIP",
):
    """
    Example:
        >>> train_features_to_write = ['gid', 'path', 'cite_gid']
        >>> image_columns = ['path']
        >>> write_tfrecords(
                train, 
                'train.tfrecords', 
                features_to_write=train_features_to_write, 
                image_columns=image_columns
            )
    """
    if features_to_write is None:
        features_to_write = df.columns

    with tf.io.TFRecordWriter(
        filename, tf.io.TFRecordOptions(compression_type=compression_type)
    ) as writer:
        for (_, row) in tqdm(df[features_to_write].iterrows(), total=len(df)):
            example = serialize_example(row, image_columns=image_columns)
            writer.write(example)


def prepare_wave(x):
    return x


def read_labeled_tfrecord(example):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return (
        prepare_wave(example["wave"]),
        tf.reshape(tf.cast(example["target"], tf.float32), [1]),
        example["wave_id"],
    )


def read_unlabeled_tfrecord(example, return_image_id):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return (
        prepare_wave(example["wave"]),
        example["wave_id"] if return_image_id else 0,
    )


def get_tfrecord_dataset(
    files,
    batch_size=16,
    repeat=False,
    cache=False,
    shuffle=False,
    labeled=True,
    return_image_ids=True,
):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )
    if cache:
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(buffer_size=2048, seed=42)
        opt = tf.data.Options()
        opt.experimental_deterministic = True
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(
            lambda example: read_unlabeled_tfrecord(example, return_image_ids),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return tfds.as_numpy(ds)


class TFRecordDataLoader:
    def __init__(
        self,
        files,
        batch_size=32,
        cache=False,
        train=True,
        repeat=False,
        shuffle=False,
        labeled=True,
        return_image_ids=True,
    ):
        self.ds = get_dataset(
            files,
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,
            shuffle=shuffle,
            labeled=labeled,
            return_image_ids=return_image_ids,
        )

        if train:
            self.num_examples = count_data_items(files)
        else:
            self.num_examples = count_data_items_test(files)

        self.batch_size = batch_size
        self.labeled = labeled
        self.return_image_ids = return_image_ids
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        n_batches = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1
