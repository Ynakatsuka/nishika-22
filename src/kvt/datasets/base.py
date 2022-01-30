import logging
import os

import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class BaseDataset(torch.utils.data.Dataset):
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
        **params,
    ):
        self.input_column = input_column
        self.target_column = target_column
        self.target_unique_values = target_unique_values
        self.num_classes = num_classes
        self.input_dir = input_dir
        self.extension = extension
        self.split = split
        self.transform = transform
        self.num_fold = num_fold
        self.idx_fold = idx_fold
        self.label_smoothing = label_smoothing
        self.enable_load = enable_load
        self.return_input_as_x = return_input_as_x

        # load
        if csv_input_dir is not None:
            df = pd.read_csv(os.path.join(csv_input_dir, csv_filename))
        else:
            df = pd.read_csv(os.path.join(input_dir, csv_filename))

        if fold_column in df.columns:
            if self.split == "validation":
                df = df[df[fold_column] == self.idx_fold]
            elif self.split == "train":
                df = df[df[fold_column] != self.idx_fold]
        else:
            logger.info(f"Thire is no {fold_column} fold column in DataFrame.")

        # image dir
        self.images_dir = images_dir

        # inputs
        if enable_load:
            self.inputs = self._extract_path_to_input_from_input_column(df)
        else:
            self.inputs = df[self.input_column].tolist()

        # targets
        if self.target_column in df.columns:
            logger.info(f"Dataset Info: {split} target describe:")
            logger.info(df[self.target_column].describe())
            self.targets = df[self.target_column].tolist()
        else:
            logger.info(
                f"Thire is no {target_column} target column in DataFrame."
            )
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def _extract_path_to_input_from_input_column(self, df):
        inputs = df[self.input_column].apply(
            lambda x: os.path.join(
                self.input_dir, self.images_dir, x + self.extension
            )
        )
        return inputs.tolist()

    def _preprocess_input(self, x):
        return x

    def _preprocess_target(self, y):
        if isinstance(y, np.ndarray):
            return y
        else:
            return np.array([y], dtype="float32")  # will be [batch_size, 1]

    def _load(self, path):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.enable_load:
            path = self.inputs[idx]
            x = self._load(path)
        else:
            x = self.inputs[idx]

        x = self._preprocess_input(x)

        if self.transform is not None:
            x = self.transform(x)

        if self.return_input_as_x:
            inputs = {"x": x}
        else:
            inputs = x

        if self.targets is not None:
            inputs["y"] = self._preprocess_target(self.targets[idx])

        return inputs


class BaseImageDataset(BaseDataset):
    def _load(self, path):
        x = cv2.imread(path)
        if x is None:
            raise ValueError(f"Failed to load an image: {path}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x


class BaseJpegImageDataset(BaseDataset):
    def _load(self, path):
        x = jpeg.JPEG(path).decode()
        return x


class BaseTextDataset(BaseDataset):
    def __init__(
        self,
        csv_filename,
        input_column,
        target_column=None,
        input_dir="../data/input",
        extension="",
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
        # for text data
        model_name=None,
        use_fast=False,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=False,
        return_attention_mask=True,
        return_token_type_ids=True,
        max_length=None,
        enable_bucket_sampler=False,
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast
        )
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.return_special_tokens_mask = return_special_tokens_mask
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.max_length = max_length
        self.enable_bucket_sampler = enable_bucket_sampler

        if self.enable_bucket_sampler:
            self.lengths = [len(inp.split()) for inp in self.inputs]

    def _preprocess_input(self, x):
        x = self.tokenizer(
            x,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask,
            return_token_type_ids=self.return_token_type_ids,
            return_special_tokens_mask=self.return_special_tokens_mask,
            max_length=self.max_length,
        )
        x = {k: v.squeeze() for k, v in x.items()}
        return x


class BaseClassificationDataset(BaseDataset):
    def _preprocess_target(self, y):
        if self.split == "train":
            smoothing = self.label_smoothing
        else:
            smoothing = 0

        n_labels = len(self.target_unique_values)
        labels = np.zeros(n_labels, dtype="float32") + smoothing / (
            n_labels - 1
        )
        labels[self.target_unique_values.index(y)] = 1.0 - smoothing

        return labels


class BaseEncodedClassificationDataset(BaseImageDataset):
    def _preprocess_target(self, y):
        if self.split == "train":
            smoothing = self.label_smoothing
        else:
            smoothing = 0

        labels = np.zeros(self.num_classes, dtype="float32") + smoothing / (
            self.num_classes - 1
        )
        labels[y] = 1.0 - smoothing

        return labels


class BaseImageEncodedClassificationDataset(
    BaseEncodedClassificationDataset, BaseImageDataset
):
    pass


class BaseImageMultiClassificationDataset(
    BaseImageDataset, BaseClassificationDataset
):
    pass
