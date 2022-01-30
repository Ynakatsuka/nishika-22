import logging

from .base import BaseTextDataset

logger = logging.getLogger(__name__)


class MLMTextDataset(BaseTextDataset):
    def __init__(
        self,
        csv_filename=None,
        input_column=None,
        target_column=None,
        input_dir=None,
        target_unique_values=None,
        num_classes=None,
        enable_load=True,
        images_dir="",
        split="train",
        transform=None,
        fold_column=None,
        num_fold=None,
        idx_fold=None,
        label_smoothing=0,
        model_name=None,
        use_fast=False,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=False,
        mlm_probability=None,
        **params,
    ):
        super().__init__(
            csv_filename=csv_filename,
            input_column=input_column,
            target_column=target_column,
            input_dir=input_dir,
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
            model_name=model_name,
            use_fast=use_fast,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_special_tokens_mask=return_special_tokens_mask,
            **params,
        )
        self.mlm_probability = mlm_probability

        logger.info(f"Model Max Length: {self.tokenizer.model_max_length}")

    def __getitem__(self, idx):
        if self.enable_load:
            path = self.inputs[idx]
            x = self._load(path)
        else:
            x = self.inputs[idx]

        if self.transform is not None:
            x = self.transform(x)

        x = self._preprocess_input(x)

        return x
