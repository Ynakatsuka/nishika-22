"""
Hooks for Visualization. Used in LightningModule.
There are some points of visualization points.
1. validation dataloader
2. predictions
3. comparison with targets
"""
import abc
import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image

try:
    import torchcam
    from torchcam.utils import overlay_mask
except ImportError:
    torchcam, overlay_mask = None, None


def make_subset_dataloader(dataloader, indices, batch_size):
    new_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataloader.dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        sampler=dataloader.sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        drop_last=dataloader.drop_last,
    )
    return new_dataloader


def hstack_pil_image(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class VisualizationHookBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dirpath, experiment_name, figsize=(20, 20), suffix=""):
        self.experiment_name = experiment_name
        self.figsize = figsize

        sns.set()
        self.filename = (
            f"{experiment_name}_{self.__class__.__name__}{suffix}.png"
        )
        self.save_path = os.path.join(dirpath, self.filename)
        self.result = {self.__class__.__name__ + suffix: self.save_path}

        try:
            os.makedirs(dirpath, exist_ok=True)
        except OSError:
            pass

    @abc.abstractmethod
    def __call__(self, model, dataloader, predictions, targets, logger=None):
        raise NotImplementedError


class ScatterPlotVisualizationHook(VisualizationHookBase):
    def __call__(self, model, dataloader, predictions, targets, logger=None):
        df = pd.DataFrame(
            {
                "predictions": predictions.flatten(),
                "targets": targets.flatten(),
            }
        )
        df.sort_values("predictions", inplace=True)

        vmin = min(predictions.min(), targets.min()) - targets.std() / 10
        vmax = max(predictions.max(), targets.max()) + targets.std() / 10

        plt.clf()
        plt.figure(figsize=self.figsize)
        sns.jointplot(
            x="predictions",
            y="targets",
            data=df,
            kind="reg",
            color=None,
            xlim=(vmin, vmax),
            ylim=(vmin, vmax),
        )
        plt.savefig(self.save_path)

        return self.result


class LiftChartVisualizationHook(VisualizationHookBase):
    def __init__(self, dirpath, experiment_name, figsize=(15, 5), bins=50):
        super().__init__(dirpath, experiment_name, figsize)
        self.bins = bins

    def __call__(self, model, dataloader, predictions, targets, logger=None):
        df = pd.DataFrame(
            {
                "predictions": predictions.flatten(),
                "targets": targets.flatten(),
            }
        )
        df.sort_values("predictions", inplace=True)

        chunk_size = np.ceil(len(df) / self.bins)
        df["bin"] = np.floor(np.arange(1, 1 + len(df)) / chunk_size)
        df.set_index("bin", inplace=True)

        plt.clf()
        plt.figure(figsize=self.figsize)
        sns.lineplot(data=df, markers=True, dashes=False)
        plt.savefig(self.save_path)

        return self.result


class ConfusionMatrixVisualizationHook(VisualizationHookBase):
    def __init__(
        self,
        dirpath,
        experiment_name,
        figsize=(20, 20),
        classes=None,
        normalize=True,
    ):
        super().__init__(dirpath, experiment_name, figsize)
        self.classes = classes
        self.normalize = normalize

    def __call__(self, model, dataloader, predictions, targets, logger=None):
        plt.clf()
        # compute confusion_matrix
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        cm = confusion_matrix(targets, predictions)

        if self.normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # plot
        plt.clf()
        plt.figure(figsize=self.figsize)
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.colorbar()

        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = ".2f" if self.normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.savefig(self.save_path)

        return self.result


class GradCamVisualizationHook(VisualizationHookBase):
    def __init__(
        self,
        dirpath,
        experiment_name,
        figsize=(20, 20),
        method="SmoothGradCAMpp",
        num_plots=15,
        input_shape=(3, 256, 256),
        target_layer=None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        select_top_predictions=True,
        use_cuda=True,
    ):
        suffix = "_top" if select_top_predictions else "_bottom"
        super().__init__(dirpath, experiment_name, figsize, suffix)
        self.method = method
        self.num_plots = num_plots
        self.input_shape = input_shape
        self.target_layer = target_layer
        self.mean = mean
        self.std = std
        self.select_top_predictions = select_top_predictions
        self.use_cuda = use_cuda

    def _extract_cam_on_first_batch(self, cam_extractor, model, dataloader):
        if self.use_cuda:
            model.cuda()
        inputs, output_cams = [], []
        batch = iter(dataloader).next()
        for x in batch["x"]:
            if self.use_cuda:
                out = model(x.unsqueeze(0).cuda())
            else:
                out = model(x.unsqueeze(0))
            activation_map = cam_extractor(
                out.unsqueeze(0).argmax().item(), out
            )
            if isinstance(activation_map, list) and len(activation_map) == 1:
                activation_map = activation_map[0]
            inputs.append(x)
            output_cams.append(activation_map)
        return inputs, output_cams

    def unnormalize(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def _plot_cams(self, save_path, inputs, cams, predictions, targets):
        n_cols = int(math.ceil(math.sqrt(self.num_plots / 2)))
        n_rows = int(math.ceil(self.num_plots / n_cols))

        plt.clf()
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=self.figsize, tight_layout=True
        )

        for i in range(n_rows * n_cols):
            if i < self.num_plots:
                inp = self.unnormalize(inputs[i])
                overlay_cam = overlay_mask(
                    to_pil_image(inp),
                    to_pil_image(cams[i], mode="F"),
                    alpha=0.5,
                )
                original_inputs = to_pil_image(inp)
                to_show = hstack_pil_image(original_inputs, overlay_cam)
                axes[i // n_cols][i % n_cols].imshow(to_show)
                axes[i // n_cols][i % n_cols].set_title(
                    f"Pred: {predictions[i]}, Target: {targets[i]}"
                )
            axes[i // n_cols][i % n_cols].axis("off")
        plt.savefig(save_path)

    def __call__(self, model, dataloader, predictions, targets, logger=None):
        cam_extractor = getattr(torchcam.cams, self.method)(
            model, input_shape=self.input_shape, target_layer=self.target_layer
        )

        # choose images that will be visualized
        assert predictions.shape == targets.shape
        deviation = ((predictions - targets) ** 2).mean(axis=1)
        if self.select_top_predictions:
            indices = np.argsort(deviation)[: self.num_plots]
        else:
            indices = np.argsort(deviation)[-self.num_plots :]

        # create new dataloader
        new_dataloader = make_subset_dataloader(
            dataloader, indices, self.num_plots
        )
        top_inputs, top_cams = self._extract_cam_on_first_batch(
            cam_extractor, model, new_dataloader
        )
        self._plot_cams(
            self.save_path,
            top_inputs,
            top_cams,
            predictions[indices],
            targets[indices],
        )

        return self.result


class RawSampleVisualizationHook(VisualizationHookBase):
    def __init__(
        self,
        dirpath,
        experiment_name,
        figsize=(20, 20),
        num_plots=15,
        select_top_predictions=True,
    ):
        suffix = "_top" if select_top_predictions else "_bottom"
        super().__init__(dirpath, experiment_name, figsize, suffix)
        self.num_plots = num_plots
        self.select_top_predictions = select_top_predictions

    def __call__(self, model, dataloader, predictions, targets, logger=None):
        # choose images that will be visualized
        assert predictions.shape == targets.shape
        deviation = ((predictions - targets) ** 2).mean(axis=1)
        if self.select_top_predictions:
            indices = np.argsort(deviation)[: self.num_plots]
        else:
            indices = np.argsort(deviation)[-self.num_plots :]

        # for plotting
        df = pd.DataFrame(
            {
                "inputs": np.array(dataloader.dataset.inputs)[indices],
                "predictions": predictions.flatten()[indices],
                "targets": targets.flatten()[indices],
            }
        )

        plt.clf()
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        pd.plotting.table(ax, df)
        plt.savefig(self.save_path)

        return self.result
