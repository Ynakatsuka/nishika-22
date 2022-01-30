import torch
from kvt.models.heads import DINOHead
from kvt.models.wrappers import MultiCropWrapper
from torch import nn

try:
    from lightly.models._momentum import _MomentumEncoderMixin
except ImportError:
    _MomentumEncoderMixin = object


class DINOModel(nn.Module, _MomentumEncoderMixin):
    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int = 32,
        out_dim: int = 128,
        m: float = 0.999,
        batch_shuffle: bool = False,
        use_bn_in_head: bool = True,
        norm_last_layer: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.projection_head = None

        # initialize momentum features and momentum projection head
        self._init_momentum_encoder()

        # multi-crop wrapper handles forward with inputs of different resolutions
        self.backbone = MultiCropWrapper(
            backbone, DINOHead(num_ftrs, out_dim, use_bn_in_head)
        )
        self.momentum_backbone = MultiCropWrapper(
            self.momentum_backbone,
            DINOHead(
                num_ftrs,
                out_dim,
                use_bn=use_bn_in_head,
                norm_last_layer=norm_last_layer,
            ),
        )

        self.m = m
        self.batch_shuffle = batch_shuffle

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x0: torch.Tensor, x1: torch.Tensor = None):
        self._momentum_update(self.m)

        # forward pass of first input x0
        out0 = self.backbone(x0).flatten(start_dim=1)

        # return out0 if x1 is None
        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():
            # shuffle for batchnorm
            if self.batch_shuffle:
                x1, shuffle = self._batch_shuffle(x1)

            # run x1 through momentum encoder
            out1 = self.momentum_backbone(x1).flatten(start_dim=1)

            # unshuffle for batchnorm
            if self.batch_shuffle:
                out1 = self._batch_unshuffle(out1, shuffle)

        return out0, out1
