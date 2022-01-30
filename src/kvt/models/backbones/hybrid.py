"""Ref: https://www.kaggle.com/debarshichanda/pytorch-hybrid-swin-transformer-cnn"""
import timm
import torch.nn as nn
from timm.models.vision_transformer_hybrid import HybridEmbed


class HybridTransformer(nn.Module):
    def __init__(
        self,
        backbone_name,
        embedder_name,
        num_classes=1000,
        pretrained=True,
        in_chans=3,
        image_size=224,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=in_chans
        )
        embedder = timm.create_model(
            embedder_name,
            in_chans=in_chans,
            features_only=True,
            out_indices=[2],
            pretrained=pretrained,
        )
        embed_dim = self.backbone.patch_embed.proj.out_channels
        self.backbone.patch_embed = HybridEmbed(
            embedder,
            img_size=image_size,
            embed_dim=embed_dim,
            in_chans=in_chans,
            patch_size=4,
        )
        self.backbone.reset_classifier(0)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, images):
        features = self.backbone(images)
        output = self.fc(features)
        return output


def hybrid_transformer(
    backbone_name="swin_base_patch4_window7_224",
    embedder_name="tf_efficientnet_b0",
    num_classes=1000,
    pretrained=False,
    in_chans=3,
    image_size=448,
    **kwargs,
):
    return HybridTransformer(
        backbone_name=backbone_name,
        embedder_name=embedder_name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_chans=in_chans,
        image_size=image_size,
    )
