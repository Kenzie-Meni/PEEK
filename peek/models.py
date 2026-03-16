"""
Reusable model definitions for PEEK experiments.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision import models


class ResNet50SpatialBackbone(nn.Module):
    """
    ResNet50 trunk up to the final spatial feature block before global pooling.
    """

    def __init__(self, backbone: models.ResNet) -> None:
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class VGG16OneDenseNoPool(nn.Module):
    """
    VGG16 conv trunk -> Flatten -> Linear(num_classes).

    This keeps the final 7x7x512 convolutional feature block and replaces the
    default VGG classifier with a single dense head.
    """

    def __init__(
        self,
        num_classes: int,
        weights: Optional[Union[str, models.VGG16_Weights]] = "IMAGENET1K_V1",
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(weights, str):
            resolved_weights = getattr(models.VGG16_Weights, weights)
        else:
            resolved_weights = weights

        vgg = models.vgg16(weights=resolved_weights)
        self.features = vgg.features
        self.backbone = self.features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * 7 * 7, num_classes)
        self.head = self.fc

        if freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if x.shape[-2:] != (7, 7):
            raise ValueError(
                f"Expected 7x7 features from VGG16 backbone, got {tuple(x.shape[-2:])}. "
                "Use 224x224 inputs for this model."
            )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class ResNet50OneDenseNoPool(nn.Module):
    """
    ResNet50 trunk -> Flatten -> Linear(num_classes).

    This keeps the final 7x7x2048 residual feature block before avgpool.
    """

    def __init__(
        self,
        num_classes: int,
        weights: Optional[models.ResNet50_Weights] = models.ResNet50_Weights.IMAGENET1K_V2,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        resnet = models.resnet50(weights=weights)
        self.backbone = ResNet50SpatialBackbone(resnet)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(2048 * 7 * 7, num_classes)
        self.head = self.fc

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if x.shape[-2:] != (7, 7):
            raise ValueError(
                f"Expected 7x7 features from ResNet50 backbone, got {tuple(x.shape[-2:])}. "
                "Use 224x224 inputs for this model."
            )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class ConvNeXtBaseOneDenseNoPool(nn.Module):
    """
    ConvNeXt-Base trunk -> Flatten -> Linear(num_classes).

    This keeps the final 7x7x1024 spatial feature block before global pooling.
    """

    def __init__(
        self,
        num_classes: int,
        weights: Optional[models.ConvNeXt_Base_Weights] = models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        convnext = models.convnext_base(weights=weights)
        self.features = convnext.features
        self.backbone = self.features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(1024 * 7 * 7, num_classes)
        self.head = self.fc

        if freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if x.shape[-2:] != (7, 7):
            raise ValueError(
                f"Expected 7x7 features from ConvNeXt-Base backbone, got {tuple(x.shape[-2:])}. "
                "Use 224x224 inputs for this model."
            )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_imagenette_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Build an ImageNet-pretrained classifier adapted to Imagenette2.
    """
    name = model_name.lower()

    if name == "vgg16":
        weights = "IMAGENET1K_V1" if pretrained else None
        return VGG16OneDenseNoPool(
            num_classes=num_classes,
            weights=weights,
            dropout=dropout,
        )

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return ResNet50OneDenseNoPool(
            num_classes=num_classes,
            weights=weights,
            dropout=dropout,
        )

    if name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        return ConvNeXtBaseOneDenseNoPool(
            num_classes=num_classes,
            weights=weights,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported Imagenette model: {model_name}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    """
    Freeze or unfreeze the pretrained backbone while keeping the task head trainable.
    """
    if not hasattr(model, "backbone") or not hasattr(model, "head"):
        raise ValueError(
            f"Model '{model_name}' does not expose the expected backbone/head split."
        )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
    for parameter in model.head.parameters():
        parameter.requires_grad = True
