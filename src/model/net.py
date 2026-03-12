"""ResNet-18 wrapper with 5-band input adapter for binary smoke classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_input_bands: int = 5, pretrained: bool = True) -> nn.Module:
    """Return a ResNet-18 adapted for *num_input_bands* with a binary head.

    * ``conv1`` is replaced to accept *num_input_bands* channels.
      The first 3 filters are copied from ImageNet weights; extra channels
      use Kaiming-normal initialization.
    * ``fc`` is replaced with a single-output linear layer (logit).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # --- Input adapter ---
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        num_input_bands,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            # Copy pretrained RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Kaiming init for extra channels
            nn.init.kaiming_normal_(
                new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu"
            )
    model.conv1 = new_conv

    # --- Binary classification head ---
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
