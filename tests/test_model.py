"""Tests for the ResNet-18 smoke classifier model."""

from __future__ import annotations

import torch

from src.model.net import build_model


def test_model_output_shape():
    """Forward pass with a random 5-band tensor should produce a (B, 1) output."""
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 5, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_model_pretrained_loads():
    """Pretrained model should build without error."""
    model = build_model(pretrained=True)
    assert model.conv1.in_channels == 5
    assert model.fc.out_features == 1
