"""
Unit tests for MSTAR model architectures.
"""

import pytest
import torch
from mstar_atr.models.aconvnet import AConvNet
from mstar_atr.models.legacy_cnn import LegacyMSTARCNN
from mstar_atr.models.resnet import ResNetSAR, build_resnet18_sar


@pytest.mark.parametrize("size", [(88, 88), (100, 100), (128, 128)])
def test_aconvnet_forward_shapes(size):
    model = AConvNet(num_classes=10, in_channels=1)
    x = torch.randn(2, 1, *size)
    out = model(x)
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
    assert model.get_target_layer_for_gradcam() is not None


def test_resnet_sar_forward():
    model = build_resnet18_sar(num_classes=10, in_channels=1)
    x = torch.randn(2, 1, 88, 88)
    out = model(x)
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
    assert model.get_target_layer_for_gradcam() is not None


def test_legacy_cnn_forward():
    model = LegacyMSTARCNN(num_classes=10, in_channels=1)
    x = torch.randn(2, 1, 100, 100)
    out = model(x)
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
    assert model.get_target_layer_for_gradcam() is not None
