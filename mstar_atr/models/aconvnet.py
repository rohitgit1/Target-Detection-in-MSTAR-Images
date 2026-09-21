"""
A-ConvNet (All-Convolutional Network) for SAR Automatic Target Recognition (ATR).

Reference:
    Chen, S., Wang, H., Xu, F., & Jin, Y. Q. (2016).
    Target classification using the deep convolutional networks for SAR images.
    IEEE Transactions on Geoscience and Remote Sensing, 54(8), 4806-4817.

A-ConvNet replaces fully-connected layers with convolutional layers, dramatically reducing
the parameter count and preventing overfitting to SAR speckle noise.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AConvNet(nn.Module):
    """
    All-Convolutional Network for MSTAR SAR ATR.

    Architecture:
        Input: (B, 1, H, W)  [Standard benchmark uses 88x88 or 128x128]
        Conv1: 1 -> 16, kernel 5x5, ReLU -> MaxPool 2x2
        Conv2: 16 -> 32, kernel 5x5, ReLU -> MaxPool 2x2
        Conv3: 32 -> 64, kernel 6x6, ReLU -> MaxPool 2x2
        Conv4: 64 -> 128, kernel 5x5, ReLU, Dropout(0.5)
        Conv5: 128 -> num_classes, kernel 3x3
        AdaptiveAvgPool2d((1, 1)) to ensure dimension invariance
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4 (Replaces FC1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Layer 5 (Replaces FC2 / Classifier)
        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=0)

        # Adaptive pooling to guarantee (B, num_classes) regardless of input resolution
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate feature maps for Grad-CAM."""
        # A-ConvNet requires at least 88x88 input due to successive valid convolutions
        if x.shape[-2] < 88 or x.shape[-1] < 88:
            x = F.interpolate(x, size=(88, 88), mode="bilinear", align_corners=False)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(F.relu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, in_channels, H, W)
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.forward_features(x)
        logits_map = self.conv5(features)
        pooled = self.global_pool(logits_map)
        return torch.flatten(pooled, 1)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the penultimate convolutional layer for Grad-CAM visual explanation."""
        return self.conv4
