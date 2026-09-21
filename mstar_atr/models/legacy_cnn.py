"""
PyTorch reproduction of the author's original 3-layer CNN architecture from CNN_Implementation.ipynb.

Preserved for heritage, backwards benchmarking, and educational comparison against modern SOTA (A-ConvNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyMSTARCNN(nn.Module):
    """
    Original 2021 CNN Architecture:
        Conv1: (in_channels -> 8, kernel 5x5) -> ReLU -> MaxPool 2x2
        Conv2: (8 -> 16, kernel 5x5) -> ReLU -> MaxPool 2x2
        Conv3: (16 -> 32, kernel 5x5) -> ReLU -> MaxPool 2x2
        Dense1: (Flatten -> 120) -> ReLU
        Dense2: (120 -> 84) -> ReLU
        Dense3: (84 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pool ensures consistent vector dimension regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = self.adaptive_pool(features)
        flat = torch.flatten(pooled, 1)
        x = F.relu(self.fc1(flat))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_target_layer_for_gradcam(self) -> nn.Module:
        return self.conv3
