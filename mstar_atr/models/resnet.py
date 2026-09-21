"""
Modern ResNet architecture adapted for Single-Channel SAR Imagery.

Unlike standard RGB ImageNet ResNets expecting 3 channels, ResNetSAR natively supports
1-channel radar reflectivity inputs with custom stem scaling and dropout regularization.
"""

from typing import Optional, Type, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard 2-layer residual block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetSAR(nn.Module):
    """
    ResNet adapted for Synthetic Aperture Radar (SAR) imagery.
    """

    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],  # ResNet-18 by default
        num_classes: int = 10,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._curr_channels = 64

        # Radar input stem (preserves spatial resolution better than aggressive 7x7 strided conv)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._curr_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._curr_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers_list = [ResidualBlock(self._curr_channels, out_channels, stride, downsample)]
        self._curr_channels = out_channels
        for _ in range(1, blocks):
            layers_list.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps before pooling."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        dropped = self.dropout(flattened)
        return self.fc(dropped)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the last convolutional layer in layer4 for Grad-CAM."""
        return self.layer4[-1].conv2


def build_resnet18_sar(num_classes: int = 10, in_channels: int = 1) -> ResNetSAR:
    """Builds a ResNet-18 model customized for SAR target recognition."""
    return ResNetSAR(layers=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)
