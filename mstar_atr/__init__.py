"""
MSTAR SAR Automatic Target Recognition (ATR) Suite.

A modern, production-grade deep learning toolkit for Synthetic Aperture Radar (SAR)
target classification, speckle filtering, interpretability (Grad-CAM), and tactical visualization.
"""

__version__ = "2.0.0"
__author__ = "Rohit Singh"

from mstar_atr.constants import (
    CLASSES,
    CLASS_ALIASES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    TARGET_METADATA,
)
from mstar_atr.filters.speckle import (
    amplitude_to_db,
    frost_filter,
    lee_filter,
    median_filter,
)
from mstar_atr.interpretation.gradcam import GradCAM, overlay_gradcam_on_sar
from mstar_atr.models.aconvnet import AConvNet
from mstar_atr.models.classical import ClassicalSARPipeline
from mstar_atr.models.legacy_cnn import LegacyMSTARCNN
from mstar_atr.models.resnet import ResNetSAR, build_resnet18_sar

__all__ = [
    "__version__",
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "TARGET_METADATA",
    "AConvNet",
    "ResNetSAR",
    "build_resnet18_sar",
    "LegacyMSTARCNN",
    "ClassicalSARPipeline",
    "lee_filter",
    "frost_filter",
    "median_filter",
    "amplitude_to_db",
    "GradCAM",
    "overlay_gradcam_on_sar",
]
