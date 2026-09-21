"""
Training and evaluation engine for MSTAR models.
"""

from mstar_atr.training.trainer import (
    evaluate_model,
    load_checkpoint,
    train_mstar_model,
)

__all__ = ["train_mstar_model", "evaluate_model", "load_checkpoint"]
