"""
Modern PyTorch training and evaluation loop for SAR Automatic Target Recognition.
"""

from typing import Dict, Optional, Tuple, Union
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mstar_atr.constants import CLASSES, DEFAULT_CROP_SIZE
from mstar_atr.data.dataset import get_mstar_dataloaders
from mstar_atr.models.aconvnet import AConvNet
from mstar_atr.models.legacy_cnn import LegacyMSTARCNN
from mstar_atr.models.resnet import build_resnet18_sar


def build_model(
    model_name: str = "aconvnet",
    num_classes: int = len(CLASSES),
    in_channels: int = 1,
) -> nn.Module:
    """Instantiates a model architecture by name."""
    name = model_name.lower().replace("-", "").replace("_", "")
    if name in ["aconvnet", "aconv"]:
        return AConvNet(num_classes=num_classes, in_channels=in_channels)
    elif name in ["resnet", "resnet18", "resnetsar"]:
        return build_resnet18_sar(num_classes=num_classes, in_channels=in_channels)
    elif name in ["legacy", "legacycnn", "cnn"]:
        return LegacyMSTARCNN(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: aconvnet, resnet18, legacy_cnn")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluates model on test/validation DataLoader.

    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_targets)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc, np.array(all_preds), np.array(all_targets)


def train_mstar_model(
    data_dir: str,
    model_name: str = "aconvnet",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
    output_dir: str = "checkpoints",
    input_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
    verbose: bool = True,
) -> Dict[str, Union[float, str, list]]:
    """
    Trains a SAR target recognition model with Cosine Annealing learning rate schedule.
    """
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if verbose:
        print(f"[MSTAR Training] Device: {device} | Architecture: {model_name} | Epochs: {epochs}")

    train_loader, test_loader = get_mstar_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        input_size=input_size,
    )

    model = build_model(model_name=model_name, num_classes=len(CLASSES), in_channels=1)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(output_dir, exist_ok=True)
    best_acc = 0.0
    best_checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pt")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping to stabilize radar speckle spike gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, device)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "classes": CLASSES,
                    "input_size": input_size,
                },
                best_checkpoint_path,
            )

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )

    elapsed = time.time() - start_time
    if verbose:
        print(f"[MSTAR Training] Completed in {elapsed:.1f}s. Best Val Accuracy: {best_acc * 100:.2f}%")
        print(f"[MSTAR Training] Checkpoint saved to: {best_checkpoint_path}")

    return {
        "best_accuracy": best_acc,
        "best_checkpoint": best_checkpoint_path,
        "history": history,
        "elapsed_seconds": elapsed,
    }


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """Loads model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "aconvnet")
    model = build_model(model_name=model_name, num_classes=len(ckpt.get("classes", CLASSES)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
