"""
End-to-end pipeline tests: Synthetic data creation, training, Grad-CAM, and classical ML.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
from mstar_atr.data.dataset import (
    MSTARDataset,
    SARTransform,
    create_synthetic_mstar_dataset,
    get_mstar_dataloaders,
)
from mstar_atr.interpretation.gradcam import GradCAM, overlay_gradcam_on_sar
from mstar_atr.models.aconvnet import AConvNet
from mstar_atr.models.classical import ClassicalSARPipeline
from mstar_atr.training.trainer import train_mstar_model


@pytest.fixture(scope="module")
def temp_mstar_dir():
    temp_dir = tempfile.mkdtemp()
    create_synthetic_mstar_dataset(temp_dir, samples_per_class=3, image_size=(88, 88))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_loading(temp_mstar_dir):
    train_dir = os.path.join(temp_mstar_dir, "train")
    transform = SARTransform(output_size=(88, 88), is_training=True)
    dataset = MSTARDataset(train_dir, transform=transform)

    assert len(dataset) == 30  # 10 classes * 3 samples
    x, y = dataset[0]
    assert x.shape == (1, 88, 88)
    assert 0 <= y < 10


def test_training_and_checkpoint(temp_mstar_dir):
    output_dir = os.path.join(temp_mstar_dir, "ckpts")
    results = train_mstar_model(
        data_dir=temp_mstar_dir,
        model_name="aconvnet",
        epochs=2,
        batch_size=8,
        output_dir=output_dir,
        input_size=(88, 88),
        verbose=False,
    )
    assert os.path.isfile(results["best_checkpoint"])
    assert "best_accuracy" in results


def test_gradcam_overlay():
    model = AConvNet(num_classes=10, in_channels=1)
    x = torch.randn(1, 1, 88, 88)
    cam = GradCAM(model)
    heatmap, pred_class, conf = cam.generate_heatmap(x)

    assert heatmap.shape == (88, 88)
    assert 0 <= pred_class < 10

    raw_sar = np.random.uniform(0, 255, size=(88, 88)).astype(np.uint8)
    overlay = overlay_gradcam_on_sar(raw_sar, heatmap)
    assert overlay.shape == (88, 88, 3)
    assert overlay.dtype == np.uint8


def test_classical_pipeline():
    X = np.random.uniform(0, 255, size=(20, 64, 64)).astype(np.float32)
    y = np.array([i % 10 for i in range(20)])

    pipeline = ClassicalSARPipeline(classifier_type="svm", n_pca_components=5)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert preds.shape == (20,)
    score = pipeline.score(X, y)
    assert 0.0 <= score <= 1.0
