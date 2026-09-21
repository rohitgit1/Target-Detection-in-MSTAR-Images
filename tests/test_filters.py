"""
Unit tests for SAR speckle filters and dynamic range transforms.
"""

import numpy as np
import pytest
from mstar_atr.filters.speckle import (
    amplitude_to_db,
    frost_filter,
    lee_filter,
    median_filter,
    normalize_image,
)


def test_normalize_image():
    arr = np.array([[10.0, 20.0], [30.0, 40.0]])
    norm = normalize_image(arr)
    assert norm.min() == 0.0
    assert norm.max() == 1.0

    u8 = normalize_image(arr, to_uint8=True)
    assert u8.dtype == np.uint8
    assert u8.min() == 0
    assert u8.max() == 255


def test_amplitude_to_db():
    arr = np.random.uniform(1.0, 100.0, size=(64, 64)).astype(np.float32)
    db = amplitude_to_db(arr)
    assert db.shape == (64, 64)
    assert 0.0 <= db.min() <= 0.1
    assert 0.9 <= db.max() <= 1.0


def test_lee_filter():
    arr = np.random.uniform(10.0, 50.0, size=(32, 32)).astype(np.float32)
    filtered = lee_filter(arr, window_size=5)
    assert filtered.shape == (32, 32)
    assert np.all(filtered >= 0.0)


def test_frost_filter():
    arr = np.random.uniform(10.0, 50.0, size=(32, 32)).astype(np.float32)
    filtered = frost_filter(arr, window_size=5)
    assert filtered.shape == (32, 32)
    assert np.all(filtered >= 0.0)


def test_median_filter():
    arr = np.random.uniform(10.0, 50.0, size=(32, 32)).astype(np.float32)
    filtered = median_filter(arr, window_size=3)
    assert filtered.shape == (32, 32)
