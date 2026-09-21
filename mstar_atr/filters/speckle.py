"""
Speckle reduction filters and dynamic range transforms for SAR (Synthetic Aperture Radar) images.

SAR imagery suffers from coherent speckle noise caused by constructive and destructive
interference among de-phased returns. Standard linear filtering blurs sharp radar scatterers.
Adaptive filters (Lee, Frost) adjust weights based on local homogeneity:
- Homogeneous clutter: apply strong smoothing to suppress speckle.
- Point scatterers / edges: preserve raw intensity to maintain vehicle geometry.
"""

from typing import Union
import numpy as np


def normalize_image(img: np.ndarray, to_uint8: bool = False) -> np.ndarray:
    """
    Normalizes an image array to [0.0, 1.0] float or [0, 255] uint8.
    """
    arr = np.asarray(img, dtype=np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val > min_val:
        norm = (arr - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(arr)
    if to_uint8:
        return np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm


def amplitude_to_db(
    img: np.ndarray,
    epsilon: float = 1e-6,
    clip_percentiles: tuple = (1.0, 99.5),
) -> np.ndarray:
    """
    Converts linear SAR amplitude to decibel (dB) scale for visual enhancement.

    dB = 20 * log10(amplitude + epsilon)

    Args:
        img: 2D or 3D numpy array of SAR amplitude values.
        epsilon: Small positive value to prevent log(0).
        clip_percentiles: Tuple of (low_percentile, high_percentile) for contrast stretching.

    Returns:
        Normalized [0, 1] array in decibel scale.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0, :, :]

    # Convert to decibels
    db = 20.0 * np.log10(np.maximum(arr, epsilon))

    # Contrast stretching using percentiles
    p_low, p_high = np.percentile(db, clip_percentiles)
    if p_high > p_low:
        stretched = np.clip((db - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        stretched = np.zeros_like(db)

    return stretched


def lee_filter(
    img: np.ndarray,
    window_size: int = 5,
    num_looks: float = 1.0,
) -> np.ndarray:
    """
    Applies the classic Lee speckle reduction filter for SAR imagery.

    The Lee filter uses the multiplicative noise model:
        R_hat = mean_I + W * (I - mean_I)
        W = max(0, 1 - (Cu^2 / Ci^2))
        where Cu = 1 / sqrt(num_looks), Ci = std_I / mean_I

    Args:
        img: 2D numpy array (single channel).
        window_size: Odd integer kernel size (e.g. 3, 5, 7).
        num_looks: Equivalent number of looks (ENL) of the SAR system.

    Returns:
        Filtered 2D numpy array with reduced speckle.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        # If multi-channel, filter channel-by-channel
        filtered_channels = [
            lee_filter(arr[:, :, c], window_size, num_looks)
            for c in range(arr.shape[2])
        ]
        return np.stack(filtered_channels, axis=2)

    h, w = arr.shape
    pad = window_size // 2
    padded = np.pad(arr, pad, mode="reflect")

    # Construct sliding window view
    shape = (h, w, window_size, window_size)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    local_mean = np.mean(windows, axis=(2, 3))
    local_var = np.var(windows, axis=(2, 3))

    cu = 1.0 / np.sqrt(max(num_looks, 0.1))
    ci = np.sqrt(local_var) / (local_mean + 1e-6)

    # Weight W
    w_weight = 1.0 - (cu ** 2) / (ci ** 2 + 1e-6)
    w_weight = np.clip(w_weight, 0.0, 1.0)

    result = local_mean + w_weight * (arr - local_mean)
    return np.maximum(result, 0.0)


def frost_filter(
    img: np.ndarray,
    window_size: int = 5,
    damping_factor: float = 2.0,
) -> np.ndarray:
    """
    Applies the Frost adaptive filter for SAR imagery.

    The Frost filter computes an exponentially damped convolution kernel
    where the damping parameter is proportional to the local coefficient of variation:
        m = damping_factor * (std_I / mean_I)
        K(x, y) = exp(-m * distance)

    Args:
        img: 2D numpy array.
        window_size: Odd integer kernel size (e.g. 5).
        damping_factor: Filter strength multiplier.

    Returns:
        Filtered 2D numpy array.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        filtered_channels = [
            frost_filter(arr[:, :, c], window_size, damping_factor)
            for c in range(arr.shape[2])
        ]
        return np.stack(filtered_channels, axis=2)

    h, w = arr.shape
    pad = window_size // 2
    padded = np.pad(arr, pad, mode="reflect")

    # Spatial distance matrix from window center
    y_coords, x_coords = np.mgrid[-pad : pad + 1, -pad : pad + 1]
    distances = np.sqrt(x_coords ** 2 + y_coords ** 2)

    shape = (h, w, window_size, window_size)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    local_mean = np.mean(windows, axis=(2, 3), keepdims=True)
    local_std = np.std(windows, axis=(2, 3), keepdims=True)
    cv = local_std / (local_mean + 1e-6)

    # Compute exponential decay weights for each pixel window
    # shape: (h, w, window_size, window_size)
    decay = np.exp(-damping_factor * cv * distances)
    decay_sum = np.sum(decay, axis=(2, 3), keepdims=True) + 1e-6
    weights = decay / decay_sum

    filtered = np.sum(windows * weights, axis=(2, 3))
    return np.maximum(filtered, 0.0)


def median_filter(img: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Applies standard median filtering for salt-and-pepper speckle spikes.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        filtered = [median_filter(arr[:, :, c], window_size) for c in range(arr.shape[2])]
        return np.stack(filtered, axis=2)

    h, w = arr.shape
    pad = window_size // 2
    padded = np.pad(arr, pad, mode="reflect")

    shape = (h, w, window_size, window_size)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    return np.median(windows, axis=(2, 3))
