"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for SAR Target Recognition.

Allows defense and radar analysts to visually audit which radar scattering centers,
dihedral reflectors, and shadow regions drove the deep learning classification decision.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM engine for PyTorch Convolutional Networks on SAR imagery.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_target_layer_for_gradcam"):
                target_layer = model.get_target_layer_for_gradcam()
            else:
                # Fallback: find the last Conv2d layer in the model
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if not conv_layers:
                    raise ValueError("No Conv2d layers found in model.")
                target_layer = conv_layers[-1]

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a 2D Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Shape (1, C, H, W)
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            Tuple of:
                - heatmap: 2D numpy array [H, W] normalized in [0, 1]
                - predicted_class: int
                - confidence: float
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            pred_class = int(torch.argmax(probs, dim=-1).item())
        else:
            pred_class = target_class

        score = logits[0, pred_class]
        score.backward(retain_graph=True)

        # Global average pool the gradients
        # activations: (1, K, H', W'), gradients: (1, K, H', W')
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only features that positively correlate with the target class

        # Upsample to original input resolution (H, W)
        _, _, h, w = input_tensor.shape
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_arr = cam_upsampled.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        max_val = np.max(cam_arr)
        min_val = np.min(cam_arr)
        if max_val > min_val:
            heatmap = (cam_arr - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(cam_arr)

        confidence = float(probs[0, pred_class].item())
        return heatmap, pred_class, confidence


def overlay_gradcam_on_sar(
    sar_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "inferno",
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on top of a single-channel SAR image.

    Args:
        sar_image: 2D numpy array [H, W] or 3D [H, W, 1]/[H, W, 3] in [0, 1] or [0, 255].
        heatmap: 2D numpy array [H, W] in [0, 1].
        alpha: Blend ratio for the heatmap (0.0 = only SAR, 1.0 = only heatmap).
        colormap_name: Matplotlib colormap ('inferno', 'jet', 'turbo', 'plasma').

    Returns:
        RGB uint8 numpy array [H, W, 3] suitable for display in PIL or Streamlit.
    """
    import matplotlib.cm as cm

    # Ensure SAR image is 2D float [0, 1]
    sar_2d = np.asarray(sar_image, dtype=np.float32)
    if sar_2d.ndim == 3:
        sar_2d = sar_2d[:, :, 0]
    if np.max(sar_2d) > 1.0:
        sar_2d = sar_2d / 255.0

    # Ensure heatmap matches dimensions
    if heatmap.shape != sar_2d.shape:
        from PIL import Image
        hm_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        hm_img = hm_img.resize((sar_2d.shape[1], sar_2d.shape[0]), resample=Image.BILINEAR)
        heatmap = np.array(hm_img, dtype=np.float32) / 255.0

    # Apply colormap to heatmap (modern matplotlib 3.8+ colormaps registry)
    import matplotlib
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps[colormap_name]
    else:
        cmap = cm.get_cmap(colormap_name)
    colored_cam = cmap(heatmap)[:, :, :3]  # (H, W, 3) float [0, 1]

    # Convert grayscale SAR to 3-channel
    sar_rgb = np.stack([sar_2d, sar_2d, sar_2d], axis=-1)

    # Alpha blending
    blended = (1.0 - alpha) * sar_rgb + alpha * colored_cam
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended
