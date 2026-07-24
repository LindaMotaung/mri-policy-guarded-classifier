"""
Class Activation Mapping (CAM) for the ResNet18 classifier.

This model's head is exactly the architecture CAM (Zhou et al., 2016) was
designed for: global-average-pool -> single linear layer. That means the
heatmap can be computed directly from the last conv block's activations and
the FC layer's weights for the predicted class - no backward pass needed
(unlike Grad-CAM), so it's cheap enough to run on every valid slice alongside
normal inference.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class CAMExtractor:
    """Captures the last conv block's output (layer4) via a forward hook."""

    def __init__(self, model: nn.Module):
        self.activations: torch.Tensor | None = None
        self._hook = model.layer4.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()

    def remove(self):
        self._hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()


def compute_cam(activations: torch.Tensor, fc_weight: torch.Tensor, class_idx: int, out_size) -> np.ndarray:
    """
    activations: (1, C, H, W) from the last conv block (e.g. layer4 output, 512x7x7)
    fc_weight:   (num_classes, C) - the model's fc.weight
    class_idx:   which class's evidence map to compute
    out_size:    (width, height) to resize the CAM to (match the slice image)

    Returns a (H, W) float32 array in [0, 1].
    """
    weights = fc_weight[class_idx].detach().cpu()          # (C,)
    activ = activations[0].detach().cpu()                  # (C, H, W)
    cam = torch.einsum("c,chw->hw", weights, activ)
    cam = torch.relu(cam)

    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam_np = (cam.numpy() * 255).astype(np.uint8)
    resized = Image.fromarray(cam_np).resize(out_size, Image.BILINEAR)
    return np.array(resized).astype(np.float32) / 255.0


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Approximate matplotlib's 'jet' colormap without adding the dependency. values in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_cam_on_image(pil_img: Image.Image, cam_2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a CAM heatmap onto the (already square-cropped) slice image."""
    heatmap_rgb = (_jet_colormap(cam_2d) * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize(pil_img.size)
    return Image.blend(pil_img.convert("RGB"), heatmap_img, alpha=alpha)
