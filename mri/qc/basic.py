from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

def qc_slice(img: Image.Image, min_foreground_ratio: float = 0.05, min_std: float = 5.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Basic QC to reject garbage slices before the model sees them.

    Added:
    - grayscale_score gate: rejects colorful screenshots/photos
    """
    rgb = img.convert("RGB")
    gray = rgb.convert("L")
    arr = np.array(gray).astype(np.float32)

    # Existing QC
    fg_ratio = float((arr > 10).mean())
    std = float(arr.std())
    mean = float(arr.mean())

    # NEW: "Is this grayscale-like?"
    rgb_arr = np.array(rgb).astype(np.float32)  # HxWx3
    # per-pixel channel spread, then average
    grayscale_score = float(np.mean(np.std(rgb_arr, axis=2)))  # 0 for perfect grayscale

    ok = True
    reasons = []

    if fg_ratio < min_foreground_ratio:
        ok = False
        reasons.append("too_much_background")

    if std < min_std:
        ok = False
        reasons.append("low_contrast")

    # Tune threshold: start strict to block screenshots
    # Typical MRI grayscale_score is near ~0–2; screenshots much higher.
    if grayscale_score > 3.0:
        ok = False
        reasons.append("not_grayscale_like")

    return ok, {
        "foreground_ratio": fg_ratio,
        "mean": mean,
        "std": std,
        "grayscale_score": grayscale_score,
        "reasons": reasons,
    }
