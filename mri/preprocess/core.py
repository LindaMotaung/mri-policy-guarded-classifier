from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


def crop_pad_square(img: Image.Image, margin: int = 8) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Robust ROI crop (foreground bbox) + square pad.
    Returns processed image + metadata.

    Why: Prevents "tiny brain" (too much black border) and prevents distortion.
    """
    meta: Dict[str, Any] = {"ok": True, "reason": None, "bbox": None}

    try:
        rgb = img.convert("RGB")
        gray = rgb.convert("L")
        arr = np.array(gray)

        # Dynamic threshold: more robust across varying contrasts than a hard-coded number
        thr = np.percentile(arr, 20)  # background-ish percentile
        thr = max(10, thr)            # avoid thresholds that are too low
        mask = arr > thr

        coords = np.argwhere(mask)
        if coords.size == 0:
            meta["ok"] = False
            meta["reason"] = "no_foreground_detected"
            return rgb, meta

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Add margin (clamped)
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(arr.shape[0], y1 + margin)
        x1 = min(arr.shape[1], x1 + margin)

        meta["bbox"] = (int(x0), int(y0), int(x1), int(y1))

        cropped = rgb.crop((x0, y0, x1, y1))

        # Pad to square
        w, h = cropped.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(cropped, ((s - w) // 2, (s - h) // 2))

        return canvas, meta

    except Exception:
        meta["ok"] = False
        meta["reason"] = "preprocess_exception"
        return img.convert("RGB"), meta
