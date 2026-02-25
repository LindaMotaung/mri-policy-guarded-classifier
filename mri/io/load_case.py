from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import os
import zipfile
import tempfile

import numpy as np
from PIL import Image
import pydicom


@dataclass
class SliceRecord:
    image: Image.Image
    meta: Dict[str, Any]


def _looks_like_dicom(path: str) -> bool:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return hasattr(ds, "SeriesInstanceUID") or hasattr(ds, "StudyInstanceUID") or hasattr(ds, "SOPClassUID")
    except Exception:
        return False


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(a.min())
        hi = float(a.max() if a.max() > a.min() else a.min() + 1.0)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def _load_dicom_series(paths: List[str]) -> Tuple[List[SliceRecord], List[str]]:
    warnings: List[str] = []
    items: List[SliceRecord] = []

    def _inst(p: str):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            v = getattr(ds, "InstanceNumber", None)
            return int(v) if v is not None else 10**9
        except Exception:
            return 10**9

    paths = sorted(paths, key=_inst)

    for p in paths:
        try:
            ds = pydicom.dcmread(p, force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            arr = ds.pixel_array

            photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
            if photo == "MONOCHROME1":
                arr = arr.max() - arr

            arr8 = _to_uint8(arr)
            img = Image.fromarray(arr8).convert("RGB")
            items.append(SliceRecord(image=img, meta={"path": p}))
        except Exception as e:
            warnings.append(f"Skipped slice (decode failed): {os.path.basename(p)} | {type(e).__name__}")

    if not items:
        warnings.append("No decodable DICOM pixel data found (might be compressed).")

    return items, warnings


def _load_images(paths: List[str]) -> Tuple[List[SliceRecord], List[str]]:
    warnings: List[str] = []
    out: List[SliceRecord] = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            out.append(SliceRecord(image=img, meta={"path": p}))
        except Exception:
            warnings.append(f"Could not open as image: {os.path.basename(p)}")
    return out, warnings


def load_case(filepaths: List[str]) -> Tuple[List[SliceRecord], Dict[str, Any]]:
    warnings: List[str] = []
    filepaths = [str(p) for p in (filepaths or [])]

    if not filepaths:
        return [], {"warnings": ["No files provided."]}

    # ZIP case (supports nested zips)
    if len(filepaths) == 1 and filepaths[0].lower().endswith(".zip"):
        zip_path = filepaths[0]

        with tempfile.TemporaryDirectory() as tmp:
            # extract outer zip
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp)

            # find nested zips (depth=1) and extract them too
            nested = []
            for root, _, files in os.walk(tmp):
                for fn in files:
                    if fn.lower().endswith(".zip"):
                        nested.append(os.path.join(root, fn))

            if nested:
                warnings.append(f"Found nested zip(s): {[os.path.basename(n) for n in nested[:5]]}")
                for nz in nested[:5]:
                    sub = os.path.join(tmp, "_nested_" + os.path.basename(nz))
                    os.makedirs(sub, exist_ok=True)
                    try:
                        with zipfile.ZipFile(nz, "r") as z2:
                            z2.extractall(sub)
                    except Exception as e:
                        warnings.append(f"Could not extract nested zip {os.path.basename(nz)}: {type(e).__name__}")

            # collect all non-zip files (including extracted nested contents)
            all_files = []
            for root, _, files in os.walk(tmp):
                for fn in files:
                    if fn.lower().endswith(".zip"):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        if os.path.getsize(p) < 128:
                            continue
                    except Exception:
                        continue
                    all_files.append(p)

            dicom_paths = [p for p in all_files if _looks_like_dicom(p)]

            if not dicom_paths:
                warnings.append("ZIP contained no DICOM files (detected by header).")
                warnings.append(f"Sample files: {[os.path.relpath(p, tmp) for p in all_files[:25]]}")
                return [], {"warnings": warnings}

            # IMPORTANT: load happens INSIDE tempdir context (prevents FileNotFoundError)
            slices, w2 = _load_dicom_series(dicom_paths)
            warnings.extend(w2)
            return slices, {"warnings": warnings}

    # Otherwise treat as regular images
    slices, w = _load_images(filepaths)
    warnings.extend(w)
    return slices, {"warnings": warnings}
