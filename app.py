from __future__ import annotations

import os
import gradio as gr
from fastapi import FastAPI
import torch
import pandas as pd

from mri.io.load_case import load_case
from mri.infer.predictor import load_bundle, predict_case, ModelBundle

# Feature flag (safe rollout)
USE_SERVICE = os.getenv("USE_SERVICE", "1") == "1"
_service = None

# MODEL_REF can be:
#   - directory with best_model.pth + policy artifacts (preferred)
#   - path to .pth checkpoint (legacy MVP or training best_model.pth)
DEFAULT_MODEL_REF = "/kaggle/working/mri-mvp/best_metric_model.pth"
MODEL_REF = os.getenv("MODEL_REF", os.getenv("MODEL_PATH", DEFAULT_MODEL_REF))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eager load (gives clearer error message early)
_bundle: ModelBundle | None = None
_bundle_error: str | None = None
try:
    _bundle = load_bundle(MODEL_REF, device)
except Exception as e:
    _bundle_error = str(e)


def run_inference(files):
    # Normalize uploaded file objects to filepaths
    files = files or []
    filepaths = []
    for f in files:
        if hasattr(f, "name"):
            filepaths.append(f.name)
        elif isinstance(f, dict) and "name" in f:
            filepaths.append(f["name"])
        else:
            filepaths.append(str(f))

    if _bundle is None:
        raise gr.Error(
            "Model bundle not loaded.\n"
            f"MODEL_REF: {MODEL_REF}\n"
            f"Error: {_bundle_error}"
        )

    slices, meta = load_case(filepaths)
    if not slices:
        raise gr.Error(f"No usable slices. Loader warnings: {meta.get('warnings', [])}")

    pil_slices = [s.image for s in slices]
    out = predict_case(_bundle, pil_slices, device=device)

    rows = []
    for r in out["per_slice"]:
        rows.append({
            "slice_index": r["slice_index"],
            "qc_ok": r["qc_ok"],
            "top_label": r.get("top_label"),
            "top_conf": r.get("top_conf"),
            "fg_ratio": r["qc"]["foreground_ratio"],
            "std": r["qc"]["std"],
            "qc_reasons": ",".join(r["qc"]["reasons"]),
        })
    df = pd.DataFrame(rows)

    case_label = out["case_probs"] if out["case_probs"] else {"ABSTAIN": 1.0}

    status = out.get("status", "ABSTAIN")
    abstain_type = out.get("abstain_type")
    abstain_reason = out.get("abstain_reason") or ""
    top_conf = out.get("top_conf") or 0.0
    p_in = out.get("p_in_domain")

    header = (
        f"{status}"
        f"{'/' + str(abstain_type) if abstain_type else ''}"
        f" | valid_slices={out['valid_slices']}"
        f" | agree={out.get('agree_rate', 0):.2f}"
        f" | top_conf_cal={float(top_conf):.3f}"
        f" | p_in_domain={float(p_in) if p_in is not None else 'na'}"
        f" | {abstain_reason}"
    )

    gallery = out["processed_images"][:32]
    return header, gallery, case_label, df, meta.get("warnings", [])


demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Files(label="Upload multiple images OR a ZIP containing a DICOM series"),
    outputs=[
        gr.Textbox(label="Case Status"),
        gr.Gallery(label="What the model saw (processed slices)", columns=4, rows=2),
        gr.Label(num_top_classes=4, label="Case Prediction / Abstain"),
        gr.Dataframe(label="Per-slice QC + Predictions"),
        gr.JSON(label="Loader Warnings"),
    ],
    title="MRI Tumor Classifier (Multi-Slice MVP)",
    description=(
        "Upload multiple image slices or a ZIP containing a DICOM series. "
        "Returns case-level prediction OR ABSTAIN (OOD/uncertain) with debug signals."
    )
)

demo.launch()
