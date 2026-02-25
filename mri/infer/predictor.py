"""
Predictor + policy engine

Goals:
- Accept *either* a raw state_dict checkpoint (legacy MVP) OR the training notebook's dict checkpoint.
- Drive label order, temperature scaling, and abstain/OOD policy from training artifacts when available.
- Always return a structured result: ACCEPT / ABSTAIN_* with reason codes (never "no output").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from mri.preprocess.core import crop_pad_square
from mri.qc.basic import qc_slice


IMG_SIZE = 224

# --- safe legacy fallback ONLY (used when no label_map is present) ---
LEGACY_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


@dataclass(frozen=True)
class DomainGuard:
    """Binary logistic regression p(in_domain) = sigmoid(w·z + b)."""
    w: torch.Tensor  # (D,) on CPU
    b: float         # scalar

    def score(self, z: torch.Tensor) -> float:
        # z: (D,) CPU tensor
        x = float(torch.dot(self.w, z).item() + self.b)
        p = 1.0 / (1.0 + float(torch.exp(torch.tensor(-x)).item()))
        return float(p)


@dataclass
class ModelBundle:
    model: nn.Module
    feat_extractor: nn.Module
    labels: List[str]                  # index -> label string
    label_map_raw: Optional[Dict[str, str]]
    num_classes: int
    temperature: float                 # T (>= 0.05)
    tau_conf: Optional[float]
    tau_domain: Optional[float]
    domain_guard: Optional[DomainGuard]
    source_dir: str


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Support DataParallel checkpoints
    return {k.replace("module.", ""): v for k, v in sd.items()}


def _resolve_bundle_dir(model_ref: str) -> Tuple[str, str]:
    """
    Returns (bundle_dir, ckpt_path).
    - If model_ref is a directory, look for best_model.pth inside.
    - If model_ref is a file, treat its parent as bundle_dir.
    """
    if os.path.isdir(model_ref):
        bundle_dir = model_ref
        ckpt_path = os.path.join(bundle_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            # allow alternate naming if user copied it
            alt = os.path.join(bundle_dir, "mri_resnet18_baseline_best.pth")
            if os.path.exists(alt):
                ckpt_path = alt
        return bundle_dir, ckpt_path

    bundle_dir = os.path.dirname(model_ref) or "."
    return bundle_dir, model_ref


def load_bundle(model_ref: str, device: torch.device) -> ModelBundle:
    """
    Loads a model bundle.
    model_ref can be:
      - path to .pth (legacy MVP state_dict OR training dict checkpoint)
      - directory containing best_model.pth + policy artifacts

    Policy artifacts (optional but preferred):
      - temperature_scaling.json   {"temperature": T, ...}
      - final_policy_config.json   {"tau_conf": ..., "tau_domain": ..., "temperature_T": ...}
      - domain_guard_lr.npz        {"coef": (1,D), "intercept": (1,)}  OR {"w": (D,), "b": ()}
    """
    if not os.path.exists(model_ref):
        raise RuntimeError(f"Model reference not found: {model_ref}")

    bundle_dir, ckpt_path = _resolve_bundle_dir(model_ref)

    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    # Sanity: avoid tiny invalid files (e.g., pointer)
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    if size_mb < 10:
        raise RuntimeError(
            f"Checkpoint looks invalid (too small: {size_mb:.2f} MB): {ckpt_path}"
        )

    raw = torch.load(ckpt_path, map_location="cpu")

    # Determine checkpoint shape
    if isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]
        label_map = raw.get("label_map")  # often {0:"glioma",...} but JSON may coerce keys to str elsewhere
        num_classes = int(raw.get("num_classes", 4))
    elif isinstance(raw, dict):
        # Legacy: assume it's a raw state_dict
        sd = raw
        label_map = None
        num_classes = 4
    else:
        raise RuntimeError(f"Unsupported checkpoint format: type={type(raw)}")

    sd = _clean_state_dict(sd)

    # Build model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    # Feature extractor = penultimate pooled features
    feat_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    # Labels from training artifact if possible
    labels: List[str]
    label_map_raw: Optional[Dict[str, str]] = None

    if isinstance(label_map, dict) and len(label_map) == num_classes:
        # normalize keys to int order
        label_map_raw = {str(k): str(v) for k, v in label_map.items()}
        labels = [label_map_raw[str(i)] for i in range(num_classes)]
    else:
        labels = LEGACY_LABELS[:num_classes]

    # Load temperature
    T = 1.0
    ts = _read_json(os.path.join(bundle_dir, "temperature_scaling.json"))
    if isinstance(ts, dict) and "temperature" in ts:
        try:
            T = float(ts["temperature"])
        except Exception:
            T = 1.0
    T = float(max(0.05, min(T, 100.0)))

    # Load policy thresholds
    tau_conf = None
    tau_domain = None
    pol = _read_json(os.path.join(bundle_dir, "final_policy_config.json"))
    if isinstance(pol, dict):
        if "tau_conf" in pol:
            try:
                tau_conf = float(pol["tau_conf"])
            except Exception:
                tau_conf = None
        if "tau_domain" in pol:
            try:
                tau_domain = float(pol["tau_domain"])
            except Exception:
                tau_domain = None
        # If policy config includes temperature_T, prefer it over temperature_scaling.json
        if "temperature_T" in pol:
            try:
                T = float(pol["temperature_T"])
                T = float(max(0.05, min(T, 100.0)))
            except Exception:
                pass

    # Load domain guard weights (no sklearn dependency)
    domain_guard = None
    npz_path = os.path.join(bundle_dir, "domain_guard_lr.npz")
    if os.path.exists(npz_path):
        try:
            import numpy as np
            d = np.load(npz_path)
            if "coef" in d and "intercept" in d:
                coef = d["coef"].astype("float32")
                intercept = d["intercept"].astype("float32")
                w = torch.tensor(coef.reshape(-1), dtype=torch.float32)  # (D,)
                b = float(intercept.reshape(-1)[0])
                domain_guard = DomainGuard(w=w, b=b)
            elif "w" in d and "b" in d:
                w = torch.tensor(d["w"].astype("float32").reshape(-1), dtype=torch.float32)
                b = float(d["b"].reshape(-1)[0])
                domain_guard = DomainGuard(w=w, b=b)
        except Exception:
            domain_guard = None

    return ModelBundle(
        model=model,
        feat_extractor=feat_extractor,
        labels=labels,
        label_map_raw=label_map_raw,
        num_classes=num_classes,
        temperature=T,
        tau_conf=tau_conf,
        tau_domain=tau_domain,
        domain_guard=domain_guard,
        source_dir=bundle_dir,
    )


_val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _probs_from_logits(logits_1d: torch.Tensor) -> List[float]:
    p = torch.softmax(logits_1d, dim=-1).detach().cpu().tolist()
    return [float(x) for x in p]


def _embed_1d(feat_extractor: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: (1,3,H,W) on device
    with torch.no_grad():
        z = feat_extractor(x)  # (1,512,1,1)
    z = z.view(z.shape[0], -1)[0].detach().cpu()  # (512,) CPU
    return z


def predict_case(
    bundle: ModelBundle,
    slices: List[Image.Image],
    device: torch.device,
    min_valid_slices: int = 3,
    abstain_agree_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Multi-slice inference with policy:
      - QC-gate slices
      - aggregate logits + embeddings across valid slices
      - apply temperature scaling (if available)
      - apply policy: ABSTAIN if (p_in_domain < tau_domain) OR (max_prob_cal < tau_conf)
    """
    labels = bundle.labels
    k = bundle.num_classes

    processed_images: List[Image.Image] = []
    per_slice: List[Dict[str, Any]] = []
    logits_list: List[torch.Tensor] = []
    emb_list: List[torch.Tensor] = []

    for idx, img in enumerate(slices):
        proc, pmeta = crop_pad_square(img)
        processed_images.append(proc)

        ok, qcmeta = qc_slice(proc)

        record: Dict[str, Any] = {
            "slice_index": idx,
            "preprocess_ok": bool(pmeta.get("ok", True)),
            "preprocess_reason": pmeta.get("reason"),
            "bbox": pmeta.get("bbox"),
            "qc_ok": bool(ok),
            "qc": qcmeta,
            "top_label": None,
            "top_conf": None,
        }

        if not ok:
            per_slice.append(record)
            continue

        x = _val_tf(proc).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = bundle.model(x)[0]  # (K,)

        probs = _probs_from_logits(logits)
        top_i = int(torch.argmax(torch.tensor(probs)).item())

        record["probs"] = {labels[i]: float(probs[i]) for i in range(k)}
        record["top_label"] = labels[top_i]
        record["top_conf"] = float(probs[top_i])

        per_slice.append(record)
        logits_list.append(logits)

        # embedding (CPU)
        try:
            emb = _embed_1d(bundle.feat_extractor, x)
            emb_list.append(emb)
        except Exception:
            # Do not fail inference if embedding path breaks
            pass

    result: Dict[str, Any] = {
        "status": "ABSTAIN",
        "abstain_type": "QC",
        "abstain_reason": None,
        "case_prediction": None,
        "case_probs": None,
        "valid_slices": len(logits_list),
        "agree_rate": 0.0,
        "top_conf": None,
        "p_in_domain": None,
        "thresholds": {
            "tau_conf": bundle.tau_conf,
            "tau_domain": bundle.tau_domain,
            "temperature": bundle.temperature,
        },
        "per_slice": per_slice,
        "processed_images": processed_images,
    }

    if len(logits_list) < min_valid_slices:
        result["abstain_reason"] = f"too_few_valid_slices<{min_valid_slices}"
        return result

    # Aggregate logits (stable) and embed
    case_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # (K,)
    T = float(bundle.temperature) if bundle.temperature else 1.0
    case_probs_cal = _probs_from_logits(case_logits / T)

    top_i = int(torch.argmax(torch.tensor(case_probs_cal)).item())
    top_label = labels[top_i]
    top_conf = float(case_probs_cal[top_i])

    result["case_prediction"] = top_label
    result["case_probs"] = {labels[i]: float(case_probs_cal[i]) for i in range(k)}
    result["top_conf"] = top_conf

    # Disagreement: how many valid slices vote for the same top label?
    votes = [r.get("top_label") for r in per_slice if r.get("qc_ok") and r.get("top_label") is not None]
    agree = sum(1 for v in votes if v == top_label) / max(1, len(votes))
    result["agree_rate"] = float(agree)

    # Domain score (case-level) if we have a guard + embeddings
    if bundle.domain_guard is not None and len(emb_list) > 0:
        z_case = torch.stack(emb_list, dim=0).mean(dim=0)  # (D,) CPU
        p_in = bundle.domain_guard.score(z_case)
        result["p_in_domain"] = float(p_in)

    # Policy evaluation
    abstain_domain = False
    abstain_lowconf = False
    abstain_disagree = False

    if bundle.tau_domain is not None and result["p_in_domain"] is not None:
        abstain_domain = result["p_in_domain"] < float(bundle.tau_domain)

    if bundle.tau_conf is not None:
        abstain_lowconf = top_conf < float(bundle.tau_conf)

    # Optional extra guard (kept from legacy MVP): disagreement can abstain
    abstain_disagree = (agree < abstain_agree_threshold)

    if abstain_domain:
        result["status"] = "ABSTAIN"
        result["abstain_type"] = "OOD"
        result["abstain_reason"] = f"p_in_domain<{bundle.tau_domain:.3f}"
        result["case_prediction"] = None
        result["case_probs"] = {"ABSTAIN_OOD": 1.0}
        return result

    if abstain_lowconf or abstain_disagree:
        result["status"] = "ABSTAIN"
        result["abstain_type"] = "UNCERTAIN"
        reason_bits = []
        if abstain_lowconf:
            reason_bits.append(f"max_prob_cal<{bundle.tau_conf:.3f}")
        if abstain_disagree:
            reason_bits.append(f"agree<{abstain_agree_threshold:.2f}")
        result["abstain_reason"] = ",".join(reason_bits) if reason_bits else "uncertain"
        result["case_prediction"] = None
        result["case_probs"] = {"ABSTAIN_UNCERTAIN": 1.0}
        return result

    # Accept
    result["status"] = "ACCEPT"
    result["abstain_type"] = None
    result["abstain_reason"] = None
    return result
