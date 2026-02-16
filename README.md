# MRI Policy-Guarded Classifier

Policy-guarded MRI 4-class tumor classifier (ResNet18) with **calibration (temperature scaling)** and **abstain-on-OOD** guardrails (domain guard), shipped with a **Gradio demo**.

This repo documents and packages an end-to-end workflow:

1) Build deterministic data manifests + splits (Kaggle)  
2) Train baseline model + calibration + OOD policy artifacts (Kaggle)  
3) Run an inference app that loads the **same artifacts** and enforces the **same policy** (Gradio)

> **Key idea:** the demo UI must consume the *same* artifacts produced by training (checkpoint + calibration + policy + domain guard), otherwise you get label-order bugs and unsafe “forced predictions” on out-of-domain scans.

---

## What this project does

### In-domain classification (4 classes)
- glioma
- meningioma
- pituitary
- no tumor

### Safety / defensibility layers
- **Temperature scaling** for probability calibration
- **Confidence gate** (`tau_conf`) — abstain when calibrated confidence is too low
- **Domain guard (OOD detector)** using embeddings + logistic regression — abstain when `p_in_domain < tau_domain`
- Challenge/OOD evaluation artifacts (stroke/dementia/other MRI domains) used to validate that the model does **not** confidently hallucinate tumor classes off-distribution

---

## Repository layout (target)

