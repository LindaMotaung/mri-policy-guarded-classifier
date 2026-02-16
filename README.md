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

mri-policy-guarded-classifier/
notebooks/
01_data_artifacts_builder.ipynb
02_training_calibration_policy.ipynb
03_mvp_gradio_policy_ui.ipynb
src/ # (optional, recommended) inference package extracted from MVP notebook
mri/
...
artifacts/
runtime/ # the minimal “bundle contract” needed by the app (DO NOT commit weights to git)
reports/ # calibration/policy reports, figures, CSVs
docs/
HANDOVER.md # deep handover / architecture doc (to be written)
README.md
LICENSE

## Notebooks (canonical names)

These are the three notebooks in the pipeline. Use these names in GitHub under `notebooks/`:

1) **`01_data_artifacts_builder.ipynb`**  
   Builds canonical manifests and split CSVs from raw Kaggle datasets.
   - Outputs: `manifest_4class_images.csv`, `manifest_4class_npz.csv`, `manifest_challenge_non_tumor.csv`
   - Outputs: `split_train_images.csv`, `split_val_images.csv`, `split_test_images.csv`, `split_external_test_npz.csv`, `split_challenge_sampled.csv`

2) **`02_training_calibration_policy.ipynb`**  
   Trains the baseline ResNet18 classifier, then produces calibration + policy artifacts.
   - Trains: ResNet18 (ImageNet pretrained), 4-class head
   - Produces: `best_model.pth` (dict checkpoint), temperature scaling, reliability summaries
   - Produces: policy artifacts + domain guard (OOD) artifacts
   - Produces: a “model bundle” suitable for deployment

3) **`03_mvp_gradio_policy_ui.ipynb`**  
   Builds the inference package + Gradio UI and loads the **model bundle contract** to match training behavior:
   - Loads images / ZIP / DICOM series
   - QC + preprocessing + per-slice inference + aggregation
   - Enforces abstain policy: confidence + OOD domain guard
   - Launches Gradio reliably on Kaggle

---

## Model bundle contract (the critical interface)

All deployments (Kaggle MVP notebook, Hugging Face Space, future API service) load the same bundle directory.

### Required runtime files
- `best_model.pth`  
  A dict checkpoint containing `model_state_dict` and `label_map`.
- `temperature_scaling.json`  
  Contains the fitted temperature `T` used to calibrate logits.
- `final_policy_config.json`  
  Contains thresholds and policy settings (e.g. `tau_conf`, `tau_domain`, keep-rates).
- `domain_guard_lr.npz`  
  Logistic regression weights + preprocessing needed to compute `p_in_domain`.

### Nice-to-have (debug/audit)
- `env_snapshot.json`
- `calibration_metrics.json`
- `domain_guard_tau_sweep.csv`
- `policy_summary_v*.csv`
- `challenge_policy_outputs_v*.csv`
- `reliability_*.png`

> **Do not** hardcode label order in the UI. Always derive it from the `label_map` inside `best_model.pth`.

---

## Datasets and provenance (Kaggle)

Training inputs are built from multiple Kaggle datasets (2D images) and evaluated on external/OOD sources.

- In-domain (4-class images):  
  - `masoudnickparvar/brain-tumor-mri-dataset`  
  - `sabersakin/brainmri`
- External evaluation (NPZ):  
  - `muazalzoubi/brain-tumor-gliomameningiomapituitary-not-tumors`
- OOD / challenge domains (for abstain behavior auditing):  
  - stroke / dementia / normal MRI domains (various)

**Important:** This repo does not redistribute Kaggle datasets. You must attach them in Kaggle or supply your own equivalents.

---

## How the abstain policy works (high level)

For a case (single image, or aggregated across slices):

1) Run the classifier → get logits  
2) Apply temperature scaling → calibrated probabilities  
3) Compute:
   - `p_max` = max calibrated class probability
   - `p_in_domain` = domain guard probability using embeddings + LR

Decision:
- If `p_in_domain < tau_domain` → **ABSTAIN_OOD**
- Else if `p_max < tau_conf` → **ABSTAIN_LOW_CONF**
- Else → **ACCEPT** and return predicted tumor class + calibrated probs

This design intentionally prevents forced tumor labels on stroke/dementia or other non-training domains.

---

## Running on Kaggle (recommended workflow)

### A) Data artifacts notebook
Run `01_data_artifacts_builder.ipynb`, then publish the `data_artifacts/` folder as a Kaggle Dataset:
- Example: `kabomolefe/mri-data-artifacts-v1`

### B) Training + policy notebook
Attach the data artifacts dataset and run `02_training_calibration_policy.ipynb`.
Publish the resulting model bundle as a Kaggle Dataset:
- Example: `kabomolefe/mri-model-bundle-v1`

### C) MVP/Gradio notebook
Attach the model bundle dataset and run `03_mvp_gradio_policy_ui.ipynb`.
Set `MODEL_REF` to the mounted bundle path before importing the app.

---

## Running locally (future)
This repo is set up to become a standard Python package + Gradio app outside Kaggle.
Planned:
- extract `/src/mri` from the MVP notebook
- add a CLI to run inference against a bundle directory
- add minimal tests (bundle contract, label map consistency, smoke inference)

---

## Roadmap (next steps)
- [ ] Convert notebook-built MVP package into committed `src/mri/` modules
- [ ] Add `docs/HANDOVER.md` with architecture, decisions, and known issues
- [ ] Add Hugging Face deployment: Space + Model repo for the bundle
- [ ] Add evaluation harness for real-world scans (slice selection, better aggregation)
- [ ] Improve training protocol (epochs, augmentation, early stopping, per-domain validation)

---

## License
MIT (code only). Dataset licenses/terms are governed by their Kaggle sources.

---

## Author
Kabo Molefe  
GitHub: https://github.com/kablewithak