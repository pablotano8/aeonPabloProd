# Brain Tumor Analysis from FLAIR MRI

Automated brain tumor segmentation and survival prediction from single-modality FLAIR MRI scans.

## Quick Start

```bash
# Install
conda env create -f environment.yml
conda activate aeon

# Run the playground (web UI)
python playground/server.py
# Open http://localhost:8000
```

Upload a FLAIR NIfTI scan, enter patient age and extent of resection, and get:
- Tumor segmentation with adjustable sensitivity
- Uncertainty map
- Survival prediction with risk group and prediction interval

## Segmentation Performance

| Metric | Value |
|--------|-------|
| Dice coefficient | 0.9148 |
| Sensitivity | 0.9000 |
| Precision | 0.9306 |

Evaluated on 74 held-out validation subjects.

**High Sensitivity mode** lowers the threshold from 0.5 to 0.001, increasing sensitivity from 90.0% to 99.0% at the cost of Dice dropping from 0.915 to 0.752.

## Survival Performance

| Metric | Value |
|--------|-------|
| C-index (5-fold CV, 20 seeds) | 0.672 ± 0.006 |
| C-index (predicted segmentation) | 0.657 ± 0.006 |
| Calibrated MAE | 245 days |

When using predicted segmentation from nnU-Net (as in the playground) instead of ground-truth masks, C-index drops by ~0.015. The gap is driven by `tumor_min` (GT-vs-predicted correlation: 0.48) — boundary differences change the minimum intensity within the tumor region. Other features are highly correlated (dist_from_center: 0.99, tumor_intensity_ratio: 0.97).

## Project Structure

```
├── checkpoints/               # Local model artifacts (gitignored, not committed)
│   ├── segmentation/          # nnU-Net model exported after training
│   │   ├── fold_all/checkpoint_final.pth
│   │   ├── dataset.json
│   │   ├── dataset_fingerprint.json
│   │   └── plans.json
│   └── survival/              # CoxPH model + calibrator
│       ├── coxph_model.pkl
│       ├── lognormal_aft.pkl
│       ├── scaler.pkl
│       └── calibrator.pkl
├── src/
│   ├── segmentation/
│   │   ├── train.py           # Train nnU-Net (wraps CLI pipeline)
│   │   ├── evaluate.py        # Evaluate on validation set
│   │   ├── infer.py           # Single-scan inference + survival
│   │   ├── predict.py         # Batch prediction for a split
│   │   └── convert_data.py    # Convert data to nnU-Net format
│   └── survival/
│       ├── features.py        # Radiomics feature extraction
│       ├── train_cox.py       # Train CoxPH (production)
│       ├── train_aft.py       # Train LogNormal AFT (alternative)
│       └── evaluate.py        # Survival model evaluation
├── playground/
│   ├── server.py              # Local web server
│   └── index.html             # NiiVue-based viewer
├── environment.yml
├── deploy_modal.sh           # Modal deploy with checkpoint bootstrap
└── README.md
```

## Segmentation

### Architecture

**nnU-Net v2** (3d_fullres) — a self-configuring 3D segmentation framework.

- PlainConvUNet, 6 stages, features [32, 64, 128, 256, 320, 320]
- InstanceNorm3d, Dice + CrossEntropy loss with deep supervision
- Input: single-channel FLAIR volume (1mm isotropic), patch size 128³
- Augmentation: rotation, scaling, elastic deformation, gamma correction, mirroring

### Training

```bash
# Single command to train from scratch
python src/segmentation/train.py

# Or with more epochs
python src/segmentation/train.py --epochs 250
```

This runs the full nnU-Net pipeline: data conversion, preprocessing, training, and copies the checkpoint to `checkpoints/segmentation/`. That directory is intentionally gitignored, so each environment keeps its own local model artifacts. Requires ~8GB VRAM. Training takes ~1.5 hours on an RTX 5090 at 100 epochs.

### Checkpoints

The app auto-downloads checkpoints when they are missing. This happens when you start the
playground, run inference, run evaluation, or deploy with `./deploy_modal.sh`. If the files
already exist under `checkpoints/`, nothing is downloaded.

### Evaluation

```bash
python src/segmentation/evaluate.py
```

### Inference

```bash
# Single scan
python src/segmentation/infer.py --input /path/to/flair.nii --output_dir output/

# With survival prediction
python src/segmentation/infer.py --input /path/to/flair.nii --output_dir output/ \
    --age 55 --eor GTR

# Batch prediction for a split
python src/segmentation/predict.py --split validation
```

Outputs: `segmentation.nii.gz` (binary mask, threshold 0.5), `segmentation_high_sens.nii.gz` (threshold 0.001, 99% sensitivity), `tumor_probability.nii.gz`, `uncertainty.nii.gz`.

## Survival Prediction

Given a FLAIR MRI scan and its tumor segmentation, the survival model predicts expected time-to-death for glioblastoma patients: a point estimate (median predicted survival in days), a prediction interval, and a risk group assignment.

### Model

**Cox Proportional Hazards** (CoxPH), implemented via [scikit-survival](https://scikit-survival.readthedocs.io/), with L2 regularization alpha = 0.1.

```
h(t | X) = h_0(t) * exp(beta_1 * age + beta_2 * dist_from_center + ... + beta_6 * eor_str)
```

CoxPH was chosen over LogNormal AFT (+0.003 C-index but requires parametric assumption), Random Survival Forests, Gradient Boosted Survival Analysis, XGBoost, and DeepSurv across 73 experiments.

### Features

6 features selected through greedy forward selection from 24 candidates. Each retained only if it improved 5-fold CV C-index consistently across 50 random seeds with p < 0.05.

| Feature | Coefficient | HR | Description |
|---------|------------|-----|-------------|
| `age` | +0.550 | 1.73 | Patient age at diagnosis. Strongest predictor (-0.074 C-index if removed). |
| `dist_from_center` | -0.249 | 0.78 | Tumor centroid distance from brain center (normalized). Central tumors = worse prognosis. |
| `tumor_intensity_ratio` | -0.222 | 0.80 | Mean FLAIR intensity in tumor / mean brain intensity. |
| `tumor_min` | -0.154 | 0.86 | Minimum FLAIR intensity in tumor. Low = necrosis = aggressive. |
| `extent_x` | +0.125 | 1.13 | Lateral tumor spread (mm). May indicate midline crossing. |
| `eor_str` | -0.113 | 0.89 | Subtotal resection indicator (binary). |

### Calibration

Isotonic calibration corrects systematic underestimation. Fitted via 5-fold out-of-fold predictions. Reduces MAE from 247 to 245 days and shifts mean prediction from 352 to 452 days (actual mean: 446).

### Risk Stratification

| Risk Group | Criterion | N | Median Survival | 1-year Survival |
|-----------|-----------|---|----------------|-----------------|
| **High Risk** | < 300 days | 24 | 126 days | 4.2% |
| **Medium Risk** | 300–450 days | 127 | 329 days | 42.5% |
| **Low Risk** | > 450 days | 85 | 495 days | 75.3% |

All group separations are statistically significant (log-rank p < 0.01).

### Training

```bash
# Train CoxPH (production model)
PYTHONPATH=src/survival python src/survival/train_cox.py --use_all_data

# Train LogNormal AFT (alternative, +0.003 C-index but parametric assumption)
PYTHONPATH=src/survival python src/survival/train_aft.py --use_all_data

# Evaluate
PYTHONPATH=src/survival python src/survival/evaluate.py \
    --model_path checkpoints/survival/coxph_model.pkl
```

### Limitations

- **Age dominates**: all five imaging features combined contribute only +0.025 C-index.
- **No molecular markers**: MGMT methylation and IDH mutation are stronger prognostic factors but unavailable from imaging.
- **Single modality**: using only FLAIR; T1ce and T2 could provide additional signal.
- **Small dataset**: N=236 limits model complexity.

## Data Format

```
data/
├── survival_info.csv          # Age, survival days, extent of resection
├── train/
│   └── {id}/
│       ├── flair.nii
│       └── seg.nii            # Binary mask (0=background, 1=tumor)
└── validation/
    └── {id}/
        ├── flair.nii
        └── seg.nii
```

## Requirements

- Python 3.11+
- PyTorch 2.x with CUDA
- nnU-Net v2 (`pip install nnunetv2`)
- MONAI, scikit-survival, lifelines, nibabel, scipy

```bash
conda env create -f environment.yml
conda activate aeon
```
