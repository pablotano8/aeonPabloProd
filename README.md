# Brain Tumor Segmentation and Survival from FLAIR MRI

Automated brain tumor segmentation and survival prediction from single-modality FLAIR MRI scans.

## Quick Start

```bash
# Install
conda env create -f environment.yml
conda activate aeon

# Run the playground (web UI)
python playground/server.py

```

Upload a FLAIR .nii scan, enter patient age and extent of resection, and get:
- Tumor segmentation
- Uncertainty map
- High-sensitivity map
- Survival prediction interval

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
| C-index (nested 5-fold CV x 5 seeds) | 0.67 ± 0.03 |
| C-index (leave-one-out CV, N=236) | 0.67 ± 0.03|
| MAE (leave-one-out CV) | 235 days |

These values correspond to using predicted segmentation from nnU-Net (as in the playground). When using ground-truth masks, C-index increases by ~0.015. The gap is driven by the `tumor_min` feature (GT-vs-predicted correlation: 0.48). Other features are highly correlated (dist_from_center: 0.99, tumor_intensity_ratio: 0.97).

## Project Structure

```
├── checkpoints/               # Local model artifacts (gitignored)
│   ├── segmentation/          # nnU-Net model exported after training
│   │   ├── fold_all/checkpoint_final.pth
│   │   ├── dataset.json
│   │   ├── dataset_fingerprint.json
│   │   └── plans.json
│   └── survival/              # CoxPH model
│       ├── coxph_model.pkl
│       └── scaler.pkl
├── src/
│   ├── checkpoint_bootstrap.py    # Auto-download checkpoints if missing
│   ├── checkpoint_source.py       # Checkpoint archive URL
│   ├── segmentation/
│   │   ├── train.py               # Train nnU-Net (wraps CLI pipeline)
│   │   ├── evaluate.py            # Evaluate on validation set
│   │   ├── infer.py               # Single-scan inference
│   │   ├── predict.py             # Batch prediction for a split
│   │   └── convert_data.py        # Convert data to nnU-Net format
│   └── survival/
│       ├── predict.py             # Survival inference (CoxPH prediction)
│       ├── features.py            # Radiomics feature extraction
│       ├── train_cox.py           # Train CoxPH (production)
│       └── evaluate.py            # Survival model evaluation
├── playground/
│   ├── server.py              # Local web server
│   └── index.html             # NiiVue-based viewer
├── environment.yml
├── deploy_modal.sh           # Modal deploy with checkpoint bootstrap
└── README.md
```

## Segmentation

### Architecture

**nnU-Net v2** (3d_fullres): self-configuring 3D segmentation framework.

The auto-configuration chosen by nn-Unet pipeline appears in `checkpoints/segmentation/plans.json`, after running the segementaiton training pipeline locally (`src/segmentation/train.py`) or after auto-downloading checkpoints when running the playground. This configuration is based on the structure of the training data, auto-parsed in `checkpoints/segmentation/dataset_fingerprint.json`. For our dataset, some aspects of the configuration are:

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

**Cox Proportional Hazards** (CoxPH), implemented via [scikit-survival](https://scikit-survival.readthedocs.io/), with L2 regularization alpha = 3.0. Trained on all available data (train + validation).

```
h(t | X) = h_0(t) * exp(beta_1 * age + beta_2 * dist_from_center + ... + beta_5 * eor_str)
```

CoxPH was chosen over LogNormal AFT, Random Survival Forests, Gradient Boosted Survival Analysis, XGBoost, and DeepSurv.

### Features

5 features selected through greedy forward selection (nested 5-fold CV x 5 seeds) from 24 candidates:

| Feature Added | C-index (nested CV) | MAE (days) |
|--------------|-------------------|------------|
| +`age` | 0.6241 ± 0.0377 | 255 ± 20 |
| +`dist_from_center` | 0.6570 ± 0.0365 | 244 ± 23 |
| +`tumor_min` | 0.6626 ± 0.0367 | 243 ± 21 |
| +`tumor_intensity_ratio` | 0.6695 ± 0.0322 | 240 ± 18 |
| +`eor_str` | **0.6722 ± 0.0324** | **237 ± 18** |

| Feature | Coefficient | HR | Description |
|---------|------------|-----|-------------|
| `age` | +0.546 | 1.73 | Patient age at diagnosis. Strongest predictor. |
| `dist_from_center` | -0.299 | 0.74 | Tumor centroid distance from brain center (normalized). Central tumors = worse prognosis. |
| `tumor_intensity_ratio` | -0.213 | 0.81 | Mean FLAIR intensity in tumor / mean brain intensity. |
| `tumor_min` | -0.154 | 0.86 | Minimum FLAIR intensity in tumor. Low = necrosis = aggressive. |
| `eor_str` | -0.120 | 0.89 | Subtotal resection indicator (binary). |

Prediction intervals are derived directly from survival function percentiles (no calibration needed with alpha=3.0).

### Risk Stratification

| Risk Group | Criterion | N | Median Survival | 1-year Survival |
|-----------|-----------|---|----------------|-----------------|
| **High Risk** | < 300 days | 24 | 126 days | 4.2% |
| **Medium Risk** | 300–450 days | 127 | 329 days | 42.5% |
| **Low Risk** | > 450 days | 85 | 495 days | 75.3% |

All group separations are statistically significant (log-rank p < 0.01).

### Training

```bash
# Train CoxPH (production model, trains on all data by default)
PYTHONPATH=src/survival python src/survival/train_cox.py

# Evaluate
PYTHONPATH=src/survival python src/survival/evaluate.py \
    --model_path checkpoints/survival/coxph_model.pkl
```

### Limitations

- **Age dominates**: all five imaging features combined contribute only +0.05 C-index.
- **No molecular markers**: they are stronger prognostic factors but unavailable from imaging.
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
- PyTorch 2.x with CUDA (for training)
- nnU-Net v2
- MONAI, scikit-survival, lifelines, nibabel, scipy

```bash
conda env create -f environment.yml
conda activate aeon
```
