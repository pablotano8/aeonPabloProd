"""Single-scan inference script for the playground.

Uses nnU-Net v2 (best model) to produce:
  - segmentation.nii.gz          — binary mask at threshold 0.5 (Dice 0.915, Sens 90.0%)
  - segmentation_high_sens.nii.gz — high-sensitivity mask (threshold 0.001, Sens 99.0%, Dice 0.752)
  - tumor_probability.nii.gz      — continuous probability map [0, 1]
  - uncertainty.nii.gz            — per-voxel uncertainty (high near decision boundary)

If --age is provided, also runs CoxPH survival prediction using radiomics
features extracted from the predicted segmentation.
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import shutil
import numpy as np
import nibabel as nib
import torch

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
REPO_ROOT = os.path.dirname(SRC_DIR)  # repository root
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "checkpoints", "segmentation")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from checkpoint_bootstrap import ensure_checkpoints

# Threshold pre-computed on 74 validation subjects to achieve ~99% sensitivity.
# At 0.001: Sensitivity=99.0%, Precision=60.6%, Dice=0.752
# At 0.500: Sensitivity=90.0%, Precision=93.1%, Dice=0.915
HIGH_SENS_THRESHOLD = 0.001


def infer(args):
    ensure_checkpoints()

    # Set nnU-Net env vars to suppress warnings
    os.environ.setdefault("nnUNet_raw", os.path.join(REPO_ROOT, "nnunet", "raw"))
    os.environ.setdefault("nnUNet_preprocessed", os.path.join(REPO_ROOT, "nnunet", "preprocessed"))
    os.environ.setdefault("nnUNet_results", os.path.join(REPO_ROOT, "nnunet", "results"))

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"nnU-Net model directory not found: {model_dir}\n"
            "Run training first or adjust --model_dir."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # nnU-Net expects input files named <subject>_0000.<ext> in a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # Copy input file with nnU-Net naming convention
        ext = ".nii.gz" if args.input.endswith(".gz") else ".nii"
        in_file = os.path.join(in_dir, f"INFER_0000{ext}")
        shutil.copy2(args.input, in_file)

        print("Loading nnU-Net model...")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,   # TTA disabled for speed
            perform_everything_on_device=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            model_dir,
            use_folds=("all",),
            checkpoint_name="checkpoint_final.pth",
        )

        print("Running inference...")
        predictor.predict_from_files(
            in_dir, out_dir,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        # Read predictions — nnU-Net output file is named after the input
        pred_nii_path = os.path.join(out_dir, f"INFER{ext}")
        prob_npz_path = os.path.join(out_dir, "INFER.npz")

        seg_nii = nib.load(pred_nii_path)
        seg_data = seg_nii.get_fdata().astype(np.float32)
        affine = seg_nii.affine

        # nnU-Net saves probabilities as shape (C, Z, Y, X) in original image space.
        # nibabel loads NIfTI in (X, Y, Z) order, so transpose axes.
        prob_npz = np.load(prob_npz_path)
        key = "probabilities" if "probabilities" in prob_npz else "softmax"
        tumor_prob = prob_npz[key][1].transpose(2, 1, 0).astype(np.float32)

    # Uncertainty: voxels near 0.5 are uncertain (range 0-1)
    uncertainty = 1.0 - np.abs(2.0 * tumor_prob - 1.0)

    # Binary segmentation at two thresholds:
    #   0.5     → Dice 0.915, Sensitivity 90.0%, Precision 93.1%
    #   0.001   → Sensitivity 99.0%, Dice 0.752  (pre-computed on validation set)
    seg_normal = (seg_data > 0.5).astype(np.float32)
    seg_high_sens = (tumor_prob >= HIGH_SENS_THRESHOLD).astype(np.float32)

    # Save outputs
    nib.save(nib.Nifti1Image(seg_normal, affine),
             os.path.join(args.output_dir, "segmentation.nii.gz"))
    nib.save(nib.Nifti1Image(seg_high_sens, affine),
             os.path.join(args.output_dir, "segmentation_high_sens.nii.gz"))
    nib.save(nib.Nifti1Image(tumor_prob, affine),
             os.path.join(args.output_dir, "tumor_probability.nii.gz"))
    nib.save(nib.Nifti1Image(uncertainty, affine),
             os.path.join(args.output_dir, "uncertainty.nii.gz"))

    # Summary stats
    tumor_volume_cm3 = seg_normal.sum() / 1000.0
    mean_uncertainty = float(uncertainty[seg_normal > 0].mean()) if seg_normal.sum() > 0 else 0.0

    print(f"\nTumor volume: {tumor_volume_cm3:.1f} cm³")
    print(f"Max tumor probability: {float(tumor_prob.max()):.4f}")
    print(f"Mean uncertainty in tumor region: {mean_uncertainty:.4f}")
    print(f"Voxels above threshold: {int(seg_normal.sum())}")
    print(f"Segmentation saved to {args.output_dir}/segmentation.nii.gz")

    # Write stats to JSON so the server doesn't need to parse stdout
    stats = {
        "tumor_volume_cm3": tumor_volume_cm3,
        "max_tumor_prob": float(tumor_prob.max()),
        "mean_uncertainty": mean_uncertainty,
        "tumor_voxels": int(seg_normal.sum()),
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f)

    # Survival prediction (requires age)
    if args.age is not None:
        predict_survival(args.input, seg_normal, affine, args.age, args.eor, args.output_dir)


def predict_survival(flair_path, seg_data, affine, age, eor, output_dir=None):
    """Run CoxPH survival prediction from predicted segmentation + clinical data."""
    sys.path.insert(0, os.path.join(SRC_DIR, "survival"))
    from features import extract_patient_features, get_feature_columns

    survival_dir = os.path.join(REPO_ROOT, "checkpoints", "survival")
    model_path = os.path.join(survival_dir, "coxph_model.pkl")
    scaler_path = os.path.join(survival_dir, "scaler.pkl")
    calibrator_path = os.path.join(survival_dir, "calibrator.pkl")

    if not os.path.exists(model_path):
        print("SURVIVAL_ERROR: Model not found")
        return

    # Save segmentation as temp nifti for feature extraction
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(nib.Nifti1Image(seg_data, affine), tmp.name)
        seg_path = tmp.name

    try:
        features = extract_patient_features(flair_path, seg_path)
    finally:
        os.unlink(seg_path)

    features["age"] = age
    features["eor_str"] = 1.0 if eor == "STR" else 0.0

    # Load model artifacts
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    calibrator = None
    if os.path.exists(calibrator_path):
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)

    feature_cols = get_feature_columns()
    X = np.array([[features[c] for c in feature_cols]], dtype=np.float64)
    X = np.nan_to_num(scaler.transform(X), nan=0.0)

    # Risk score
    risk_score = float(model.predict(X)[0])

    # Median survival from survival function
    surv_func = model.predict_survival_function(X)[0]
    below_half = np.where(surv_func.y <= 0.5)[0]
    median_survival = float(surv_func.x[below_half[0]]) if len(below_half) > 0 else float(surv_func.x[-1])

    # Calibrated median
    calibrated_median = median_survival
    if calibrator is not None:
        calibrated_median = float(calibrator.predict([median_survival])[0])

    # Risk group based on tertiles from training data
    # Thresholds derived from training distribution (see ml_findings.md)
    if calibrated_median < 300:
        risk_group = "High Risk"
    elif calibrated_median < 450:
        risk_group = "Medium Risk"
    else:
        risk_group = "Low Risk"

    # IQR from survival function (25th and 75th percentiles)
    below_75 = np.where(surv_func.y <= 0.25)[0]
    below_25 = np.where(surv_func.y <= 0.75)[0]
    p25 = float(surv_func.x[below_25[0]]) if len(below_25) > 0 else float(surv_func.x[0])
    p75 = float(surv_func.x[below_75[0]]) if len(below_75) > 0 else float(surv_func.x[-1])

    if calibrator is not None:
        p25 = float(calibrator.predict([p25])[0])
        p75 = float(calibrator.predict([p75])[0])

    print(f"\nMedian survival: {calibrated_median / 30.44:.1f} months ({calibrated_median:.0f} days)")
    print(f"Risk group: {risk_group}")
    print(f"Prediction interval: [{p25:.0f}, {p75:.0f}] days")

    if output_dir is not None:
        stats_path = os.path.join(output_dir, "stats.json")
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            stats = {}
        stats["survival"] = {
            "median_days": calibrated_median,
            "risk_group": risk_group,
            "risk_score": risk_score,
            "p25_days": p25,
            "p75_days": p75,
            "median_months": calibrated_median / 30.44,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run brain tumor segmentation on a single FLAIR scan")
    parser.add_argument("--input", type=str, required=True, help="Path to input FLAIR NIfTI file")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to nnU-Net trained model folder")
    parser.add_argument("--output_dir", type=str, default="playground/output")
    parser.add_argument("--age", type=float, default=None,
                        help="Patient age in years (enables survival prediction)")
    parser.add_argument("--eor", type=str, default="GTR", choices=["GTR", "STR", "NA"],
                        help="Extent of resection (GTR=gross total, STR=subtotal)")
    args = parser.parse_args()

    infer(args)
