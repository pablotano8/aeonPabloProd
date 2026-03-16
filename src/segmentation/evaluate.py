"""Evaluation script for brain tumor segmentation.

Evaluates the nnU-Net model on validation data, reporting:
  Dice, HD95, Sensitivity, Precision
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import numpy as np
import nibabel as nib
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from checkpoint_bootstrap import ensure_checkpoints


def compute_sensitivity_precision(pred_np, label_np):
    pred_bool = pred_np > 0
    label_bool = label_np > 0
    tp = (pred_bool & label_bool).sum()
    fp = (pred_bool & ~label_bool).sum()
    fn = (~pred_bool & label_bool).sum()
    sensitivity = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return float(sensitivity), float(precision)


def compute_metrics(pred_np, label_np, device):
    """Return (dice, hd95, sensitivity, precision) given binary numpy arrays."""
    pred = torch.from_numpy(pred_np.astype(np.int32)).long().unsqueeze(0).unsqueeze(0).to(device)
    label = torch.from_numpy(label_np.astype(np.int32)).long().unsqueeze(0).unsqueeze(0).to(device)

    pred_2ch = torch.cat([1 - pred.float(), pred.float()], dim=1)
    label_2ch = torch.zeros(1, 2, *label.shape[2:], device=device)
    label_2ch.scatter_(1, label, 1)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")

    dice_metric(pred_2ch, label_2ch)
    dice_val = dice_metric.aggregate().item()

    try:
        hd95_metric(pred_2ch, label_2ch)
        hd95_val = hd95_metric.aggregate().item()
    except Exception:
        hd95_val = float("nan")

    sens, prec = compute_sensitivity_precision(pred_np, label_np)
    return dice_val, hd95_val, sens, prec


def evaluate(args):
    ensure_checkpoints()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.environ.setdefault("nnUNet_raw", os.path.join(REPO_ROOT, "nnunet", "raw"))
    os.environ.setdefault("nnUNet_preprocessed", os.path.join(REPO_ROOT, "nnunet", "preprocessed"))
    os.environ.setdefault("nnUNet_results", os.path.join(REPO_ROOT, "nnunet", "results"))

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"nnU-Net model dir not found: {model_dir}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True,
        use_mirroring=args.tta,
        perform_everything_on_device=True,
        device=device, verbose=False, verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=("all",), checkpoint_name="checkpoint_final.pth",
    )

    val_dir = os.path.join(args.data_dir, "validation")
    subjects = sorted(
        [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))],
        key=lambda x: int(x),
    )

    results = []
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        list_of_lists = []
        for sid in subjects:
            src = os.path.join(val_dir, sid, "flair.nii")
            dst = os.path.join(in_dir, f"{sid}_0000.nii")
            shutil.copy2(src, dst)
            list_of_lists.append([dst])

        output_files = [os.path.join(out_dir, f"{sid}.nii.gz") for sid in subjects]

        print(f"Running nnU-Net on {len(subjects)} subjects (TTA={args.tta})...")
        predictor.predict_from_files(
            list_of_lists, output_files,
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=1, num_processes_segmentation_export=1,
        )

        for i, sid in enumerate(subjects):
            pred_np = nib.load(output_files[i]).get_fdata(dtype=np.float32)
            label_np = nib.load(os.path.join(val_dir, sid, "seg.nii")).get_fdata(dtype=np.float32)
            pred_np = (pred_np > 0).astype(np.int32)

            dice, hd95, sens, prec = compute_metrics(pred_np, label_np, device)
            results.append({
                "subject_id": sid, "dice": dice, "hd95": hd95,
                "sensitivity": sens, "precision": prec,
            })
            print(f"  [{i+1}/{len(subjects)}] {sid}: Dice={dice:.4f}  HD95={hd95:.1f}mm  Sens={sens:.4f}  Prec={prec:.4f}")

    elapsed = time.time() - start

    # Summary
    all_dice = [r["dice"] for r in results]
    all_hd95 = [h for h in (r["hd95"] for r in results) if not np.isnan(h)]
    all_sens = [r["sensitivity"] for r in results]
    all_prec = [r["precision"] for r in results]

    summary = {
        "n_subjects": len(results),
        "dice_mean": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "dice_median": float(np.median(all_dice)),
        "hd95_mean": float(np.mean(all_hd95)) if all_hd95 else float("nan"),
        "sensitivity_mean": float(np.mean(all_sens)),
        "precision_mean": float(np.mean(all_prec)),
        "eval_time_seconds": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS  (n={len(results)})")
    print(f"{'='*60}")
    print(f"Dice:        {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}  (median {summary['dice_median']:.4f})")
    print(f"HD95:        {summary['hd95_mean']:.2f} mm")
    print(f"Sensitivity: {summary['sensitivity_mean']:.4f}")
    print(f"Precision:   {summary['precision_mean']:.4f}")
    print(f"Eval time:   {elapsed:.0f}s")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_subject": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation")
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--model_dir", default=os.path.join(REPO_ROOT, "checkpoints", "segmentation"))
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation (mirroring)")
    args = parser.parse_args()
    evaluate(args)
