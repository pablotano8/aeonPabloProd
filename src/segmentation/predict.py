"""Batch segmentation prediction for a set of patients.

Runs nnU-Net inference on all patients in a given split directory,
producing binary segmentation masks. Skips patients that already
have predictions unless --overwrite is set.

Usage:
    # Predict all validation patients
    python src/segmentation/predict.py --split validation

    # Predict all training patients
    python src/segmentation/predict.py --split train
"""

import argparse
import os
import shutil
import tempfile
import numpy as np
import nibabel as nib
import torch
import sys


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from checkpoint_bootstrap import ensure_checkpoints


def get_patients(data_dir, split):
    """Return sorted list of patient IDs in a split directory."""
    split_dir = os.path.join(data_dir, split)
    return sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))],
        key=lambda x: int(x),
    )


def predict_batch(data_dir, split, model_dir, output_dir, overwrite=False):
    """Run nnU-Net prediction on all patients in a split."""
    ensure_checkpoints()

    os.environ.setdefault("nnUNet_raw", os.path.join(os.path.dirname(model_dir), "..", "..", "raw"))
    os.environ.setdefault("nnUNet_preprocessed", os.path.join(os.path.dirname(model_dir), "..", "..", "preprocessed"))
    os.environ.setdefault("nnUNet_results", os.path.join(os.path.dirname(model_dir), "..", ".."))

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    patients = get_patients(data_dir, split)
    split_dir = os.path.join(data_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    # Filter to patients needing prediction
    need_pred = []
    for pid in patients:
        out_path = os.path.join(output_dir, f"{pid}.nii.gz")
        if overwrite or not os.path.exists(out_path):
            need_pred.append(pid)

    if not need_pred:
        print(f"All {len(patients)} predictions already exist in {output_dir}")
        return

    print(f"Need predictions for {len(need_pred)}/{len(patients)} patients")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=("all",), checkpoint_name="checkpoint_final.pth",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "input")
        pred_out = os.path.join(tmpdir, "output")
        os.makedirs(in_dir)
        os.makedirs(pred_out)

        for pid in need_pred:
            src = os.path.join(split_dir, pid, "flair.nii")
            dst = os.path.join(in_dir, f"PAT{pid}_0000.nii")
            shutil.copy2(src, dst)

        print(f"Running inference on {len(need_pred)} patients...")
        predictor.predict_from_files(
            in_dir, pred_out,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        for pid in need_pred:
            src = os.path.join(pred_out, f"PAT{pid}.nii")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, f"{pid}.nii.gz"))

    print(f"Done. Predictions saved to {output_dir}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="Batch segmentation prediction")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--data_dir", default=os.path.join(repo_root, "data"))
    parser.add_argument("--model_dir", default=os.path.join(repo_root, "checkpoints", "segmentation"))
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: data/<split>_predictions/)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, f"{args.split}_predictions")

    predict_batch(args.data_dir, args.split, args.model_dir, args.output_dir, args.overwrite)
