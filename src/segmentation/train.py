"""Train the nnU-Net segmentation model.

Wraps the nnU-Net v2 CLI pipeline into a single script:
  1. Convert data to nnU-Net format (symlinks)
  2. Plan and preprocess
  3. Train (fold=all, 100 epochs)

After training, copy the checkpoint to checkpoints/segmentation/ for inference.

Usage:
    python src/segmentation/train.py
    python src/segmentation/train.py --epochs 250
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(cmd, env=None):
    """Run a command, streaming output in real time."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def train(args):
    nnunet_raw = str(REPO_ROOT / "nnunet" / "raw")
    nnunet_preprocessed = str(REPO_ROOT / "nnunet" / "preprocessed")
    nnunet_results = str(REPO_ROOT / "nnunet" / "results")

    env = os.environ.copy()
    env["nnUNet_raw"] = nnunet_raw
    env["nnUNet_preprocessed"] = nnunet_preprocessed
    env["nnUNet_results"] = nnunet_results

    # Step 1: Convert data
    print("=" * 60)
    print("Step 1: Converting data to nnU-Net format")
    print("=" * 60)
    from convert_data import convert
    convert()

    # Step 2: Plan and preprocess
    print("\n" + "=" * 60)
    print("Step 2: Planning and preprocessing")
    print("=" * 60)
    run(["nnUNetv2_plan_and_preprocess", "-d", "1", "--verify_dataset_integrity"], env=env)

    # Step 3: Train
    trainer = f"nnUNetTrainer_{args.epochs}epochs"
    print("\n" + "=" * 60)
    print(f"Step 3: Training ({trainer}, fold=all)")
    print("=" * 60)
    run([
        "nnUNetv2_train", "1", "3d_fullres", "all",
        "-tr", trainer, "--npz",
    ], env=env)

    # Step 4: Copy checkpoint for inference
    print("\n" + "=" * 60)
    print("Step 4: Copying checkpoint to checkpoints/segmentation/")
    print("=" * 60)
    src_dir = (
        Path(nnunet_results) / "Dataset001_BrainFLAIR"
        / f"{trainer}__nnUNetPlans__3d_fullres"
    )
    dst_dir = REPO_ROOT / "checkpoints" / "segmentation"
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "fold_all").mkdir(exist_ok=True)

    for fname in ["dataset.json", "dataset_fingerprint.json", "plans.json"]:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(dst_dir / fname))

    ckpt_src = src_dir / "fold_all" / "checkpoint_final.pth"
    if ckpt_src.exists():
        shutil.copy2(str(ckpt_src), str(dst_dir / "fold_all" / "checkpoint_final.pth"))
        print(f"Checkpoint saved to {dst_dir / 'fold_all' / 'checkpoint_final.pth'}")
    else:
        print(f"WARNING: {ckpt_src} not found")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nnU-Net segmentation model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    train(args)
