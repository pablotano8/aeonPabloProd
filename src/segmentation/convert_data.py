"""Convert data to nnU-Net v2 format for training.

Creates symlinks from data/{train,validation}/ to the nnU-Net raw directory
structure expected by nnUNetv2_plan_and_preprocess.

Usage:
    python src/segmentation/convert_data.py

Then train with:
    export nnUNet_raw="$(pwd)/nnunet/raw"
    export nnUNet_preprocessed="$(pwd)/nnunet/preprocessed"
    export nnUNet_results="$(pwd)/nnunet/results"
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
    nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainer_100epochs --npz
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_TRAIN = REPO_ROOT / "data" / "train"
SRC_VAL = REPO_ROOT / "data" / "validation"
DST = REPO_ROOT / "nnunet" / "raw" / "Dataset001_BrainFLAIR"


def convert():
    images_tr = DST / "imagesTr"
    labels_tr = DST / "labelsTr"
    images_ts = DST / "imagesTs"

    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Training subjects
    train_count = 0
    for subject in sorted(SRC_TRAIN.iterdir(), key=lambda x: int(x.name)):
        if not subject.is_dir():
            continue
        flair = subject / "flair.nii"
        seg = subject / "seg.nii"
        if not (flair.exists() and seg.exists()):
            continue
        case = f"BrainFLAIR_{subject.name.zfill(4)}"
        img_dst = images_tr / f"{case}_0000.nii"
        lbl_dst = labels_tr / f"{case}.nii"
        if not img_dst.exists():
            os.symlink(flair.resolve(), img_dst)
        if not lbl_dst.exists():
            os.symlink(seg.resolve(), lbl_dst)
        train_count += 1

    # Validation subjects (test set for nnU-Net)
    val_count = 0
    for subject in sorted(SRC_VAL.iterdir(), key=lambda x: int(x.name)):
        if not subject.is_dir():
            continue
        flair = subject / "flair.nii"
        if not flair.exists():
            continue
        case = f"BrainFLAIR_{subject.name.zfill(4)}"
        img_dst = images_ts / f"{case}_0000.nii"
        if not img_dst.exists():
            os.symlink(flair.resolve(), img_dst)
        val_count += 1

    dataset_json = {
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": train_count,
        "file_ending": ".nii",
    }
    with open(DST / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Training cases: {train_count}")
    print(f"Validation/test cases: {val_count}")
    print(f"Dataset written to {DST}")


if __name__ == "__main__":
    convert()
