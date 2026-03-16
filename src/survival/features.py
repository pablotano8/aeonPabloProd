"""Feature extraction for survival analysis from FLAIR MRI + segmentation masks."""

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage


def parse_survival_csv(csv_path):
    """Parse survival_info.csv, extracting time and event indicator."""
    df = pd.read_csv(csv_path)

    def parse_row(val):
        val = str(val)
        m = re.match(r"ALIVE \((\d+) days later\)", val)
        if m:
            return int(m.group(1)), 0  # censored
        return int(val), 1  # dead

    parsed = df["Survival_days"].apply(parse_row)
    df["time"] = [p[0] for p in parsed]
    df["event"] = [p[1] for p in parsed]
    return df


def extract_patient_features(flair_path, seg_path):
    """Extract radiomics-style features from a single patient's FLAIR + segmentation."""
    seg_img = nib.load(seg_path)
    flair_img = nib.load(flair_path)
    seg = seg_img.get_fdata()
    flair = flair_img.get_fdata()
    voxel_sizes = seg_img.header.get_zooms()
    voxel_vol = float(np.prod(voxel_sizes))

    mask = seg > 0.5
    tumor_voxels = int(mask.sum())

    features = {}
    features["tumor_voxels"] = tumor_voxels
    features["tumor_volume_mm3"] = tumor_voxels * voxel_vol
    features["log_tumor_volume"] = np.log1p(tumor_voxels * voxel_vol)

    if tumor_voxels == 0:
        # No tumor — fill with zeros
        for key in [
            "tumor_mean", "tumor_std", "tumor_median", "tumor_max",
            "tumor_min", "tumor_skewness", "tumor_kurtosis", "tumor_energy",
            "tumor_intensity_ratio", "surface_area_mm2", "sphericity",
            "extent_x", "extent_y", "extent_z", "centroid_x", "centroid_y", "centroid_z",
            "dist_from_center", "compactness",
        ]:
            features[key] = 0.0
        return features

    tumor_intensities = flair[mask]

    # First-order intensity features
    features["tumor_mean"] = float(tumor_intensities.mean())
    features["tumor_std"] = float(tumor_intensities.std())
    features["tumor_median"] = float(np.median(tumor_intensities))
    features["tumor_max"] = float(tumor_intensities.max())
    features["tumor_min"] = float(tumor_intensities.min())

    # Skewness and kurtosis
    if features["tumor_std"] > 0:
        centered = tumor_intensities - features["tumor_mean"]
        features["tumor_skewness"] = float(np.mean(centered**3) / (features["tumor_std"]**3))
        features["tumor_kurtosis"] = float(np.mean(centered**4) / (features["tumor_std"]**4) - 3)
    else:
        features["tumor_skewness"] = 0.0
        features["tumor_kurtosis"] = 0.0

    features["tumor_energy"] = float(np.sum(tumor_intensities**2))

    # Intensity ratio (tumor vs brain)
    brain_mask = flair > np.percentile(flair[flair > 0], 5) if (flair > 0).any() else flair > 0
    brain_mean = float(flair[brain_mask].mean()) if brain_mask.any() else 1.0
    features["tumor_intensity_ratio"] = features["tumor_mean"] / brain_mean if brain_mean > 0 else 0.0

    # Shape features
    # Surface area approximation via erosion
    eroded = ndimage.binary_erosion(mask)
    surface_voxels = int(mask.sum() - eroded.sum())
    # Approximate surface area: each surface voxel contributes ~voxel_face_area
    avg_face_area = (voxel_sizes[0] * voxel_sizes[1] + voxel_sizes[0] * voxel_sizes[2] + voxel_sizes[1] * voxel_sizes[2]) / 3
    features["surface_area_mm2"] = surface_voxels * avg_face_area

    # Sphericity
    vol = features["tumor_volume_mm3"]
    sa = features["surface_area_mm2"]
    if sa > 0:
        features["sphericity"] = (np.pi ** (1/3) * (6 * vol) ** (2/3)) / sa
    else:
        features["sphericity"] = 0.0

    # Bounding box extents
    coords = np.argwhere(mask)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extent = (bbox_max - bbox_min + 1) * np.array(voxel_sizes)
    features["extent_x"] = float(extent[0])
    features["extent_y"] = float(extent[1])
    features["extent_z"] = float(extent[2])

    # Centroid (normalized to image dimensions)
    centroid = coords.mean(axis=0)
    img_shape = np.array(seg.shape, dtype=float)
    features["centroid_x"] = float(centroid[0] / img_shape[0])
    features["centroid_y"] = float(centroid[1] / img_shape[1])
    features["centroid_z"] = float(centroid[2] / img_shape[2])

    # Distance from brain center (normalized)
    brain_center = img_shape / 2
    features["dist_from_center"] = float(np.sqrt(np.sum(((centroid - brain_center) / img_shape) ** 2)))

    # Compactness (tumor volume / bounding box volume)
    bbox_vol = np.prod(bbox_max - bbox_min + 1)
    features["compactness"] = float(tumor_voxels / bbox_vol) if bbox_vol > 0 else 0.0

    return features


def build_feature_dataframe(data_dir, survival_csv_path, split="train",
                            predictions_dir=None):
    """Build a feature dataframe for all patients in a given split.

    Args:
        data_dir: Root data directory containing train/ and validation/.
        survival_csv_path: Path to survival_info.csv.
        split: Which split to use ("train" or "validation").
        predictions_dir: If provided, use predicted segmentation masks from this
            directory (e.g. data/validation_predictions/) instead of ground-truth
            seg.nii files. Each file should be named {patient_id}.nii.gz.
    """
    surv_df = parse_survival_csv(survival_csv_path)
    split_dir = os.path.join(data_dir, split)
    patient_ids = set(os.listdir(split_dir))

    rows = []
    for _, row in surv_df.iterrows():
        pid = str(int(row["ID"]))
        if pid not in patient_ids:
            continue

        pdir = os.path.join(split_dir, pid)
        flair_path = os.path.join(pdir, "flair.nii")

        if predictions_dir is not None:
            seg_path = os.path.join(predictions_dir, f"{pid}.nii.gz")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(predictions_dir, f"{pid}.nii")
        else:
            seg_path = os.path.join(pdir, "seg.nii")

        if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
            continue

        feats = extract_patient_features(flair_path, seg_path)
        feats["ID"] = int(row["ID"])
        feats["age"] = row["Age"]
        feats["time"] = row["time"]
        feats["event"] = row["event"]

        # Extent of resection encoding
        eor = row["Extent_of_Resection"]
        feats["eor_gtr"] = 1.0 if eor == "GTR" else 0.0
        feats["eor_str"] = 1.0 if eor == "STR" else 0.0

        rows.append(feats)

    return pd.DataFrame(rows)


def get_feature_columns():
    """Return the list of feature column names used for model input."""
    return [
        "age", "dist_from_center",
        "tumor_min", "tumor_intensity_ratio",
        "eor_str",
    ]
