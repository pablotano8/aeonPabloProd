"""CoxPH survival prediction from segmentation + clinical inputs."""

import os
import pickle
import tempfile
import numpy as np
import nibabel as nib

_SURVIVAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "checkpoints", "survival",
)


def predict_survival(flair_path, seg_data, affine, age, eor,
                     checkpoint_dir=None, output_dir=None):
    """Run CoxPH survival prediction from predicted segmentation + clinical data.

    Args:
        flair_path:     Path to input FLAIR NIfTI file.
        seg_data:       Binary segmentation mask (numpy array).
        affine:         NIfTI affine matrix for seg_data.
        age:            Patient age in years.
        eor:            Extent of resection — "GTR" or "STR".
        checkpoint_dir: Path to directory containing coxph_model.pkl and scaler.pkl.
                        Defaults to checkpoints/survival/ relative to repo root.
        output_dir:     If provided, merges survival results into stats.json there.

    Returns:
        dict with median_days, risk_group, risk_score, p5/p25/p75/p95_days,
        median_months — or None if the model checkpoint is not found.
    """
    from features import extract_patient_features, get_feature_columns

    survival_dir = checkpoint_dir or _SURVIVAL_DIR
    model_path = os.path.join(survival_dir, "coxph_model.pkl")
    scaler_path = os.path.join(survival_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        print("SURVIVAL_ERROR: Model not found")
        return None

    # Save segmentation as temp NIfTI for feature extraction
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(nib.Nifti1Image(seg_data, affine), tmp.name)
        seg_path = tmp.name

    try:
        features = extract_patient_features(flair_path, seg_path)
    finally:
        os.unlink(seg_path)

    features["age"] = age
    features["eor_str"] = 1.0 if eor == "STR" else 0.0

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_cols = get_feature_columns()
    X = np.array([[features[c] for c in feature_cols]], dtype=np.float64)
    X = np.nan_to_num(scaler.transform(X), nan=0.0)

    risk_score = float(model.predict(X)[0])
    surv_func = model.predict_survival_function(X)[0]

    def _percentile(sf, p):
        below = np.where(sf.y <= (1 - p))[0]
        return float(sf.x[below[0]]) if len(below) > 0 else float(sf.x[-1])

    p5  = _percentile(surv_func, 0.05)
    p25 = _percentile(surv_func, 0.25)
    median_survival = _percentile(surv_func, 0.50)
    p75 = _percentile(surv_func, 0.75)
    p95 = _percentile(surv_func, 0.95)

    if median_survival < 300:
        risk_group = "High Risk"
    elif median_survival < 450:
        risk_group = "Medium Risk"
    else:
        risk_group = "Low Risk"

    print(f"\nMedian survival: {median_survival / 30.44:.1f} months ({median_survival:.0f} days)")
    print(f"Risk group: {risk_group}")
    print(f"50% PI: [{p25:.0f}, {p75:.0f}] days  |  90% PI: [{p5:.0f}, {p95:.0f}] days")

    result = {
        "median_days": median_survival,
        "risk_group": risk_group,
        "risk_score": risk_score,
        "p5_days": p5,
        "p25_days": p25,
        "p75_days": p75,
        "p95_days": p95,
        "median_months": median_survival / 30.44,
    }

    if output_dir is not None:
        import json
        stats_path = os.path.join(output_dir, "stats.json")
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            stats = {}
        stats["survival"] = result
        with open(stats_path, "w") as f:
            json.dump(stats, f)

    return result
