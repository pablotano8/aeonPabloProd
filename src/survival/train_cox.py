"""Training script for Cox Proportional Hazards survival model (production)."""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from features import build_feature_dataframe, get_feature_columns


def make_survival_array(df):
    """Convert dataframe to structured array for scikit-survival."""
    return np.array(
        [(bool(e), t) for e, t in zip(df["event"], df["time"])],
        dtype=[("event", bool), ("time", float)],
    )


def main():
    parser = argparse.ArgumentParser(description="Train CoxPH survival model")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--survival_csv", default="data/survival_info.csv")
    parser.add_argument("--alpha", type=float, default=0.1, help="L2 regularization")
    parser.add_argument("--output_dir", default="checkpoints/survival")
    parser.add_argument("--features_cache", default="data/train_features.csv")
    parser.add_argument("--val_features_cache", default="data/val_features.csv")
    parser.add_argument("--use_all_data", action="store_true",
                        help="Train on train+val combined (for production deployment)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build or load features
    if os.path.exists(args.features_cache):
        print(f"Loading cached features from {args.features_cache}")
        train_df = pd.read_csv(args.features_cache)
    else:
        print("Extracting training features...")
        train_df = build_feature_dataframe(args.data_dir, args.survival_csv, split="train")
        train_df.to_csv(args.features_cache, index=False)
        print(f"Saved features to {args.features_cache}")

    if args.use_all_data:
        if os.path.exists(args.val_features_cache):
            val_df = pd.read_csv(args.val_features_cache)
        else:
            val_df = build_feature_dataframe(args.data_dir, args.survival_csv, split="validation")
            val_df.to_csv(args.val_features_cache, index=False)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"Using ALL data: {len(train_df)} patients (train + val)")
    else:
        print(f"Training set: {len(train_df)} patients")

    feature_cols = get_feature_columns()
    X_raw = train_df[feature_cols].values.astype(np.float64)

    # Standardize features
    scaler = StandardScaler()
    X_train = np.nan_to_num(scaler.fit_transform(X_raw), nan=0.0)

    # Train CoxPH
    print(f"Training CoxPH (alpha={args.alpha}) with {X_train.shape[1]} features...")
    y_train = make_survival_array(train_df)
    model = CoxPHSurvivalAnalysis(alpha=args.alpha)
    model.fit(X_train, y_train)

    c_train = concordance_index_censored(y_train["event"], y_train["time"],
                                         model.predict(X_train))[0]
    print(f"Training C-index: {c_train:.4f}")

    # Print coefficients
    print("\nCoefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        hr = np.exp(coef)
        print(f"  {feat:30s}: coef={coef:+.4f}  HR={hr:.3f}")

    # Save model and scaler
    model_path = os.path.join(args.output_dir, "coxph_model.pkl")
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved to {model_path}")

    # Fit calibrator using out-of-fold predictions
    print("Fitting calibrator via 5-fold cross-validation...")
    oof_predicted = np.zeros(len(X_train))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_train_idx, fold_val_idx in kf.split(X_train):
        fold_scaler = StandardScaler()
        X_ft = np.nan_to_num(fold_scaler.fit_transform(X_raw[fold_train_idx]), nan=0.0)
        X_fv = np.nan_to_num(fold_scaler.transform(X_raw[fold_val_idx]), nan=0.0)

        fold_model = CoxPHSurvivalAnalysis(alpha=args.alpha)
        fold_model.fit(X_ft, y_train[fold_train_idx])

        surv_funcs = fold_model.predict_survival_function(X_fv)
        for i, sf in zip(fold_val_idx, surv_funcs):
            below = np.where(sf.y <= 0.5)[0]
            oof_predicted[i] = sf.x[below[0]] if len(below) > 0 else sf.x[-1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_predicted, train_df["time"].values)
    cal_path = os.path.join(args.output_dir, "calibrator.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {cal_path}")


if __name__ == "__main__":
    main()
