"""Training script for LogNormal Accelerated Failure Time survival model.

The AFT model directly models survival time: log(T) = X*beta + sigma*epsilon.
It achieved CV C-index 0.676 on our data (+0.003 over CoxPH), but requires the
parametric assumption that survival follows a log-normal distribution. This holds
well for our nearly-uncensored dataset (99.6% events) but may not generalize to
datasets with substantial censoring.

Use train_cox.py for the production model (CoxPH, no parametric assumptions).
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from lifelines import LogNormalAFTFitter

from features import build_feature_dataframe, get_feature_columns


def make_survival_array(df):
    """Convert dataframe to structured array for scikit-survival."""
    return np.array(
        [(bool(e), t) for e, t in zip(df["event"], df["time"])],
        dtype=[("event", bool), ("time", float)],
    )


def train_lognormal_aft(X_train_scaled, train_df, feature_cols, penalizer=0.01):
    """Train a LogNormal AFT model using lifelines."""
    df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    df["time"] = train_df["time"].values
    df["event"] = train_df["event"].values
    fitter = LogNormalAFTFitter(penalizer=penalizer)
    fitter.fit(df, duration_col="time", event_col="event")
    return fitter


def main():
    parser = argparse.ArgumentParser(description="Train LogNormal AFT survival model")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--survival_csv", default="data/survival_info.csv")
    parser.add_argument("--penalizer", type=float, default=0.01, help="L2 penalizer")
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

    # Train LogNormal AFT
    print(f"Training LogNormal AFT (penalizer={args.penalizer}) with {X_train.shape[1]} features...")
    model = train_lognormal_aft(X_train, train_df, feature_cols, penalizer=args.penalizer)

    # Training C-index
    pred_median = model.predict_median(
        pd.DataFrame(X_train, columns=feature_cols)
    ).values.flatten()
    y_train = make_survival_array(train_df)
    c_train = concordance_index_censored(
        y_train["event"], y_train["time"], -pred_median
    )[0]
    print(f"Training C-index: {c_train:.4f}")

    # Print summary
    model.print_summary()

    # Save model and scaler
    model_path = os.path.join(args.output_dir, "lognormal_aft.pkl")
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

        fold_model = train_lognormal_aft(
            X_ft, train_df.iloc[fold_train_idx], feature_cols, penalizer=args.penalizer
        )
        pred = fold_model.predict_median(
            pd.DataFrame(X_fv, columns=feature_cols)
        ).values.flatten()
        oof_predicted[fold_val_idx] = pred

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_predicted, train_df["time"].values)
    cal_path = os.path.join(args.output_dir, "calibrator.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {cal_path}")


if __name__ == "__main__":
    main()
