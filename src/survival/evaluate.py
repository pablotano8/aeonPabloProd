"""Evaluation script for survival analysis benchmark."""

import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from features import build_feature_dataframe, get_feature_columns

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from checkpoint_bootstrap import ensure_checkpoints


def make_survival_array(df):
    """Convert dataframe to structured array for scikit-survival."""
    return np.array(
        [(bool(e), t) for e, t in zip(df["event"], df["time"])],
        dtype=[("event", bool), ("time", float)],
    )


def evaluate(model, scaler, val_df, train_df):
    """Run full benchmark evaluation."""
    feature_cols = get_feature_columns()
    X_val = val_df[feature_cols].values.astype(np.float64)
    X_val = scaler.transform(X_val)
    X_val = np.nan_to_num(X_val, nan=0.0)

    y_val = make_survival_array(val_df)
    y_train = make_survival_array(train_df)

    results = {}

    risk_scores = model.predict(X_val)

    # 1. C-index (primary metric)
    c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        y_val["event"], y_val["time"], risk_scores
    )
    results["c_index"] = float(c_index)
    results["concordant_pairs"] = int(concordant)
    results["discordant_pairs"] = int(discordant)

    # 2. Integrated Brier Score
    try:
        surv_funcs = model.predict_survival_function(X_val)
        times = np.linspace(
            max(y_train["time"].min(), y_val["time"].min()) + 1,
            min(y_train["time"].max(), y_val["time"].max()) - 1,
            100,
        )
        surv_probs = np.zeros((len(X_val), len(times)))
        for i, sf in enumerate(surv_funcs):
            surv_probs[i] = sf(times)
        ibs = integrated_brier_score(y_train, y_val, surv_probs, times)
        results["integrated_brier_score"] = float(ibs)
    except Exception as e:
        results["integrated_brier_score"] = None
        results["ibs_error"] = str(e)

    # 3. MAE of predicted median survival time
    try:
        surv_funcs = model.predict_survival_function(X_val)
        predicted_medians = []
        for sf in surv_funcs:
            sf_times = sf.x
            sf_probs = sf.y
            below_half = np.where(sf_probs <= 0.5)[0]
            if len(below_half) > 0:
                predicted_medians.append(float(sf_times[below_half[0]]))
            else:
                predicted_medians.append(float(sf_times[-1]))
        predicted_medians = np.array(predicted_medians)

        actual_times = val_df["time"].values
        mae = float(np.mean(np.abs(predicted_medians - actual_times)))
        median_ae = float(np.median(np.abs(predicted_medians - actual_times)))
        results["mae_days"] = mae
        results["median_ae_days"] = median_ae
        results["predicted_medians"] = predicted_medians.tolist()

    except Exception as e:
        results["mae_days"] = None
        results["mae_error"] = str(e)

    # 4. 1-year survival AUC
    try:
        actual_1yr = (val_df["time"].values > 365).astype(int)
        if len(np.unique(actual_1yr)) > 1:
            auc_1yr = roc_auc_score(actual_1yr, -risk_scores)
            results["auc_1year"] = float(auc_1yr)
        else:
            results["auc_1year"] = None
            results["auc_1year_note"] = "Only one class in validation set"
    except Exception as e:
        results["auc_1year"] = None
        results["auc_error"] = str(e)

    return results


def main():
    ensure_checkpoints()

    parser = argparse.ArgumentParser(description="Evaluate survival model")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--survival_csv", default="data/survival_info.csv")
    parser.add_argument("--model_path", required=True, help="Path to trained model pickle")
    parser.add_argument("--scaler_path", default="checkpoints/survival/scaler.pkl")
    parser.add_argument("--output", default="results/survival_eval.json")
    parser.add_argument("--train_features_cache", default="data/train_features.csv")
    parser.add_argument("--val_features_cache", default="data/val_features.csv")
    args = parser.parse_args()

    # Load model and scaler
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Build or load features
    if os.path.exists(args.val_features_cache):
        print(f"Loading cached val features from {args.val_features_cache}")
        val_df = pd.read_csv(args.val_features_cache)
    else:
        print("Extracting validation features...")
        val_df = build_feature_dataframe(args.data_dir, args.survival_csv, split="validation")
        val_df.to_csv(args.val_features_cache, index=False)

    if os.path.exists(args.train_features_cache):
        train_df = pd.read_csv(args.train_features_cache)
    else:
        print("Extracting training features...")
        train_df = build_feature_dataframe(args.data_dir, args.survival_csv, split="train")
        train_df.to_csv(args.train_features_cache, index=False)

    print(f"Validation set: {len(val_df)} patients")
    print(f"Training set: {len(train_df)} patients (for IBS reference)")

    # Evaluate
    results = evaluate(model, scaler, val_df, train_df)

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"C-index (primary):        {results['c_index']:.4f}")
    if results.get("integrated_brier_score") is not None:
        print(f"Integrated Brier Score:   {results['integrated_brier_score']:.4f}")
    if results.get("mae_days") is not None:
        print(f"MAE (days):               {results['mae_days']:.1f}")
        print(f"Median AE (days):         {results['median_ae_days']:.1f}")
    if results.get("auc_1year") is not None:
        print(f"1-year survival AUC:      {results['auc_1year']:.4f}")
    print("=" * 50)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
