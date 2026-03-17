"""Check proportional hazards assumption via scaled Schoenfeld residuals.

Usage:
    python check_ph_assumption.py --data_dir /path/to/data --output ph_assumption.png
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path for feature imports
sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_dataframe, get_feature_columns

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


FEATURE_LABELS = {
    "age": "Age",
    "dist_from_center": "Distance from Center",
    "tumor_min": "Tumor Min. Intensity",
    "tumor_intensity_ratio": "Tumor Intensity Ratio",
    "eor_str": "Extent of Resection (STR)",
}


def load_or_extract_features(data_dir, survival_csv, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    print("Extracting features (this takes a few minutes)...")
    train_df = build_feature_dataframe(data_dir, survival_csv, split="train")
    val_df = build_feature_dataframe(data_dir, survival_csv, split="validation")
    df = pd.concat([train_df, val_df], ignore_index=True)
    df.to_csv(cache_path, index=False)
    print(f"Cached features to {cache_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/pablo/Downloads/data")
    parser.add_argument("--survival_csv", default="/Users/pablo/Downloads/data/survival_info.csv")
    parser.add_argument("--cache", default="/tmp/all_features_cache.csv")
    parser.add_argument("--output", default="ph_assumption.png")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_or_extract_features(args.data_dir, args.survival_csv, args.cache)
    feature_cols = get_feature_columns()

    model_df = df[feature_cols + ["time", "event"]].dropna()
    print(f"Fitting CoxPH on {len(model_df)} patients, {len(feature_cols)} features")

    # ── Fit CoxPH with lifelines ───────────────────────────────────────────────
    cph = CoxPHFitter(penalizer=0.1)   # mild L2, consistent with alpha≈3 in sksurv
    cph.fit(model_df, duration_col="time", event_col="event")

    print("\nLifelines CoxPH summary:")
    cph.print_summary()

    # ── Schoenfeld residual test ───────────────────────────────────────────────
    results = proportional_hazard_test(cph, model_df, time_transform="rank")
    print("\nProportional Hazards test (Grambsch-Therneau):")
    print(results.summary)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_feats = len(feature_cols)
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("white")

    # Title banner
    fig.suptitle(
        "Proportional Hazards Assumption Check\n"
        "Scaled Schoenfeld Residuals vs. Time  ·  Grambsch-Therneau test",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35,
                  top=0.88, bottom=0.10, left=0.07, right=0.97)

    # Schoenfeld residuals are stored in cph.schoenfeld_residuals_
    sch_resids = cph.compute_residuals(model_df, kind="schoenfeld")  # index=time, cols=features

    p_values = results.summary["p"].to_dict()

    colors = plt.cm.tab10(np.linspace(0, 0.7, n_feats))

    # Map residual row indices → actual event times
    event_times = model_df["time"].values  # same order as model_df

    for i, feat in enumerate(feature_cols):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])

        times = event_times[sch_resids.index.values]   # actual days
        resids = sch_resids[feat].values

        # Scatter
        ax.scatter(times, resids, alpha=0.45, s=18, color=colors[i], zorder=2)

        # LOWESS smooth
        from scipy.ndimage import uniform_filter1d
        order = np.argsort(times)
        t_sorted = times[order]
        r_sorted = resids[order]
        window = max(5, len(t_sorted) // 8)
        smoothed = uniform_filter1d(r_sorted.astype(float), size=window)
        ax.plot(t_sorted, smoothed, color=colors[i], linewidth=2.0, zorder=3, label="LOWESS")

        # PH holds → flat at 0
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=1)

        p = p_values.get(feat, float("nan"))
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
        color_p = "#d62728" if p < 0.05 else "#2ca02c"
        ax.set_title(
            f"{FEATURE_LABELS.get(feat, feat)}\n{p_str}",
            fontsize=10, fontweight="bold", color=color_p,
        )
        ax.set_xlabel("Time (days)", fontsize=8)
        ax.set_ylabel("Scaled Schoenfeld\nResidual", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend panel in last subplot (bottom-right)
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")
    legend_text = (
        "How to read this plot\n\n"
        "Each dot is one patient event.\n"
        "The curve is a smoothed trend.\n\n"
        "Flat trend → PH holds ✓\n"
        "Sloped / curved trend → PH violated ✗\n\n"
        "Green p-value → no evidence against PH\n"
        "Red p-value (p < 0.05) → possible violation"
    )
    ax_leg.text(
        0.05, 0.95, legend_text,
        transform=ax_leg.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )

    plt.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()
