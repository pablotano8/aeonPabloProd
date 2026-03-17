"""Compare Cox PH corrections: all combinations of stratifying on PH-violating features.

PH violators (Grambsch-Therneau p < 0.05): age, dist_from_center, tumor_min
We try every subset as strata (keeping the rest as covariates) plus the baseline.

Usage:
    python ph_correction_experiment.py
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(__file__))
from features import get_feature_columns

FEATURES = get_feature_columns()
PH_VIOLATORS = ["age", "dist_from_center", "tumor_min"]
REFERENCE_TIME = 365.0


# ── helpers ───────────────────────────────────────────────────────────────────

def cv_cindex(fit_fn, df, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in kf.split(df):
        train = df.iloc[train_idx].copy().reset_index(drop=True)
        test  = df.iloc[test_idx].copy().reset_index(drop=True)
        cph = fit_fn(train)
        risk = cph.predict_partial_hazard(test)
        c = concordance_index(test["time"], -risk, test["event"])
        scores.append(c)
    return np.array(scores)


# ── Generalised stratified Cox ────────────────────────────────────────────────

def make_stratified_fitter(strata_features):
    """Return a fit_fn that stratifies on `strata_features` (discretised to tertiles)
    and uses all remaining FEATURES as covariates."""
    covariate_features = [f for f in FEATURES if f not in strata_features]
    cat_cols = [f"{f}_cat" for f in strata_features]

    def fit_fn(train):
        # Compute tertile cuts on training data only
        cuts = {}
        for feat in strata_features:
            _, bins = pd.qcut(train[feat], q=3, retbins=True, duplicates="drop")
            bins[0] = -np.inf;  bins[-1] = np.inf
            cuts[feat] = bins

        def add_cats(df):
            d = df.copy()
            for feat, bins in cuts.items():
                n = len(bins) - 1
                labels = [f"{feat[:4]}_{i}" for i in range(n)]
                d[f"{feat}_cat"] = pd.cut(d[feat], bins=bins, labels=labels)
            return d

        t = add_cats(train)
        sc = StandardScaler()
        if covariate_features:
            t[covariate_features] = sc.fit_transform(t[covariate_features])

        fit_cols = covariate_features + cat_cols + ["time", "event"]
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(t[fit_cols], duration_col="time", event_col="event",
                strata=cat_cols if cat_cols else None)

        orig_pred = cph.predict_partial_hazard

        def predict_partial_hazard(df):
            d = add_cats(df).copy()
            if covariate_features:
                d[covariate_features] = sc.transform(d[covariate_features])
            return orig_pred(d[fit_cols])

        cph.predict_partial_hazard = predict_partial_hazard
        return cph

    return fit_fn


# ── Baseline ──────────────────────────────────────────────────────────────────

def fit_baseline(train):
    cols = FEATURES + ["time", "event"]
    sc = StandardScaler()
    t = train[cols].copy()
    t[FEATURES] = sc.fit_transform(t[FEATURES])
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(t, duration_col="time", event_col="event")
    orig_pred = cph.predict_partial_hazard

    def predict_partial_hazard(df):
        d = df[cols].copy()
        d[FEATURES] = sc.transform(d[FEATURES])
        return orig_pred(d)

    cph.predict_partial_hazard = predict_partial_hazard
    return cph


# ── log(t) interaction Cox ────────────────────────────────────────────────────

def make_logt_fitter(interaction_features):
    """Add feature × log(t) terms for `interaction_features` only.
    Evaluated at REFERENCE_TIME for prediction."""
    interaction_cols = [f"{f}_x_logt" for f in interaction_features]
    all_feats = FEATURES + interaction_cols
    log_ref = np.log(REFERENCE_TIME)

    def fit_fn(train):
        def add_interactions(df, log_t):
            d = df.copy()
            for feat in interaction_features:
                d[f"{feat}_x_logt"] = d[feat] * log_t
            return d

        t = train.copy()
        t = add_interactions(t, np.log(t["time"].clip(lower=1)))
        sc = StandardScaler()
        t[all_feats] = sc.fit_transform(t[all_feats])

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(t[all_feats + ["time", "event"]], duration_col="time", event_col="event")

        orig_pred = cph.predict_partial_hazard

        def predict_partial_hazard(df):
            d = add_interactions(df.copy(), log_ref)
            d[all_feats] = sc.transform(d[all_feats])
            return orig_pred(d[all_feats + ["time", "event"]])

        cph.predict_partial_hazard = predict_partial_hazard
        return cph

    return fit_fn


# ── Run experiment ─────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("/tmp/all_features_cache.csv")
    df = df[FEATURES + ["time", "event"]].dropna().reset_index(drop=True)
    print(f"Dataset: {len(df)} patients, {int(df['event'].sum())} events\n")

    # Baseline + all non-empty subsets of PH_VIOLATORS as strata
    models = {"Baseline\n(no strata)": fit_baseline}
    for r in range(1, len(PH_VIOLATORS) + 1):
        for subset in combinations(PH_VIOLATORS, r):
            covariate_feats = [f for f in FEATURES if f not in subset]
            strata_label = "+".join(f[:4] for f in subset)
            label = f"Strata:\n{strata_label}"
            models[label] = make_stratified_fitter(list(subset))

    # log(t) interaction subsets (excluding age, and all combos)
    logt_subsets = [
        ["dist_from_center", "tumor_min"],   # no age
        ["dist_from_center"],
        ["tumor_min"],
        ["age", "dist_from_center", "tumor_min"],  # all three
    ]
    for subset in logt_subsets:
        label = "log(t)×\n" + "+".join(f[:4] for f in subset)
        models[label] = make_logt_fitter(subset)

    results = {}
    for name, fit_fn in models.items():
        scores = cv_cindex(fit_fn, df)
        results[name] = scores
        flat = name.replace("\n", " ")
        print(f"{flat}")
        print(f"  C-index: {scores.mean():.4f} ± {scores.std():.4f}  "
              f"(folds: {', '.join(f'{s:.3f}' for s in scores)})\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n = len(results)
    fig, ax = plt.subplots(figsize=(max(14, n * 1.5), 6))
    fig.patch.set_facecolor("white")

    labels = list(results.keys())
    means  = np.array([results[k].mean() for k in labels])
    stds   = np.array([results[k].std()  for k in labels])

    def model_color(label):
        if label.startswith("Baseline"):  return "#4878CF"
        if label.startswith("Strata:"):   return "#FFA500"
        return "#6ACC65"  # log(t) models

    colors = [model_color(l) for l in labels]

    x = np.arange(n)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, width=0.6, error_kw=dict(linewidth=1.3))

    for i, k in enumerate(labels):
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(results[k]))
        ax.scatter(x[i] + jitter, results[k], color="black",
                   s=18, zorder=5, alpha=0.55)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("5-fold CV C-index", fontsize=11)
    ax.set_title("PH Corrections: Stratification vs log(t) Interactions",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0.50, 0.82)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#4878CF", alpha=0.85, label="Baseline (no correction)"),
        Patch(facecolor="#FFA500", alpha=0.85, label="Stratified Cox"),
        Patch(facecolor="#6ACC65", alpha=0.85, label="log(t) interactions"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = "ph_correction_experiment.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
