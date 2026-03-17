"""Check Cox assumptions: log-linearity (martingale residuals) and independence.

Log-linearity: plot martingale residuals vs each raw covariate.
  Flat LOWESS → linearity holds.
  Curved LOWESS → feature needs transformation.

Independence: plot martingale residuals vs patient ID (enrollment order)
  and run a Ljung-Box autocorrelation test.
  Random scatter → independence holds.
  Trend or oscillation → possible clustering / temporal enrollment effect.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(__file__))
from features import get_feature_columns

FEATURES = get_feature_columns()

FEATURE_LABELS = {
    "age": "Age (years)",
    "dist_from_center": "Distance from Center",
    "tumor_min": "Tumor Min. Intensity",
    "tumor_intensity_ratio": "Tumor Intensity Ratio",
    "eor_str": "Extent of Resection (STR)",
}


def lowess(x, y, frac=0.4):
    """Simple LOWESS via numpy for plotting."""
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    order = np.argsort(x)
    smoothed = sm_lowess(y[order], x[order], frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def main():
    df = pd.read_csv("/tmp/all_features_cache.csv")
    df = df[FEATURES + ["time", "event", "ID"]].dropna().reset_index(drop=True)
    print(f"Dataset: {len(df)} patients, {int(df['event'].sum())} events")

    # ── Fit Cox ───────────────────────────────────────────────────────────────
    model_df = df[FEATURES + ["time", "event"]].copy()
    sc = StandardScaler()
    model_df[FEATURES] = sc.fit_transform(model_df[FEATURES])

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(model_df, duration_col="time", event_col="event")

    # ── Martingale residuals ──────────────────────────────────────────────────
    # M_i = event_i - cumulative_hazard_i
    mart = cph.compute_residuals(model_df, kind="martingale")["martingale"].values
    print(f"Martingale residuals: mean={mart.mean():.4f}, std={mart.std():.4f}")

    # Raw (unstandardised) covariates for plotting
    raw = df[FEATURES].values

    # ── Figure 1: Log-linearity ───────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig1.patch.set_facecolor("white")
    fig1.suptitle(
        "Log-Linearity Check: Martingale Residuals vs. Raw Covariates\n"
        "Flat LOWESS → linearity holds   |   Curved LOWESS → consider transformation",
        fontsize=12, fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(FEATURES)))

    for i, feat in enumerate(FEATURES):
        ax = axes[i // 3][i % 3]
        x = raw[:, i]
        ax.scatter(x, mart, alpha=0.35, s=16, color=colors[i])
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

        try:
            xs, ys = lowess(x, mart)
            ax.plot(xs, ys, color=colors[i], linewidth=2.2, label="LOWESS")
        except Exception:
            pass

        # Pearson correlation (linearity indicator)
        r, p = pearsonr(x, mart)
        ax.set_title(
            f"{FEATURE_LABELS.get(feat, feat)}\nr = {r:.2f}  (p = {p:.3f})",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("Martingale residual", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend panel
    ax_leg = axes[1][2]
    ax_leg.axis("off")
    note = (
        "Martingale residual:\n"
        "  M_i = event_i − Ĥ(tᵢ|xᵢ)\n\n"
        "Expected value ≈ 0 everywhere\n"
        "if the functional form is correct.\n\n"
        "A curved LOWESS suggests the\n"
        "feature needs a transformation\n"
        "(e.g. log, square, spline).\n\n"
        "Note: eor_str is binary (0/1),\n"
        "so linearity is trivially satisfied."
    )
    ax_leg.text(0.05, 0.95, note, transform=ax_leg.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                          edgecolor="#cccccc"))

    plt.tight_layout()
    fig1.savefig("assumption_loglinearity.png", dpi=180, bbox_inches="tight",
                 facecolor="white")
    print("Saved assumption_loglinearity.png")

    # ── Figure 2: Independence ────────────────────────────────────────────────
    # Sort by patient ID — proxy for enrollment order
    id_order = np.argsort(df["ID"].values)
    mart_sorted = mart[id_order]
    ids_sorted  = df["ID"].values[id_order]

    # Ljung-Box test for autocorrelation (lags 1–10)
    lb = acorr_ljungbox(mart_sorted, lags=10, return_df=True)
    min_p = lb["lb_pvalue"].min()
    print(f"\nLjung-Box autocorrelation test (lags 1-10):")
    print(lb.to_string())

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.patch.set_facecolor("white")
    fig2.suptitle(
        "Independence Check: Martingale Residuals by Patient ID (Enrollment Order)",
        fontsize=12, fontweight="bold",
    )

    # Left: residuals vs ID
    ax = axes2[0]
    ax.scatter(ids_sorted, mart_sorted, alpha=0.4, s=16, color="#4878CF")
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    try:
        xs, ys = lowess(ids_sorted.astype(float), mart_sorted, frac=0.3)
        ax.plot(xs, ys, color="#E05C5C", linewidth=2.2, label="LOWESS trend")
        ax.legend(fontsize=9)
    except Exception:
        pass
    ax.set_xlabel("Patient ID (proxy for enrollment order)", fontsize=10)
    ax.set_ylabel("Martingale residual", fontsize=10)
    ax.set_title("Residuals vs. Enrollment Order\n(trend = systematic group effect)",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: Ljung-Box p-values by lag
    ax2 = axes2[1]
    ax2.bar(lb.index, lb["lb_pvalue"], color="#6ACC65", alpha=0.8, width=0.6)
    ax2.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="p = 0.05")
    ax2.set_xlabel("Lag", fontsize=10)
    ax2.set_ylabel("Ljung-Box p-value", fontsize=10)
    ax2.set_title(
        f"Autocorrelation Test (Ljung-Box)\nMin p = {min_p:.3f}  "
        f"{'— no autocorrelation detected' if min_p > 0.05 else '— autocorrelation detected!'}",
        fontsize=10, fontweight="bold",
        color="#2ca02c" if min_p > 0.05 else "#d62728",
    )
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig2.savefig("assumption_independence.png", dpi=180, bbox_inches="tight",
                 facecolor="white")
    print("Saved assumption_independence.png")


if __name__ == "__main__":
    main()
