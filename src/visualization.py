# ============================================================
# visualization.py - All plotting functions
#
# Functions:
#   1. plot_feature_importance   – 4-panel tree-based importance
#   2. plot_roc_curves           – ROC per model (binary)
#   3. plot_precision_recall     – PR per model (binary)
#   4. plot_confusion_matrices   – heatmap grid (binary + multiclass)
#   5. plot_learning_curves      – train vs val by sample size
#   6. plot_calibration          – reliability diagrams (binary)
#   7. plot_class_distribution   – before/after SMOTE
#   8. plot_multiclass_cm        – single large multiclass CM
#   9. plot_actual_vs_predicted  – scatter (temporal regression)
#  10. plot_residuals            – residual distribution (temporal)
#  11. plot_correlation_heatmap  – feature correlation (EDA)
#  12. plot_forecast_timeline    – Gantt-style forecast chart
# ── Explainability (thesis Ch. 4) ─────────────────────────
#  13. plot_shap_summary         – SHAP beeswarm (direction of effect)
#  14. plot_shap_dependence      – SHAP dependence for top features
# 14b. shap_business_narrative   – auto-generated SHAP interpretation
# 14c. shap_interaction_analysis – interaction vs component decomposition
# 14d. shap_sparsity_analysis   – feature sparsity & lean-model sizing
#  15. plot_partial_dependence   – PDP + ICE for top features
# ── Statistical rigour (thesis Ch. 4) ────────────────────
#  16a. plot_model_comparison_forest – AUC/F1 forest plot with 95% CI
#  16b. plot_lr_coefficients     – Logistic Regression coef bar chart
# ── Temporal regression diagnostics (thesis Ch. 4) ───────
#  17. plot_qq_residuals         – Q-Q plot for residual normality
#  18. plot_heteroscedasticity   – Residuals vs fitted + Breusch-Pagan
#  19. plot_error_by_vehicle_type – MAE breakdown by retrofit type
# ============================================================
import gc
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.inspection import PartialDependenceDisplay
from itertools import cycle
from scipy import stats as sp_stats

from . import config

# ── Consistent thesis palette ────────────────────────────────
_PALETTE = list(mcolors.TABLEAU_COLORS.values())          # 10 distinct colours
_FIGSIZE_WIDE = (14, 8)
_FIGSIZE_SQUARE = (10, 8)


def plot_feature_importance(baseline_results, top_n=20):
    """Feature importance panels for all tree-based models."""
    feature_cols = baseline_results["feature_cols"]

    # Auto-detect models that expose feature_importances_
    tree_names = [
        name for name, info in baseline_results["results"].items()
        if hasattr(info.get("model"), "feature_importances_")
    ]
    if not tree_names:
        print("No models with feature_importances_ found."); return None

    n = len(tree_names)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    fig.suptitle(f"Feature Importance — Top {top_n}", fontsize=16, fontweight="bold")

    all_imp = {}
    for idx, name in enumerate(tree_names):
        ax = axes[idx]
        model = baseline_results["results"][name]["model"]
        imp = model.feature_importances_
        df = pd.DataFrame({"feature": feature_cols, "importance": imp})
        df = df.sort_values("importance", ascending=False).head(top_n)

        ax.barh(range(len(df)), df["importance"], color="steelblue")
        ax.set_yticks(range(len(df))); ax.set_yticklabels(df["feature"], fontsize=9)
        ax.set_xlabel("Importance"); ax.set_title(name, fontweight="bold")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3, linestyle="--")

        for f, i in zip(feature_cols, imp):
            all_imp.setdefault(f, []).append(i)

    # Hide unused subplot axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

    avg = pd.DataFrame([
        {"feature": f, "avg_importance": np.mean(v)} for f, v in all_imp.items()
    ]).sort_values("avg_importance", ascending=False)
    print("Top 10 consensus features:")
    print(avg.head(10).to_string(index=False))
    return avg


# ── 2. ROC Curves (binary) ──────────────────────────────────

def plot_roc_curves(binary_results):
    """
    One ROC curve per model, all overlaid on a single plot.

    Parameters
    ----------
    binary_results : dict
        Output of binary_classification / hyperparameter_tuning / smote_classification.
        Must contain 'results' (per-model dict with 'y_pred_proba' and 'roc_auc')
        and 'y_test'.
    """
    results = binary_results["results"]
    y_test = binary_results["y_test"]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    colours = cycle(_PALETTE)

    # Sort models by AUC descending so legend reads best → worst
    sorted_names = sorted(results.keys(),
                          key=lambda n: results[n].get("roc_auc", 0),
                          reverse=True)

    for name in sorted_names:
        m = results[name]
        y_proba = m.get("y_pred_proba")
        if y_proba is None:
            continue
        # Use per-model y_test if available (subsampled models like LLM-Embed)
        _yt = m.get("y_test_eval", y_test)
        if len(np.asarray(_yt)) != len(np.asarray(y_proba)):
            continue  # skip size-mismatched (cached) models
        fpr, tpr, _ = roc_curve(_yt, y_proba)
        roc_auc = auc(fpr, tpr)
        colour = next(colours)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC = 0.5)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Binary Classification", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 3. Precision-Recall Curves (binary) ─────────────────────

def plot_precision_recall(binary_results):
    """
    One PR curve per model, all overlaid.  Includes iso-F1 contours
    and Average Precision (AP) in the legend.

    Parameters
    ----------
    binary_results : dict
        Same structure as plot_roc_curves.
    """
    results = binary_results["results"]
    y_test = binary_results["y_test"]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    # ── Iso-F1 curves (light grey, for thesis reference) ─────
    f1_values = np.linspace(0.2, 0.8, num=4)
    for f1 in f1_values:
        x = np.linspace(0.01, 1, 200)
        y = f1 * x / (2 * x - f1)
        mask = (y >= 0) & (y <= 1)
        ax.plot(x[mask], y[mask], color="grey", alpha=0.15, lw=0.8)
        ax.annotate(f"F1={f1:.1f}", xy=(0.85, y[mask][-1] if mask.any() else 0.5),
                    fontsize=7, color="grey", alpha=0.5)

    # ── PR curves per model ──────────────────────────────────
    colours = cycle(_PALETTE)
    sorted_names = sorted(results.keys(),
                          key=lambda n: results[n].get("roc_auc", 0),
                          reverse=True)

    for name in sorted_names:
        m = results[name]
        y_proba = m.get("y_pred_proba")
        if y_proba is None:
            continue
        # Use per-model y_test if available (subsampled models like LLM-Embed)
        _yt = m.get("y_test_eval", y_test)
        if len(np.asarray(_yt)) != len(np.asarray(y_proba)):
            continue  # skip size-mismatched (cached) models
        precision, recall, _ = precision_recall_curve(_yt, y_proba)
        ap = average_precision_score(_yt, y_proba)
        colour = next(colours)
        ax.plot(recall, precision, color=colour, lw=2,
                label=f"{name}  (AP = {ap:.4f})")

    # Baseline: prevalence line
    prevalence = float(np.mean(y_test))
    ax.axhline(y=prevalence, color="k", linestyle="--", lw=1, alpha=0.4,
               label=f"Baseline (prevalence = {prevalence:.2f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Binary Classification",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 4. Confusion Matrices (binary + multiclass) ─────────────

def plot_confusion_matrices(binary_results):
    """
    Grid of confusion-matrix heatmaps — one per model (binary).

    Parameters
    ----------
    binary_results : dict
        Output of binary_classification / smote / tuning.
        Each results[name] must have 'confusion_matrix' dict
        with keys tn, fp, fn, tp.
    """
    results = binary_results["results"]
    names = list(results.keys())
    n = len(names)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    fig.suptitle("Confusion Matrices — Binary Classification",
                 fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(names):
        ax = axes.flat[idx]
        # Prefer optimal-threshold CM; fall back to default-threshold CM
        cm_dict = (results[name].get("confusion_matrix_opt")
                   or results[name].get("confusion_matrix"))
        if cm_dict is None:
            ax.set_visible(False); continue
        cm = np.array([[cm_dict["tn"], cm_dict["fp"]],
                       [cm_dict["fn"], cm_dict["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Retrofit", "Retrofit"],
                    yticklabels=["No Retrofit", "Retrofit"],
                    cbar=False, ax=ax, annot_kws={"size": 13})
        roc = results[name].get("roc_auc", 0)
        thr = results[name].get("optimal_threshold", 0.5)
        ax.set_title(f"{name}\n(AUC={roc:.3f}, thr={thr:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


def plot_multiclass_cm_full(y_test, y_pred, class_names=None, title=""):
    """
    Full multiclass confusion-matrix heatmap.

    Parameters
    ----------
    y_test : array-like   – true labels (string or int)
    y_pred : array-like   – predicted labels
    class_names : list     – optional display names
    title : str
    """
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if class_names is None:
        class_names = [str(l) for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title or "Multiclass Confusion Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 5. Learning Curves ──────────────────────────────────────

def plot_learning_curves(binary_results, model_names=None, scoring="roc_auc"):
    """
    Learning curves (train size vs score) for selected binary models.

    Parameters
    ----------
    binary_results : dict
        Must contain 'X_train', 'y_train', 'results'.
    model_names : list[str], optional
        Which models to plot. Defaults to top-4 by ROC-AUC.
    scoring : str
        Sklearn scoring metric (default 'roc_auc').
    """
    results = binary_results["results"]
    X_tr = binary_results["X_train"]
    y_tr = binary_results["y_train"]

    # Skip non-sklearn models (e.g. LLM-Embed, LLM-Prompted) that lack BaseEstimator tags
    _SKIP_LEARN_CURVE = {"LLM-Embed LogReg", "LLM-Prompted"}

    if model_names is None:
        ranked = sorted(results.keys(),
                        key=lambda n: results[n].get("roc_auc", 0),
                        reverse=True)
        model_names = [n for n in ranked if n not in _SKIP_LEARN_CURVE][:4]
    else:
        model_names = [n for n in model_names if n not in _SKIP_LEARN_CURVE]

    n = len(model_names)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows))
    fig.suptitle(f"Learning Curves ({scoring})",
                 fontsize=15, fontweight="bold", y=1.02)
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=config.RANDOM_STATE)
    sizes = np.linspace(0.15, 1.0, 7)

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        model = results[name].get("model")
        if model is None:
            ax.set_visible(False); continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_tr, y_tr,
                    cv=cv, scoring=scoring,
                    train_sizes=sizes, n_jobs=-1,
                    random_state=config.RANDOM_STATE,
                )
        except Exception as exc:
            print(f"  ⚠ Learning curve skipped for {name}: {exc}")
            ax.set_visible(False)
            continue

        tr_mean = train_scores.mean(axis=1)
        tr_std = train_scores.std(axis=1)
        va_mean = val_scores.mean(axis=1)
        va_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color=_PALETTE[0])
        ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std,
                        alpha=0.15, color=_PALETTE[1])
        ax.plot(train_sizes, tr_mean, "o-", color=_PALETTE[0],
                lw=2, label="Train")
        ax.plot(train_sizes, va_mean, "s-", color=_PALETTE[1],
                lw=2, label="Validation")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training samples")
        ax.set_ylabel(scoring.replace("_", " ").title())
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 6. Calibration Plots (reliability diagrams) ─────────────

def plot_calibration(binary_results, n_bins=10):
    """
    Reliability diagram — one curve per binary model overlaid.
    Shows how well predicted probabilities match actual frequencies.

    Parameters
    ----------
    binary_results : dict
        Must contain 'results' (with 'y_pred_proba') and 'y_test'.
    n_bins : int
        Number of bins for the calibration curve.
    """
    results = binary_results["results"]
    y_test = binary_results["y_test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [2, 1]})

    colours = cycle(_PALETTE)
    sorted_names = sorted(results.keys(),
                          key=lambda n: results[n].get("roc_auc", 0),
                          reverse=True)

    for name in sorted_names:
        m = results[name]
        y_proba = m.get("y_pred_proba")
        if y_proba is None:
            continue
        # Use per-model y_test if available (subsampled models like LLM-Embed)
        _yt = m.get("y_test_eval", y_test)
        if len(np.asarray(_yt)) != len(np.asarray(y_proba)):
            continue  # skip size-mismatched (cached) models
        colour = next(colours)

        # Calibration curve
        prob_true, prob_pred = calibration_curve(_yt, y_proba, n_bins=n_bins)
        ax1.plot(prob_pred, prob_true, "s-", color=colour, lw=1.5,
                 label=name, markersize=5)

        # Histogram of predicted probabilities
        ax2.hist(y_proba, bins=30, range=(0, 1), alpha=0.3,
                 color=colour, label=name)

    # Perfect calibration diagonal
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
             label="Perfectly calibrated")
    ax1.set_xlabel("Mean predicted probability", fontsize=11)
    ax1.set_ylabel("Fraction of positives", fontsize=11)
    ax1.set_title("Calibration (Reliability Diagram)",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.05])

    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Probability Distribution", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax2.grid(alpha=0.25, linestyle="--")

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 7. Class distribution (before/after SMOTE) ─────────────

def plot_class_distribution(data_prep, smote_y=None):
    """
    Side-by-side bar charts showing class counts before and after SMOTE.

    Parameters
    ----------
    data_prep : dict
        Output of prepare_binary (must contain 'y_train').
    smote_y : array-like, optional
        y labels after SMOTE resampling.  If None, only the
        original distribution is shown.
    """
    y_before = data_prep["y_train"]
    panels = 2 if smote_y is not None else 1

    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 5))
    if panels == 1:
        axes = [axes]

    # ── Before SMOTE ──
    vc = pd.Series(y_before).value_counts().sort_index()
    axes[0].bar(vc.index.astype(str), vc.values,
                color=[_PALETTE[0], _PALETTE[1]][:len(vc)], edgecolor="black")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + max(vc) * 0.01, f"{v:,}", ha="center", fontsize=10)
    axes[0].set_title("Before SMOTE", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    # ── After SMOTE ──
    if smote_y is not None:
        vc2 = pd.Series(smote_y).value_counts().sort_index()
        axes[1].bar(vc2.index.astype(str), vc2.values,
                    color=[_PALETTE[2], _PALETTE[3]][:len(vc2)], edgecolor="black")
        for i, v in enumerate(vc2.values):
            axes[1].text(i, v + max(vc2) * 0.01, f"{v:,}", ha="center", fontsize=10)
        axes[1].set_title("After SMOTE", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Class"); axes[1].set_ylabel("Count")
        axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Class Distribution — Binary Target",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 8. Temporal: actual vs predicted scatter ─────────────────

def plot_actual_vs_predicted(temporal_results, target="duration"):
    """
    Scatter plot: actual (x) vs predicted (y) days for every temporal model.
    A perfect model sits on the diagonal.

    Parameters
    ----------
    temporal_results : dict
        Output of temporal_prediction().
    target : str
        'duration'.
    """
    sub = temporal_results.get(target)
    if sub is None:
        print(f"  No temporal results for '{target}'"); return

    results = sub["results"]
    y_actual = sub["y_test"]

    # Pick top-6 models by R²
    ranked = sorted(results.keys(),
                    key=lambda nm: results[nm].get("r2", -999), reverse=True)
    show = ranked[:6]
    if not show:
        print(f"  No temporal models to plot for '{target}'"); return
    n = len(show)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"Actual vs Predicted — {target.replace('_', ' ').title()}",
                 fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_1d(axes).flatten()

    lo = 0
    hi = max(y_actual.max(), max(results[nm]["y_pred"].max() for nm in show)) * 1.05

    for idx, name in enumerate(show):
        ax = axes[idx]
        yp = results[name]["y_pred"]
        r2 = results[name]["r2"]
        mae = results[name]["mae"]
        colour = _PALETTE[idx % len(_PALETTE)]

        ax.scatter(y_actual, yp, alpha=0.3, s=12, color=colour, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_title(f"{name}\nR²={r2:.3f}  MAE={mae:.1f}d",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual (days)"); ax.set_ylabel("Predicted (days)")
        ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2, linestyle="--")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 9. Temporal: residual distribution ───────────────────────

def plot_residuals(temporal_results, target="duration"):
    """
    Histogram + KDE of residuals (actual − predicted) for top models.

    Parameters
    ----------
    temporal_results : dict
        Output of temporal_prediction().
    target : str
        'duration'.
    """
    sub = temporal_results.get(target)
    if sub is None:
        print(f"  No temporal results for '{target}'"); return

    results = sub["results"]
    y_actual = sub["y_test"]

    ranked = sorted(results.keys(),
                    key=lambda n: results[n].get("r2", -999), reverse=True)
    show = ranked[:6]
    n = len(show)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    fig.suptitle(f"Residual Distribution — {target.replace('_', ' ').title()}",
                 fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_1d(axes).flatten()

    for idx, name in enumerate(show):
        ax = axes[idx]
        residuals = y_actual - results[name]["y_pred"]
        colour = _PALETTE[idx % len(_PALETTE)]

        sns.histplot(residuals, bins=40, kde=True, color=colour, ax=ax,
                     edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", lw=1.2, alpha=0.7)
        med = float(np.median(residuals))
        ax.axvline(med, color="orange", linestyle=":", lw=1,
                   label=f"median={med:.1f}d")
        ax.set_title(f"{name}\nMAE={results[name]['mae']:.1f}d",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Residual (days)")
        ax.legend(fontsize=8); ax.grid(alpha=0.2, linestyle="--")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 10. Forecast timeline (multi-panel dashboard) ───────────

def plot_forecast_timeline(forecast_results):
    """
    4-panel forecast dashboard:

      (A) Stacked bar: predicted visits/month by retrofit type, with
          calibrated totals (black line) and naive baseline (red dashed).
      (B) Retrofit-type distribution: horizontal bar showing share of
          each predicted type across the full forecast horizon.
      (C) Duration box-plot by retrofit type: predicted duration (days)
          distribution per type — shows the median and spread.
      (D) Confidence histogram: distribution of binary probability
          (prob_coming) for the vehicles selected for forecasting.

    Parameters
    ----------
    forecast_results : dict
        Output of ``forecast.run_forecast()``.  Expected keys:
        ``pivot_future``, ``pivot_calibrated``, ``naive_baseline``,
        ``df_forecast``, ``n_unseen``, ``n_vehicles``, ``n_visits``,
        ``threshold``, ``model_metadata``.
    """
    pivot = forecast_results.get("pivot_future")
    if pivot is None or pivot.empty:
        print("  No future forecast data to plot"); return

    pivot_cal = forecast_results.get("pivot_calibrated")
    naive     = forecast_results.get("naive_baseline")
    df        = forecast_results.get("df_forecast")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Forecast Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # ── Panel A: stacked bar (visits per month by type) ──────
    ax_a = axes[0, 0]
    pivot.plot(kind="bar", stacked=True, ax=ax_a, colormap="tab20",
               edgecolor="white", linewidth=0.4)

    totals = pivot.sum(axis=1)
    for i, (month, total) in enumerate(totals.items()):
        ax_a.text(i, total + max(totals) * 0.01, f"{int(total)}",
                  ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Calibrated totals overlay
    if pivot_cal is not None and not pivot_cal.empty:
        cal_totals = pivot_cal.sum(axis=1).reindex(pivot.index).dropna()
        if len(cal_totals):
            x_pos = [list(pivot.index).index(m) for m in cal_totals.index
                     if m in pivot.index]
            ax_a.plot(x_pos, cal_totals.values, "ko-", lw=2, markersize=4,
                      label="Calibrated total", zorder=5)

    # Naive baseline overlay
    if naive is not None:
        naive_pivot = naive.get("pivot_naive")
        if naive_pivot is not None and not naive_pivot.empty:
            naive_totals = naive_pivot.sum(axis=1).reindex(pivot.index).dropna()
            if len(naive_totals):
                x_pos_n = [list(pivot.index).index(m) for m in naive_totals.index
                           if m in pivot.index]
                ax_a.plot(x_pos_n, naive_totals.values, "r--", lw=2.5,
                          alpha=0.8, label="Naive baseline", zorder=4)

    ax_a.set_title("(A) Visits per Month by Retrofit Type",
                   fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Month"); ax_a.set_ylabel("Number of Visits")
    ax_a.legend(title="Type / Baseline", fontsize=7, title_fontsize=8,
                bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_a.tick_params(axis="x", rotation=45)
    ax_a.grid(axis="y", alpha=0.25, linestyle="--")

    # ── Panel B: type distribution (horizontal bar) ──────────
    ax_b = axes[0, 1]
    type_counts = pivot.sum(axis=0).sort_values(ascending=True)
    colours_b = plt.cm.tab20(np.linspace(0, 1, len(type_counts)))
    bars_b = ax_b.barh(type_counts.index.astype(str), type_counts.values,
                       color=colours_b, edgecolor="white", linewidth=0.5)
    total_visits = type_counts.sum()
    for bar, val in zip(bars_b, type_counts.values):
        pct = val / total_visits * 100 if total_visits else 0
        ax_b.text(bar.get_width() + max(type_counts.values) * 0.01,
                  bar.get_y() + bar.get_height() / 2,
                  f"{int(val)}  ({pct:.1f}%)", va="center", fontsize=8)
    ax_b.set_title("(B) Total Forecast by Retrofit Type",
                   fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Total Predicted Visits")
    ax_b.grid(axis="x", alpha=0.25, linestyle="--")

    # ── Panel C: duration box-plot by type ───────────────────
    ax_c = axes[1, 0]
    if df is not None and config.COL_PREDICTED_DURATION in df.columns and config.COL_PREDICTED_TYPE in df.columns:
        dur_data = df[[config.COL_PREDICTED_TYPE, config.COL_PREDICTED_DURATION]].dropna()
        if len(dur_data) > 0:
            type_order = (dur_data.groupby(config.COL_PREDICTED_TYPE)[config.COL_PREDICTED_DURATION]
                          .median().sort_values().index.tolist())
            sns.boxplot(data=dur_data, x=config.COL_PREDICTED_TYPE, y=config.COL_PREDICTED_DURATION,
                        order=type_order, ax=ax_c, palette="tab20",
                        fliersize=2, linewidth=0.8)
            # Overlay medians as text
            medians = dur_data.groupby(config.COL_PREDICTED_TYPE)[config.COL_PREDICTED_DURATION].median()
            for i, t in enumerate(type_order):
                med = medians.get(t, 0)
                ax_c.text(i, med + 1, f"{med:.0f}d", ha="center", va="bottom",
                          fontsize=7, fontweight="bold", color="black")
            ax_c.set_title("(C) Predicted Duration by Retrofit Type",
                           fontsize=12, fontweight="bold")
            ax_c.set_xlabel("Retrofit Type"); ax_c.set_ylabel("Duration (days)")
            ax_c.tick_params(axis="x", rotation=45)
            ax_c.grid(axis="y", alpha=0.25, linestyle="--")
        else:
            ax_c.text(0.5, 0.5, "No duration data", ha="center", va="center",
                      transform=ax_c.transAxes, fontsize=12, color="gray")
            ax_c.set_title("(C) Duration — N/A", fontsize=12)
    else:
        ax_c.text(0.5, 0.5, "No duration predictions available",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=12, color="gray")
        ax_c.set_title("(C) Duration — N/A", fontsize=12)

    # ── Panel D: confidence histogram (prob_coming) ──────────
    ax_d = axes[1, 1]
    if df is not None and config.COL_PROB_COMING in df.columns:
        probs = df[config.COL_PROB_COMING].dropna()
        if len(probs) > 0:
            sns.histplot(probs, bins=30, kde=True, ax=ax_d,
                         color=_PALETTE[0], edgecolor="white", linewidth=0.4)
            threshold = forecast_results.get("threshold", 0.5)
            ax_d.axvline(threshold, color="red", linestyle="--", lw=1.5,
                         label=f"Threshold = {threshold:.2f}")
            ax_d.axvline(probs.median(), color="orange", linestyle=":", lw=1.5,
                         label=f"Median = {probs.median():.2f}")
            ax_d.set_title("(D) Binary Probability Distribution (Forecast Vehicles)",
                           fontsize=12, fontweight="bold")
            ax_d.set_xlabel("P(comes for retrofit)")
            ax_d.set_ylabel("Count")
            ax_d.legend(fontsize=9)
            ax_d.grid(axis="y", alpha=0.25, linestyle="--")
        else:
            ax_d.text(0.5, 0.5, "No prob_coming data", ha="center", va="center",
                      transform=ax_d.transAxes, fontsize=12, color="gray")
            ax_d.set_title("(D) Confidence — N/A", fontsize=12)
    else:
        ax_d.text(0.5, 0.5, "No probability data available",
                  ha="center", va="center", transform=ax_d.transAxes,
                  fontsize=12, color="gray")
        ax_d.set_title("(D) Confidence — N/A", fontsize=12)

    # ── Summary annotation ───────────────────────────────────
    n_unseen  = int(forecast_results.get("n_unseen", 0))
    n_veh     = int(forecast_results.get("n_vehicles", 0))
    n_vis     = int(forecast_results.get("n_visits", 0))
    meta      = forecast_results.get("model_metadata", {})
    summary   = (f"Unseen vehicles: {n_unseen:,}  |  "
                 f"Predicted to come: {n_veh:,}  |  "
                 f"Total visits: {n_vis:,}  |  "
                 f"Binary: {meta.get('binary_model', '?')}  |  "
                 f"Multiclass: {meta.get('multiclass_model', '?')}  |  "
                 f"Duration: {meta.get('temporal_duration_model', '?')}")
    fig.text(0.5, 0.005, summary, ha="center", va="bottom",
             fontsize=9, style="italic", color="gray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show(); plt.close("all"); gc.collect()


# ── 13. SHAP Summary (beeswarm) ─────────────────────────────

def plot_shap_summary(baseline_results, model_name=None, max_display=20):
    """
    SHAP beeswarm plot showing direction-of-effect for each feature.

    Uses TreeExplainer for tree models (fast) and falls back to
    KernelExplainer/Explainer for non-tree models.

    Parameters
    ----------
    baseline_results : dict
        Output of binary_classification / smote / tuning.
        Must contain 'results', 'X_train', 'X_test', 'feature_cols'.
    model_name : str, optional
        Which model to explain. Defaults to best by ROC-AUC.
    max_display : int
        Number of features to show (default 20).

    Returns
    -------
    shap.Explanation  – SHAP values (reusable for dependence plots).
    """
    import shap

    results = baseline_results["results"]
    X_tr = baseline_results["X_train"]
    X_te = baseline_results["X_test"]

    # Skip API-based models for SHAP (extremely slow, not tree-based)
    _SKIP_SHAP = {"LLM-Embed LogReg", "LLM-Prompted"}
    if model_name is None:
        model_name = max((n for n in results if n not in _SKIP_SHAP),
                         key=lambda n: results[n].get("roc_auc", 0))
    model = results[model_name]["model"]
    print(f"  SHAP Summary for: {model_name}")

    # Pick the right explainer
    X_plot = X_te                          # features matrix for summary_plot
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_te)
    except Exception:
        # Subsample for non-tree models (KernelExplainer is slow)
        bg = shap.sample(X_tr, min(100, len(X_tr)))
        explainer = shap.Explainer(model.predict_proba, bg)
        n_sub = min(500, len(X_te))
        X_plot = X_te[:n_sub]              # keep in sync with SHAP values
        shap_values = explainer(X_plot)

    # For binary classifiers TreeExplainer may return 3-D array (n, f, 2)
    vals = shap_values
    if hasattr(vals, "values") and vals.values.ndim == 3:
        vals = vals[..., 1]  # positive-class slice

    fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.35)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*global RNG.*", category=FutureWarning)
        shap.summary_plot(vals, X_plot, plot_type="dot",
                          max_display=max_display, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()
    return shap_values


# ── 14. SHAP Dependence (top features) ──────────────────────

def plot_shap_dependence(baseline_results, shap_values=None,
                         model_name=None, top_n=5):
    """
    SHAP dependence plots for the top-N most impactful features.

    Each subplot shows feature value (x) vs SHAP value (y) coloured
    by the strongest interaction feature (auto-detected by SHAP).

    Parameters
    ----------
    baseline_results : dict
        Same as plot_shap_summary.
    shap_values : shap.Explanation, optional
        Pre-computed SHAP values (from plot_shap_summary return).
        If None, they are recomputed.
    model_name : str, optional
        Which model. Defaults to best by ROC-AUC.
    top_n : int
        Number of top features to plot (default 5).
    """
    import shap

    results = baseline_results["results"]
    X_tr = baseline_results["X_train"]
    X_te = baseline_results["X_test"]

    _SKIP_SHAP = {"LLM-Embed LogReg", "LLM-Prompted"}
    if model_name is None:
        model_name = max((n for n in results if n not in _SKIP_SHAP),
                         key=lambda n: results[n].get("roc_auc", 0))
    model = results[model_name]["model"]

    # Compute SHAP if not provided
    if shap_values is None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_te)
        except Exception:
            bg = shap.sample(X_tr, min(100, len(X_tr)))
            explainer = shap.Explainer(model.predict_proba, bg)
            n_sub = min(500, len(X_te))
            X_te = X_te[:n_sub]
            shap_values = explainer(X_te)
    else:
        # Align X_te to pre-computed shap_values row count
        _sv = shap_values
        if hasattr(_sv, "values"):
            _n = _sv.values.shape[0]
        else:
            _n = len(_sv)
        if len(X_te) > _n:
            X_te = X_te[:_n]

    vals = shap_values
    if hasattr(vals, "values") and vals.values.ndim == 3:
        vals = vals[..., 1]

    # Identify top-N features by mean |SHAP|
    mean_abs = np.abs(vals.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    feature_names = (X_te.columns.tolist()
                     if hasattr(X_te, "columns")
                     else [f"Feature {i}" for i in range(vals.values.shape[1])])

    ncols = min(3, top_n)
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"SHAP Dependence — {model_name} (top {top_n})",
                 fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_1d(axes).flatten()

    for idx_pos, feat_idx in enumerate(top_idx):
        ax = axes[idx_pos]
        fname = feature_names[feat_idx]
        shap.dependence_plot(
            feat_idx, vals.values,
            X_te if hasattr(X_te, "iloc") else pd.DataFrame(X_te, columns=feature_names),
            interaction_index="auto", ax=ax, show=False,
        )
        ax.set_title(fname, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2, linestyle="--")

    for idx_pos in range(top_n, len(axes)):
        axes[idx_pos].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 15. Partial Dependence + ICE ─────────────────────────────

def plot_partial_dependence(baseline_results, model_name=None,
                            feature_names=None, top_n=6, subsample=100):
    """
    PDP + ICE (Individual Conditional Expectation) for top features.

    Parameters
    ----------
    baseline_results : dict
        Must contain 'results', 'X_test', 'feature_cols'.
    model_name : str, optional
        Which model to explain. Defaults to best tree model by AUC.
    feature_names : list[str], optional
        Specific features to plot. If None, auto-selects top-N by
        tree-based feature_importances_.
    top_n : int
        Number of features if auto-selecting (default 6).
    subsample : int
        Number of ICE lines to draw per feature (default 100).
    """
    results = baseline_results["results"]
    X_te = baseline_results["X_test"]
    feat_cols = baseline_results["feature_cols"]

    # Pick a tree model (PDP requires predict / predict_proba)
    if model_name is None:
        tree_candidates = [n for n in results
                           if hasattr(results[n].get("model"), "feature_importances_")]
        if not tree_candidates:
            print("  No tree model found for PDP — skipping"); return
        model_name = max(tree_candidates,
                         key=lambda n: results[n].get("roc_auc", 0))
    model = results[model_name]["model"]
    print(f"  PDP + ICE for: {model_name}")

    # Auto-select features by importance
    if feature_names is None:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[-top_n:][::-1]
            feature_names = [feat_cols[i] for i in top_idx]
        else:
            feature_names = feat_cols[:top_n]

    # Clip extreme values to 1st–99th percentile to prevent float32
    # overflow in sklearn's PDP grid evaluation (affects features like
    # age_x_fleet, av_age_months, basistyp_freq with extreme outliers).
    X_pdp = X_te.copy()
    for col in X_pdp.columns:
        if pd.api.types.is_numeric_dtype(X_pdp[col]):
            lo, hi = X_pdp[col].quantile([0.01, 0.99])
            if lo < hi:
                X_pdp[col] = X_pdp[col].clip(lo, hi)
    # Upcast to float64 so PDP grid arithmetic doesn't overflow
    X_pdp = X_pdp.astype("float64")

    n = len(feature_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"Partial Dependence + ICE — {model_name}",
                 fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_1d(axes).flatten()

    for idx, fname in enumerate(feature_names):
        ax = axes[idx]
        try:
            PartialDependenceDisplay.from_estimator(
                model, X_pdp, [fname],
                kind="both",         # PDP line + ICE
                subsample=subsample,
                ax=ax,
                ice_lines_kw={"color": _PALETTE[0], "alpha": 0.08, "lw": 0.5},
                pd_line_kw={"color": "red", "lw": 2.5},
            )
            ax.set_title(fname, fontsize=11, fontweight="bold")
            ax.grid(alpha=0.2, linestyle="--")
        except Exception as exc:
            ax.set_visible(False)
            print(f"  ✗ PDP for {fname}: {exc}")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 16a. Forest plot: bootstrap CIs for AUC & F1 ────────────

def plot_model_comparison_forest(binary_results):
    """
    Forest plot showing AUC and F1 with 95 % bootstrap confidence intervals
    for every (approach × model) combination, plus McNemar significance flags.

    Parameters
    ----------
    binary_results : dict
        Output of evaluation.final_model_comparison().
        Must contain 'comparison' (DataFrame), 'bootstrap_cis' (dict),
        and optionally 'mcnemar_results' (DataFrame).
    """
    df = binary_results["comparison"]
    cis = binary_results.get("bootstrap_cis", {})
    mcnemar_df = binary_results.get("mcnemar_results")

    # Build a lookup for McNemar significance (winner vs other)
    sig_set = set()
    if mcnemar_df is not None and not mcnemar_df.empty:
        for _, row in mcnemar_df.iterrows():
            if row.get("sig (α=.05)") == "✓":
                sig_set.add(row["Winner vs"])

    # Prepare data: one row per model, sorted by Composite descending
    models = []
    for _, row in df.iterrows():
        key = f"{row['Approach']}|{row['Model']}"
        ci = cis.get(key)
        label = f"{row['Approach']}: {row['Model']}"
        auc_pt = row["ROC-AUC"]
        f1_pt = row.get("F1-Opt", row.get("F1-Score", 0))
        auc_lo = ci["auc"]["ci_lo"] if ci else auc_pt
        auc_hi = ci["auc"]["ci_hi"] if ci else auc_pt
        f1_lo = ci["f1"]["ci_lo"] if ci else f1_pt
        f1_hi = ci["f1"]["ci_hi"] if ci else f1_pt
        is_sig = row["Model"] in sig_set
        models.append({
            "label": label, "auc": auc_pt, "auc_lo": auc_lo, "auc_hi": auc_hi,
            "f1": f1_pt, "f1_lo": f1_lo, "f1_hi": f1_hi,
            "significant": is_sig,
        })

    n = len(models)
    if n == 0:
        print("  No models to plot"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, n * 0.4)),
                                    sharey=True)
    fig.suptitle("Model Comparison — 95 % Bootstrap Confidence Intervals",
                 fontsize=15, fontweight="bold", y=1.02)

    y_pos = np.arange(n)
    labels = [m["label"] for m in models]

    # ── AUC panel ────────────────────────────────────────────
    for i, m in enumerate(models):
        colour = _PALETTE[0] if i == 0 else (_PALETTE[3] if m["significant"] else _PALETTE[1])
        marker = "D" if i == 0 else "o"
        ax1.errorbar(
            m["auc"], i,
            xerr=[[max(m["auc"] - m["auc_lo"], 0)], [max(m["auc_hi"] - m["auc"], 0)]],
            fmt=marker, color=colour, capsize=4, capthick=1.5, lw=1.5,
            markersize=7 if i == 0 else 5,
        )
        # Annotate point value
        ax1.text(m["auc_hi"] + 0.003, i, f"{m['auc']:.3f}",
                 va="center", fontsize=7, color=colour)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("ROC-AUC", fontsize=11)
    ax1.set_title("ROC-AUC with 95% CI", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25, linestyle="--")
    ax1.axvline(0.5, color="grey", linestyle=":", lw=0.8, alpha=0.5)

    # ── F1 panel ─────────────────────────────────────────────
    for i, m in enumerate(models):
        colour = _PALETTE[0] if i == 0 else (_PALETTE[3] if m["significant"] else _PALETTE[1])
        marker = "D" if i == 0 else "o"
        ax2.errorbar(
            m["f1"], i,
            xerr=[[max(m["f1"] - m["f1_lo"], 0)], [max(m["f1_hi"] - m["f1"], 0)]],
            fmt=marker, color=colour, capsize=4, capthick=1.5, lw=1.5,
            markersize=7 if i == 0 else 5,
        )
        ax2.text(m["f1_hi"] + 0.003, i, f"{m['f1']:.3f}",
                 va="center", fontsize=7, color=colour)

    ax2.invert_yaxis()
    ax2.set_xlabel("F1-Score", fontsize=11)
    ax2.set_title("F1-Score with 95% CI", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.25, linestyle="--")

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="D", color=_PALETTE[0], linestyle="None",
               markersize=7, label="Winner"),
        Line2D([0], [0], marker="o", color=_PALETTE[1], linestyle="None",
               markersize=5, label="Not sig. different (McNemar)"),
        Line2D([0], [0], marker="o", color=_PALETTE[3], linestyle="None",
               markersize=5, label="Sig. different (p < 0.05)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()


# ── 16b. Logistic Regression Coefficients ────────────────────

def plot_lr_coefficients(baseline_results, top_n=25):
    """
    Horizontal bar chart of Logistic Regression coefficients.

    Positive coefficients push the prediction toward *Retrofit* (class 1),
    negative toward *No Retrofit* (class 0).  Bars are coloured by sign.

    Parameters
    ----------
    baseline_results : dict
        Must contain 'results' with a "Logistic Regression" entry
        and 'feature_cols'.
    top_n : int
        Number of largest-|coef| features to display (default 25).
    """
    lr_entry = baseline_results["results"].get("Logistic Regression")
    if lr_entry is None:
        print("  Logistic Regression not found in results — skipping"); return

    model = lr_entry["model"]

    # Extract coefficients (sklearn LR stores shape (1, n_features) for binary)
    from sklearn.linear_model import LogisticRegression as _LR
    if not isinstance(model, _LR):
        print("  Model is not a bare LogisticRegression — skipping"); return

    coefs = model.coef_[0]
    feature_cols = baseline_results["feature_cols"]

    df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).head(top_n)

    # Sort by coefficient value for the plot (not abs)
    df = df.sort_values("coefficient", ascending=True)

    colours = [_PALETTE[0] if c > 0 else _PALETTE[3] for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(len(df)), df["coefficient"], color=colours, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (log-odds)", fontsize=11)
    ax.set_title("Logistic Regression — Feature Coefficients (top by |coef|)",
                 fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=_PALETTE[0], label="→ Retrofit (positive)"),
        Patch(facecolor=_PALETTE[3], label="→ No Retrofit (negative)"),
    ], loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

    # Print summary table
    df_print = df.sort_values("abs_coef", ascending=False)[["feature", "coefficient"]].reset_index(drop=True)
    print(f"\nTop {len(df_print)} LR coefficients (by |coef|):")
    print(df_print.to_string(index=False))
    return df_print


# ── 17. Temporal: Q-Q plot for residual normality ────────────

def plot_qq_residuals(temporal_results, target="duration"):
    """
    Q-Q (quantile-quantile) plot of regression residuals against a
    normal theoretical distribution.  Points falling on the diagonal
    indicate normally distributed residuals — a key OLS assumption.

    Also annotates each subplot with the Shapiro-Wilk test statistic
    and p-value (on a random subsample of 5 000 points when n > 5 000,
    because Shapiro-Wilk is limited to n ≤ 5 000).

    Parameters
    ----------
    temporal_results : dict
        Output of temporal_prediction().
    target : str
        'duration'.
    """
    sub = temporal_results.get(target)
    if sub is None:
        print(f"  No temporal results for '{target}'"); return

    results = sub["results"]
    y_actual = sub["y_test"]

    # Pick top-6 models by R²
    ranked = sorted(results.keys(),
                    key=lambda n: results[n].get("r2", -999), reverse=True)
    show = ranked[:6]
    n = len(show)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(
        f"Q-Q Plot of Residuals — {target.replace('_', ' ').title()}",
        fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_1d(axes).flatten()

    for idx, name in enumerate(show):
        ax = axes[idx]
        residuals = np.asarray(y_actual) - np.asarray(results[name]["y_pred"])
        colour = _PALETTE[idx % len(_PALETTE)]

        # scipy.stats.probplot draws Q-Q against normal
        (osm, osr), (slope, intercept, _) = sp_stats.probplot(
            residuals, dist="norm")
        ax.scatter(osm, osr, s=10, alpha=0.4, color=colour, edgecolors="none")
        # Reference line
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.2, alpha=0.7)

        # Shapiro-Wilk test (subsample if n > 5000)
        _rng = np.random.RandomState(42)
        _samp = (residuals if len(residuals) <= 5000
                 else _rng.choice(residuals, size=5000, replace=False))
        sw_stat, sw_p = sp_stats.shapiro(_samp)

        ax.set_title(
            f"{name}\nShapiro-Wilk W={sw_stat:.4f}, p={sw_p:.2e}",
            fontsize=10, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Residuals (days)")
        ax.grid(alpha=0.2, linestyle="--")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()
    print(f"  ✓ Q-Q residual plot ({target}) — "
          f"{len(show)} models shown")


# ── 18. Temporal: heteroscedasticity check ───────────────────

def plot_heteroscedasticity(temporal_results, target="duration"):
    """
    Residuals-vs-fitted scatter plot for the top temporal models.

    Heteroscedasticity = the variance of the residuals changes with
    the predicted value.  If the spread of the point cloud widens
    (or narrows) along the x-axis, the assumption of homoscedastic
    errors is violated.

    Each subplot also shows:
      • A LOWESS smoother (orange) to reveal any systematic trend
        in the residual magnitude.
      • The Breusch-Pagan test statistic (LM) and p-value.
        H₀: constant variance.  p < 0.05 → reject → heteroscedastic.

    Parameters
    ----------
    temporal_results : dict
        Output of temporal_prediction().
    target : str
        'duration'.
    """
    sub = temporal_results.get(target)
    if sub is None:
        print(f"  No temporal results for '{target}'"); return

    results = sub["results"]
    y_actual = np.asarray(sub["y_test"])

    # Top-6 models by R²
    ranked = sorted(results.keys(),
                    key=lambda n: results[n].get("r2", -999), reverse=True)
    show = ranked[:6]
    n = len(show)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(
        f"Residuals vs Fitted — {target.replace('_', ' ').title()}",
        fontsize=15, fontweight="bold", y=1.02)
    axes = np.atleast_1d(axes).flatten()

    for idx, name in enumerate(show):
        ax = axes[idx]
        y_pred = np.asarray(results[name]["y_pred"])
        residuals = y_actual - y_pred
        colour = _PALETTE[idx % len(_PALETTE)]

        # Scatter: fitted (x) vs residuals (y)
        ax.scatter(y_pred, residuals, s=10, alpha=0.3,
                   color=colour, edgecolors="none")
        ax.axhline(0, color="red", linestyle="--", lw=1.2, alpha=0.6)

        # LOWESS smoother (statsmodels is already a dependency via temporal.py)
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(
                residuals, y_pred, frac=0.3, return_sorted=True)
            ax.plot(lowess[:, 0], lowess[:, 1], color="orange",
                    lw=2, label="LOWESS")
        except Exception:
            pass  # graceful fallback if statsmodels unavailable

        # Breusch-Pagan test
        bp_text = ""
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            # OLS residuals vs fitted (constant + fitted)
            _exog = sm.add_constant(y_pred)
            lm_stat, lm_p, f_stat, f_p = het_breuschpagan(residuals, _exog)
            bp_text = f"BP LM={lm_stat:.1f}, p={lm_p:.2e}"
        except Exception:
            bp_text = "BP: n/a"

        ax.set_title(
            f"{name}\n{bp_text}",
            fontsize=10, fontweight="bold")
        ax.set_xlabel("Fitted (days)")
        ax.set_ylabel("Residual (days)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2, linestyle="--")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()
    print(f"  ✓ Heteroscedasticity check ({target}) — "
          f"{len(show)} models shown")


# ── 19. Temporal: prediction error by vehicle / retrofit type ─

def plot_error_by_vehicle_type(temporal_results, target="duration",
                               model=None, top_n=15):
    """
    Box-plot of absolute prediction errors grouped by retrofit type
    (umr_art_clean).  Reveals which vehicle/retrofit categories the
    model struggles with most — useful for targeted re-training or
    business-rule overrides.

    Also prints a summary table sorted by median absolute error.

    Parameters
    ----------
    temporal_results : dict
        Output of temporal_prediction().
    target : str
        'duration'.
    model : str or None
        Specific model name to analyse.  If None, uses best_overall.
    top_n : int
        Show only the top-N types with highest median |error|.
    """
    sub = temporal_results.get(target)
    if sub is None:
        print(f"  No temporal results for '{target}'"); return

    results = sub["results"]
    y_actual = np.asarray(sub["y_test"])
    types = sub.get("y_test_types")

    if types is None:
        print("  ⚠ Vehicle-type info not available in results "
              "(y_test_types missing). Re-run temporal_prediction().")
        return

    # Choose model
    name = model or sub.get("best_overall", list(results.keys())[0])
    if name not in results:
        print(f"  Model '{name}' not found in results"); return

    y_pred = np.asarray(results[name]["y_pred"])
    abs_err = np.abs(y_actual - y_pred)

    # Build DataFrame
    err_df = pd.DataFrame({
        "type": types,
        "abs_error": abs_err,
        "actual": y_actual,
        "predicted": y_pred,
    })

    # Aggregate stats per type
    stats = (
        err_df.groupby("type")["abs_error"]
        .agg(["median", "mean", "std", "count"])
        .rename(columns={"median": "median_ae", "mean": "mean_ae",
                         "std": "std_ae", "count": "n"})
        .sort_values("median_ae", ascending=False)
    )

    # ── Box plot (top-N worst types) ─────────────────────────
    top_types = stats.head(top_n).index.tolist()
    plot_df = err_df[err_df["type"].isin(top_types)].copy()

    # Order boxes by median error (worst first)
    order = stats.loc[top_types].index.tolist()

    fig, ax = plt.subplots(figsize=(12, max(5, len(order) * 0.45)))
    sns.boxplot(
        data=plot_df, y="type", x="abs_error", order=order,
        palette=_PALETTE[:len(order)], orient="h",
        fliersize=2, linewidth=0.8, ax=ax,
    )

    # Annotate sample size
    for i, t in enumerate(order):
        n_samples = int(stats.loc[t, "n"])
        med = stats.loc[t, "median_ae"]
        ax.text(med + ax.get_xlim()[1] * 0.01, i,
                f" n={n_samples:,}", va="center", fontsize=8,
                color="grey")

    ax.set_xlabel("Absolute Error (days)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        f"Prediction Error by Retrofit Type — {name}\n"
        f"({target.replace('_', ' ').title()}, top {len(order)} worst types)",
        fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

    # ── Summary table ────────────────────────────────────────
    stats_print = stats.round(1).reset_index()
    stats_print.columns = ["Retrofit Type", "Median AE", "Mean AE",
                           "Std AE", "N"]
    stats_print["N"] = stats_print["N"].astype(int)
    print(f"\n  Error by vehicle type — {name} ({target}):")
    print(f"  {'─' * 70}")
    print(stats_print.to_string(index=False))
    print(f"  {'─' * 70}")
    print(f"  ✓ {len(stats)} types analysed, "
          f"worst median AE = {stats['median_ae'].iloc[0]:.1f} days "
          f"({stats.index[0]})")
    return stats_print