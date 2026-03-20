# ============================================================
# 07_data_modeling_and_evaluation_automl.py -- AutoML Benchmark
#
# Runs AutoGluon + TabPFN on the same train/test splits
# produced by 02_feature_engineering.py, then compares against
# the manually-tuned models from modules 03/04/05.
#
# Structure
# ---------
#   Section A  -- Binary classification    (ROC-AUC)
#   Section A2 -- TabPFN binary
#   Section B  -- Multiclass classification (F1-weighted)
#   Section B2 -- TabPFN multiclass
#   Section C  -- Temporal regression       (MAE in days)
#   Section C2 -- TabPFN temporal
#   Section D -- Comparison utilities
#   Section E -- Orchestrator             (run_automl_benchmark)
# ============================================================
import gc
import os
import shutil
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from . import config

# AutoGluon -- graceful fallback if not installed
try:
    from autogluon.tabular import TabularPredictor
    _AUTOGLUON = True
except ImportError:
    _AUTOGLUON = False

# TabPFN -- pre-trained tabular foundation model (sklearn-compatible)
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    _TABPFN = True
except ImportError:
    _TABPFN = False

_TABPFN_MAX_TRAIN = 10_000   # TabPFN training-set ceiling


def _make_tabpfn_classifier(X_tr, y_tr, **kwargs):
    """Create + fit TabPFNClassifier, falling back to v2 if v2.5 fails."""
    try:
        clf = TabPFNClassifier(**kwargs)
        clf.fit(X_tr, y_tr)
        return clf
    except Exception as exc:
        if "download" in str(exc).lower() or "authentication" in str(exc).lower() \
                or "gated" in str(exc).lower() or "V2_5" in str(exc):
            from tabpfn.constants import ModelVersion
            print("  [TabPFN] v2.5 unavailable (gated model) → falling back to v2")
            clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
            clf.fit(X_tr, y_tr)
            return clf
        raise


def _make_tabpfn_regressor(X_tr, y_tr, **kwargs):
    """Create + fit TabPFNRegressor, falling back to v2 if v2.5 fails."""
    try:
        reg = TabPFNRegressor(**kwargs)
        reg.fit(X_tr, y_tr)
        return reg
    except Exception as exc:
        if "download" in str(exc).lower() or "authentication" in str(exc).lower() \
                or "gated" in str(exc).lower() or "V2_5" in str(exc):
            from tabpfn.constants import ModelVersion
            print("  [TabPFN] v2.5 unavailable (gated model) → falling back to v2")
            reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
            reg.fit(X_tr, y_tr)
            return reg
        raise
_TABPFN_MAX_FEAT  = 100      # TabPFN feature ceiling


_LABEL = "__target__"   # internal column name for AutoGluon label


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _build_ag_frames(X_train, y_train, X_test, y_test, label=_LABEL):
    """Concatenate features + target into single DataFrames for AutoGluon."""
    train_df = X_train.copy()
    train_df[label] = y_train.values if hasattr(y_train, "values") else y_train
    test_df = X_test.copy()
    test_df[label] = y_test.values if hasattr(y_test, "values") else y_test
    # AutoGluon cannot interpret pandas StringDtype — cast to plain object
    for df in (train_df, test_df):
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and str(df[col].dtype) != "object":
                df[col] = df[col].astype(object)
    return train_df, test_df


def _cleanup(path):
    """Remove AutoGluon model artifacts to free disk."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


# ── sklearn-compatible wrappers ──────────────────────────────
# These let the forecast module (08) call .predict() / .predict_proba()
# with the same API as manual sklearn models.

class _AGBinaryWrapper:
    """Wrap AutoGluon predictor to mimic sklearn binary classifier."""
    def __init__(self, predictor):
        self._pred = predictor

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        df = self._pred.predict_proba(X)
        pos = 1 if 1 in df.columns else df.columns[-1]
        neg = 0 if 0 in df.columns else df.columns[0]
        return np.column_stack([df[neg].values, df[pos].values])

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._pred.predict(X).values


class _AGMulticlassWrapper:
    """Wrap AutoGluon predictor to mimic sklearn multiclass classifier."""
    def __init__(self, predictor, classes):
        self._pred = predictor
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        df = self._pred.predict_proba(X)
        # Return columns in the same order as self.classes_
        cols = [c for c in self.classes_ if c in df.columns]
        return df[cols].values if cols else df.values

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._pred.predict(X).values


class _AGRegressionWrapper:
    """Wrap AutoGluon predictor to mimic sklearn regressor."""
    def __init__(self, predictor):
        self._pred = predictor

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._pred.predict(X).values


# ── TabPFN wrappers (feature-subset aware) ───────────────────

class _TabPFNBinaryWrapper:
    """Wrap TabPFN classifier + remember which feature subset was used."""
    def __init__(self, clf, feat_cols):
        self._clf = clf
        self._feat_cols = feat_cols

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        Xs = X[self._feat_cols] if set(self._feat_cols).issubset(X.columns) else X.iloc[:, :len(self._feat_cols)]
        return self._clf.predict_proba(Xs.values)

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        Xs = X[self._feat_cols] if set(self._feat_cols).issubset(X.columns) else X.iloc[:, :len(self._feat_cols)]
        return self._clf.predict(Xs.values)


class _TabPFNMulticlassWrapper:
    """Wrap TabPFN classifier for multiclass + remember feature subset."""
    def __init__(self, clf, feat_cols, classes):
        self._clf = clf
        self._feat_cols = feat_cols
        self.classes_ = np.array(classes)

    def _subset(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if set(self._feat_cols).issubset(X.columns):
            return X[self._feat_cols].values
        return X.iloc[:, :len(self._feat_cols)].values

    def predict_proba(self, X):
        return self._clf.predict_proba(self._subset(X))

    def predict(self, X):
        return self._clf.predict(self._subset(X))


class _TabPFNRegressionWrapper:
    """Wrap TabPFN regressor + remember which feature subset was used."""
    def __init__(self, reg, feat_cols):
        self._reg = reg
        self._feat_cols = feat_cols

    def _subset(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if set(self._feat_cols).issubset(X.columns):
            return X[self._feat_cols].values
        return X.iloc[:, :len(self._feat_cols)].values

    def predict(self, X):
        return self._reg.predict(self._subset(X))


# ══════════════════════════════════════════════════════════════
# Section A -- Binary Classification (ROC-AUC)
# ══════════════════════════════════════════════════════════════

def automl_binary(data_prep):
    """Run AutoGluon on the same binary split from prepare_binary().

    Parameters
    ----------
    data_prep : dict  returned by 02_feature_engineering.prepare_binary()
        Must contain X_train, X_test, y_train, y_test.

    Returns
    -------
    dict  with keys: roc_auc, accuracy, precision, recall, f1,
          leaderboard (DataFrame), elapsed_s, best_model.
    """
    if not _AUTOGLUON:
        print("  AutoGluon not installed -- skipping binary benchmark")
        return None

    print(f"\n{'='*70}")
    print("  AUTOML BINARY CLASSIFICATION")
    print(f"{'='*70}")

    train_df, test_df = _build_ag_frames(
        data_prep["X_train"], data_prep["y_train"],
        data_prep["X_test"], data_prep["y_test"],
    )

    save_path = os.path.join(config.AUTOML_SAVE_DIR, "binary")
    _cleanup(save_path)

    t0 = time.time()
    predictor = TabularPredictor(
        label=_LABEL, eval_metric="roc_auc",
        path=save_path, verbosity=1,
    ).fit(
        train_df,
        time_limit=config.AUTOML_TIME_LIMIT_BINARY,
        presets=config.AUTOML_PRESETS,
    )
    elapsed = time.time() - t0

    # Evaluate
    y_true = test_df[_LABEL]
    y_pred_cls = predictor.predict(test_df)
    y_pred_proba = predictor.predict_proba(test_df)
    # AutoGluon returns DataFrame; select positive-class column
    pos_col = 1 if 1 in y_pred_proba.columns else y_pred_proba.columns[-1]
    y_proba = y_pred_proba[pos_col]

    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred_cls)
    prec = precision_score(y_true, y_pred_cls, zero_division=0)
    rec = recall_score(y_true, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true, y_pred_cls, zero_division=0)

    lb = predictor.leaderboard(test_df, silent=True)
    best = getattr(predictor, "model_best", None) or lb.iloc[0]["model"]

    print(f"\n  AutoGluon Binary Results ({elapsed:.0f}s):")
    print(f"    Best model : {best}")
    print(f"    ROC-AUC    : {auc:.4f}")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    Precision  : {prec:.4f}")
    print(f"    Recall     : {rec:.4f}")
    print(f"    F1         : {f1:.4f}")
    print(f"\n  Leaderboard (top 5):")
    print(lb.head(5).to_string(index=False))

    result = {
        "roc_auc": auc, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "leaderboard": lb,
        "best_model": best, "elapsed_s": elapsed,
        "model": _AGBinaryWrapper(predictor),
        "_save_path": save_path,
    }

    gc.collect()
    return result


# ── TabPFN Binary ────────────────────────────────────────────

def tabpfn_binary(data_prep):
    """Run TabPFN on the same binary split from prepare_binary().

    TabPFN is a pre-trained transformer for tabular data — zero-shot,
    no hyperparameter tuning needed.  Sklearn-compatible API.

    Parameters
    ----------
    data_prep : dict  returned by 02_feature_engineering.prepare_binary()

    Returns
    -------
    dict  with keys: roc_auc, accuracy, precision, recall, f1,
          elapsed_s, best_model, model.
    """
    if not _TABPFN:
        print("  TabPFN not installed -- skipping binary TabPFN benchmark")
        return None

    print(f"\n{'='*70}")
    print("  TABPFN BINARY CLASSIFICATION")
    print(f"{'='*70}")

    X_train = data_prep["X_train"]
    y_train = data_prep["y_train"]
    X_test = data_prep["X_test"]
    y_test = data_prep["y_test"]

    # TabPFN limits: subsample training set if needed
    n_train = len(X_train)
    if n_train > _TABPFN_MAX_TRAIN:
        rng = np.random.RandomState(config.RANDOM_STATE)
        idx = rng.choice(n_train, _TABPFN_MAX_TRAIN, replace=False)
        X_tr = X_train.iloc[idx]
        y_tr = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
        print(f"  Subsampled training set: {n_train} → {_TABPFN_MAX_TRAIN}")
    else:
        X_tr, y_tr = X_train, y_train

    # Limit features
    feat_cols = list(X_tr.columns)
    if len(feat_cols) > _TABPFN_MAX_FEAT:
        feat_cols = feat_cols[:_TABPFN_MAX_FEAT]
        print(f"  Truncated features to {_TABPFN_MAX_FEAT}")
        X_tr = X_tr[feat_cols]
        X_te = X_test[feat_cols]
    else:
        X_te = X_test

    t0 = time.time()
    clf = _make_tabpfn_classifier(X_tr.values, np.asarray(y_tr), device="cpu")
    elapsed = time.time() - t0

    y_pred = clf.predict(X_te.values)
    y_proba_all = clf.predict_proba(X_te.values)
    y_proba = y_proba_all[:, 1] if y_proba_all.shape[1] == 2 else y_proba_all[:, -1]

    y_true = np.asarray(y_test)
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  TabPFN Binary Results ({elapsed:.1f}s):")
    print(f"    ROC-AUC    : {auc:.4f}")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    Precision  : {prec:.4f}")
    print(f"    Recall     : {rec:.4f}")
    print(f"    F1         : {f1:.4f}")

    # Build a thin sklearn-like wrapper that keeps the feature subset
    model = _TabPFNBinaryWrapper(clf, feat_cols)

    result = {
        "roc_auc": auc, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "best_model": "TabPFN", "elapsed_s": elapsed,
        "model": model,
    }
    gc.collect()
    return result


# ══════════════════════════════════════════════════════════════
# Section B -- Multiclass Classification (F1-weighted)
# ══════════════════════════════════════════════════════════════

def automl_multiclass(data_prep):
    """Run AutoGluon on the same multiclass split from prepare_multiclass().

    Parameters
    ----------
    data_prep : dict  returned by 02_feature_engineering.prepare_multiclass()
        Must contain X_train, X_test, y_train, y_test.

    Returns
    -------
    dict  with keys: f1_weighted, f1_macro, accuracy, roc_auc_ovr,
          leaderboard, elapsed_s, best_model.
    """
    if not _AUTOGLUON:
        print("  AutoGluon not installed -- skipping multiclass benchmark")
        return None

    print(f"\n{'='*70}")
    print("  AUTOML MULTICLASS CLASSIFICATION")
    print(f"{'='*70}")

    train_df, test_df = _build_ag_frames(
        data_prep["X_train"], data_prep["y_train"],
        data_prep["X_test"], data_prep["y_test"],
    )

    save_path = os.path.join(config.AUTOML_SAVE_DIR, "multiclass")
    _cleanup(save_path)

    t0 = time.time()
    predictor = TabularPredictor(
        label=_LABEL, eval_metric="f1_weighted",
        path=save_path, verbosity=1,
    ).fit(
        train_df,
        time_limit=config.AUTOML_TIME_LIMIT_MULTICLASS,
        presets=config.AUTOML_PRESETS,
    )
    elapsed = time.time() - t0

    y_true = test_df[_LABEL]
    y_pred = predictor.predict(test_df)

    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # ROC-AUC OVR (may fail if classes differ between train/test)
    try:
        y_proba = predictor.predict_proba(test_df)
        auc_ovr = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc_ovr = np.nan

    lb = predictor.leaderboard(test_df, silent=True)
    best = getattr(predictor, "model_best", None) or lb.iloc[0]["model"]

    print(f"\n  AutoGluon Multiclass Results ({elapsed:.0f}s):")
    print(f"    Best model  : {best}")
    print(f"    F1-weighted : {f1w:.4f}")
    print(f"    F1-macro    : {f1m:.4f}")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    ROC-AUC OVR : {auc_ovr:.4f}")
    print(f"\n  Leaderboard (top 5):")
    print(lb.head(5).to_string(index=False))

    # Determine class labels for the wrapper (string labels AutoGluon trained on)
    ag_classes = list(predictor.class_labels)

    result = {
        "f1_weighted": f1w, "f1_macro": f1m, "accuracy": acc,
        "roc_auc_ovr": auc_ovr, "leaderboard": lb,
        "best_model": best, "elapsed_s": elapsed,
        "model": _AGMulticlassWrapper(predictor, ag_classes),
        "_save_path": save_path,
    }

    gc.collect()
    return result


# ── TabPFN Multiclass ────────────────────────────────────────

def tabpfn_multiclass(data_prep):
    """Run TabPFN on the same multiclass split from prepare_multiclass().

    Parameters
    ----------
    data_prep : dict  returned by 02_feature_engineering.prepare_multiclass()

    Returns
    -------
    dict  with keys: f1_weighted, f1_macro, accuracy, roc_auc_ovr,
          elapsed_s, best_model, model.
    """
    if not _TABPFN:
        print("  TabPFN not installed -- skipping multiclass TabPFN benchmark")
        return None

    print(f"\n{'='*70}")
    print("  TABPFN MULTICLASS CLASSIFICATION")
    print(f"{'='*70}")

    X_train = data_prep["X_train"]
    y_train = data_prep["y_train"]
    X_test = data_prep["X_test"]
    y_test = data_prep["y_test"]

    # Subsample training set if needed
    n_train = len(X_train)
    if n_train > _TABPFN_MAX_TRAIN:
        rng = np.random.RandomState(config.RANDOM_STATE)
        idx = rng.choice(n_train, _TABPFN_MAX_TRAIN, replace=False)
        X_tr = X_train.iloc[idx]
        y_tr = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
        print(f"  Subsampled training set: {n_train} → {_TABPFN_MAX_TRAIN}")
    else:
        X_tr, y_tr = X_train, y_train

    # Limit features
    feat_cols = list(X_tr.columns)
    if len(feat_cols) > _TABPFN_MAX_FEAT:
        feat_cols = feat_cols[:_TABPFN_MAX_FEAT]
        print(f"  Truncated features to {_TABPFN_MAX_FEAT}")
        X_tr = X_tr[feat_cols]
        X_te = X_test[feat_cols]
    else:
        X_te = X_test

    y_tr_arr = np.asarray(y_tr)
    y_true = np.asarray(y_test)
    classes = np.unique(y_tr_arr)

    t0 = time.time()
    clf = _make_tabpfn_classifier(X_tr.values, y_tr_arr, device="cpu")
    elapsed = time.time() - t0

    y_pred = clf.predict(X_te.values)
    y_proba = clf.predict_proba(X_te.values)

    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    try:
        auc_ovr = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc_ovr = np.nan

    print(f"\n  TabPFN Multiclass Results ({elapsed:.1f}s):")
    print(f"    F1-weighted : {f1w:.4f}")
    print(f"    F1-macro    : {f1m:.4f}")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    ROC-AUC OVR : {auc_ovr:.4f}")

    model = _TabPFNMulticlassWrapper(clf, feat_cols, classes)

    result = {
        "f1_weighted": f1w, "f1_macro": f1m, "accuracy": acc,
        "roc_auc_ovr": auc_ovr,
        "best_model": "TabPFN", "elapsed_s": elapsed,
        "model": model,
    }
    gc.collect()
    return result


# ══════════════════════════════════════════════════════════════
# Section C -- Temporal Regression (MAE in days)
# ══════════════════════════════════════════════════════════════

def automl_temporal(data_prep):
    """Run AutoGluon regression on the same temporal split from prepare_temporal().

    AutoGluon trains on log1p(duration); predictions are expm1'd back to
    days so MAE/RMSE are directly comparable to module-05 results.

    Parameters
    ----------
    data_prep : dict  returned by 02_feature_engineering.prepare_temporal()
        Must contain X_train, X_test, y_train_log, y_test_log, y_test_raw.

    Returns
    -------
    dict  with keys: mae_days, rmse_days, r2, mae_log, leaderboard,
          elapsed_s, best_model.
    """
    if not _AUTOGLUON:
        print("  AutoGluon not installed -- skipping temporal benchmark")
        return None

    print(f"\n{'='*70}")
    print("  AUTOML TEMPORAL REGRESSION")
    print(f"{'='*70}")

    # AutoGluon trains on log1p target (same as module 05)
    train_df, test_df = _build_ag_frames(
        data_prep["X_train"], data_prep["y_train_log"],
        data_prep["X_test"], data_prep["y_test_log"],
    )

    save_path = os.path.join(config.AUTOML_SAVE_DIR, "temporal")
    _cleanup(save_path)

    t0 = time.time()
    predictor = TabularPredictor(
        label=_LABEL, problem_type="regression",
        eval_metric="mean_absolute_error",
        path=save_path, verbosity=1,
    ).fit(
        train_df,
        time_limit=config.AUTOML_TIME_LIMIT_TEMPORAL,
        presets=config.AUTOML_PRESETS,
    )
    elapsed = time.time() - t0

    # Predict on log scale
    y_pred_log = predictor.predict(test_df).values  # Series → numpy
    y_true_log = np.asarray(data_prep["y_test_log"])

    mae_log = mean_absolute_error(y_true_log, y_pred_log)

    # Convert back to days for comparison with module 05
    y_pred_days = np.expm1(np.clip(y_pred_log, 0, 15))  # clip extreme log preds
    y_true_days = data_prep["y_test_raw"]

    mae_days = mean_absolute_error(y_true_days, y_pred_days)
    rmse_days = np.sqrt(mean_squared_error(y_true_days, y_pred_days))
    r2 = r2_score(y_true_days, y_pred_days)

    lb = predictor.leaderboard(test_df, silent=True)
    best = getattr(predictor, "model_best", None) or lb.iloc[0]["model"]

    print(f"\n  AutoGluon Temporal Results ({elapsed:.0f}s):")
    print(f"    Best model : {best}")
    print(f"    MAE (days) : {mae_days:.1f}")
    print(f"    RMSE (days): {rmse_days:.1f}")
    print(f"    R2         : {r2:.4f}")
    print(f"    MAE (log)  : {mae_log:.4f}")
    print(f"\n  Leaderboard (top 5):")
    print(lb.head(5).to_string(index=False))

    # Build a model_dict compatible with 08's _predict_best_temporal()
    temporal_model_dict = {
        "best_model": "AutoGluon",
        "best_overall": "AutoGluon",
        "results": {
            "AutoGluon": {
                "model": _AGRegressionWrapper(predictor),
                "mae": mae_days,
            },
        },
        "encoding_map": data_prep.get("encoding_map", {}),
        "global_mean": data_prep.get("global_mean", 0),
        "cat_cols": data_prep.get("categorical_cols_clean", []),
        "num_cols": data_prep.get("num_cols", []),
        "top_feats": data_prep.get("feature_cols", []),
        "freq_map": data_prep.get("freq_map", {}),
        "interaction_pairs": data_prep.get("interaction_pairs", []),
        "group_maps": data_prep.get("group_maps", {}),
        "overall_target_median": data_prep.get("overall_median", 7.0),
        "log_transformed": True,
    }

    result = {
        "mae_days": mae_days, "rmse_days": rmse_days, "r2": r2,
        "mae_log": mae_log, "leaderboard": lb,
        "best_model": best, "elapsed_s": elapsed,
        "model": _AGRegressionWrapper(predictor),
        "temporal_model_dict": temporal_model_dict,
        "_save_path": save_path,
    }

    gc.collect()
    return result


# ── TabPFN Temporal ──────────────────────────────────────────

def tabpfn_temporal(data_prep):
    """Run TabPFN regressor on the same temporal split from prepare_temporal().

    Trains on log1p(duration); predictions are expm1'd back to days
    so MAE/RMSE are directly comparable to module-05 / AutoGluon results.

    Parameters
    ----------
    data_prep : dict  returned by 02_feature_engineering.prepare_temporal()

    Returns
    -------
    dict  with keys: mae_days, rmse_days, r2, mae_log,
          elapsed_s, best_model, model, temporal_model_dict.
    """
    if not _TABPFN:
        print("  TabPFN not installed -- skipping temporal TabPFN benchmark")
        return None

    print(f"\n{'='*70}")
    print("  TABPFN TEMPORAL REGRESSION")
    print(f"{'='*70}")

    X_train = data_prep["X_train"]
    y_train_log = data_prep["y_train_log"]
    X_test = data_prep["X_test"]

    # Subsample training set if needed
    n_train = len(X_train)
    if n_train > _TABPFN_MAX_TRAIN:
        rng = np.random.RandomState(config.RANDOM_STATE)
        idx = rng.choice(n_train, _TABPFN_MAX_TRAIN, replace=False)
        X_tr = X_train.iloc[idx]
        y_tr = y_train_log.iloc[idx] if hasattr(y_train_log, "iloc") else y_train_log[idx]
        print(f"  Subsampled training set: {n_train} → {_TABPFN_MAX_TRAIN}")
    else:
        X_tr, y_tr = X_train, y_train_log

    # Limit features
    feat_cols = list(X_tr.columns)
    if len(feat_cols) > _TABPFN_MAX_FEAT:
        feat_cols = feat_cols[:_TABPFN_MAX_FEAT]
        print(f"  Truncated features to {_TABPFN_MAX_FEAT}")
        X_tr = X_tr[feat_cols]
        X_te = X_test[feat_cols]
    else:
        X_te = X_test

    t0 = time.time()
    reg = _make_tabpfn_regressor(X_tr.values, np.asarray(y_tr), device="cpu")
    elapsed = time.time() - t0

    # Predict on log scale
    y_pred_log = reg.predict(X_te.values)
    y_true_log = data_prep["y_test_log"]
    mae_log = mean_absolute_error(y_true_log, y_pred_log)

    # Convert back to days
    y_pred_days = np.expm1(np.clip(y_pred_log, 0, 15))
    y_true_days = data_prep["y_test_raw"]

    mae_days = mean_absolute_error(y_true_days, y_pred_days)
    rmse_days = np.sqrt(mean_squared_error(y_true_days, y_pred_days))
    r2 = r2_score(y_true_days, y_pred_days)

    print(f"\n  TabPFN Temporal Results ({elapsed:.1f}s):")
    print(f"    MAE (days) : {mae_days:.1f}")
    print(f"    RMSE (days): {rmse_days:.1f}")
    print(f"    R2         : {r2:.4f}")
    print(f"    MAE (log)  : {mae_log:.4f}")

    model = _TabPFNRegressionWrapper(reg, feat_cols)

    # Build a model_dict compatible with 08's _predict_best_temporal()
    temporal_model_dict = {
        "best_model": "TabPFN",
        "best_overall": "TabPFN",
        "results": {
            "TabPFN": {
                "model": model,
                "mae": mae_days,
            },
        },
        "encoding_map": data_prep.get("encoding_map", {}),
        "global_mean": data_prep.get("global_mean", 0),
        "cat_cols": data_prep.get("categorical_cols_clean", []),
        "num_cols": data_prep.get("num_cols", []),
        "top_feats": data_prep.get("feature_cols", []),
        "freq_map": data_prep.get("freq_map", {}),
        "interaction_pairs": data_prep.get("interaction_pairs", []),
        "group_maps": data_prep.get("group_maps", {}),
        "overall_target_median": data_prep.get("overall_median", 7.0),
        "log_transformed": True,
    }

    result = {
        "mae_days": mae_days, "rmse_days": rmse_days, "r2": r2,
        "mae_log": mae_log,
        "best_model": "TabPFN", "elapsed_s": elapsed,
        "model": model,
        "temporal_model_dict": temporal_model_dict,
    }
    gc.collect()
    return result


def cleanup_all():
    """Remove all AutoGluon model artifacts (call after forecast is done)."""
    for sub in ("binary", "multiclass", "temporal"):
        _cleanup(os.path.join(config.AUTOML_SAVE_DIR, sub))


# ══════════════════════════════════════════════════════════════
# Section D -- Comparison Utilities
# ══════════════════════════════════════════════════════════════

def compare_binary(manual_results, automl_results, tabpfn_results=None):
    """Side-by-side binary comparison: Manual vs AutoGluon vs TabPFN."""
    if automl_results is None and tabpfn_results is None:
        print("  No AutoML/TabPFN binary results -- skipping comparison")
        return None

    manual_auc = manual_results["best_metrics"].get("ROC-AUC", np.nan)
    manual_model = manual_results.get("best_model", "?")

    rows = [{"Pipeline": "Manual", "Model": manual_model, "ROC-AUC": manual_auc}]

    if automl_results is not None:
        rows.append({"Pipeline": "AutoGluon", "Model": automl_results["best_model"],
                     "ROC-AUC": automl_results["roc_auc"]})
    if tabpfn_results is not None:
        rows.append({"Pipeline": "TabPFN", "Model": "TabPFN",
                     "ROC-AUC": tabpfn_results["roc_auc"]})

    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print(f"\n  Binary Classification -- Comparison:")
    print(df.to_string(index=False))

    winner = df.iloc[0]["Pipeline"]
    print(f"\n  Winner: {winner}  (ROC-AUC = {df.iloc[0]['ROC-AUC']:.4f})")
    return df


def compare_multiclass(manual_results, automl_results, tabpfn_results=None):
    """Side-by-side multiclass comparison: Manual vs AutoGluon vs TabPFN."""
    if automl_results is None and tabpfn_results is None:
        print("  No AutoML/TabPFN multiclass results -- skipping comparison")
        return None

    if manual_results["comparison"].empty:
        print("  No manual multiclass results — skipping comparison")
        return None
    best_manual = manual_results["comparison"].iloc[0]
    manual_f1w = best_manual.get("F1_weighted", np.nan)
    manual_model = best_manual.get("Model", "?")

    rows = [{"Pipeline": "Manual", "Model": manual_model, "F1_weighted": manual_f1w}]

    if automl_results is not None:
        rows.append({"Pipeline": "AutoGluon", "Model": automl_results["best_model"],
                     "F1_weighted": automl_results["f1_weighted"]})
    if tabpfn_results is not None:
        rows.append({"Pipeline": "TabPFN", "Model": "TabPFN",
                     "F1_weighted": tabpfn_results["f1_weighted"]})

    df = pd.DataFrame(rows).sort_values("F1_weighted", ascending=False).reset_index(drop=True)
    print(f"\n  Multiclass Classification -- Comparison:")
    print(df.to_string(index=False))

    winner = df.iloc[0]["Pipeline"]
    print(f"\n  Winner: {winner}  (F1_weighted = {df.iloc[0]['F1_weighted']:.4f})")
    return df


def compare_temporal(manual_results, automl_results, tabpfn_results=None):
    """Side-by-side temporal comparison: Manual vs AutoGluon vs TabPFN."""
    if automl_results is None and tabpfn_results is None:
        print("  No AutoML/TabPFN temporal results -- skipping comparison")
        return None

    dur = manual_results.get("duration") or {}
    best_name = dur.get("best_overall", "?")
    manual_mae = dur.get("results", {}).get(best_name, {}).get("mae", np.nan)

    rows = [{"Pipeline": "Manual", "Model": best_name, "MAE_days": manual_mae}]

    if automl_results is not None:
        rows.append({"Pipeline": "AutoGluon", "Model": automl_results["best_model"],
                     "MAE_days": automl_results["mae_days"]})
    if tabpfn_results is not None:
        rows.append({"Pipeline": "TabPFN", "Model": "TabPFN",
                     "MAE_days": tabpfn_results["mae_days"]})

    df = pd.DataFrame(rows).sort_values("MAE_days").reset_index(drop=True)
    print(f"\n  Temporal Regression -- Comparison:")
    print(df.to_string(index=False))

    winner = df.iloc[0]["Pipeline"]
    print(f"\n  Winner: {winner}  (MAE = {df.iloc[0]['MAE_days']:.1f} days)")
    return df


# ══════════════════════════════════════════════════════════════
# Section E -- Orchestrator
# ══════════════════════════════════════════════════════════════

def run_automl_benchmark(data_prep_binary, data_prep_multiclass,
                         data_prep_temporal,
                         binary_results=None, multi_results=None,
                         temporal_results=None):
    """Run AutoGluon + TabPFN on all three tasks and compare against manual pipeline.

    Parameters
    ----------
    data_prep_binary     : dict from prepare_binary()
    data_prep_multiclass : dict from prepare_multiclass()
    data_prep_temporal   : dict from prepare_temporal()
    binary_results       : dict from final_model_comparison() (module 03)
    multi_results        : dict from multiclass_retrofit_types() (module 04)
    temporal_results     : dict from temporal_prediction() (module 06)

    Returns
    -------
    dict  with sub-dicts for AutoGluon ('binary', 'multiclass', 'temporal'),
          TabPFN ('tabpfn_binary', 'tabpfn_multiclass', 'tabpfn_temporal'),
          and 3-way comparison DataFrames (*_comparison).
    """
    print(f"\n{'='*70}")
    print("  AUTOML BENCHMARK  (AutoGluon + TabPFN)")
    print(f"{'='*70}")

    t0 = time.time()
    out = {}

    # ── AutoGluon ────────────────────────────────────────────
    ag_bin = ag_mc = ag_tmp = None
    if _AUTOGLUON:
        try:
            ag_bin = automl_binary(data_prep_binary)
        except Exception as exc:
            print(f"  ✗ AutoGluon binary failed: {exc}")
        try:
            ag_mc = automl_multiclass(data_prep_multiclass)
        except Exception as exc:
            print(f"  ✗ AutoGluon multiclass failed: {exc}")
        try:
            ag_tmp = automl_temporal(data_prep_temporal)
        except Exception as exc:
            print(f"  ✗ AutoGluon temporal failed: {exc}")

    out["binary"]     = ag_bin
    out["multiclass"] = ag_mc
    out["temporal"]   = ag_tmp

    # ── TabPFN ───────────────────────────────────────────────
    tp_bin = tp_mc = tp_tmp = None
    if _TABPFN:
        try:
            tp_bin = tabpfn_binary(data_prep_binary)
        except Exception as exc:
            print(f"  ✗ TabPFN binary failed: {exc}")
        try:
            tp_mc = tabpfn_multiclass(data_prep_multiclass)
        except Exception as exc:
            print(f"  ✗ TabPFN multiclass failed: {exc}")
        try:
            tp_tmp = tabpfn_temporal(data_prep_temporal)
        except Exception as exc:
            print(f"  ✗ TabPFN temporal failed: {exc}")

    out["tabpfn_binary"]     = tp_bin
    out["tabpfn_multiclass"] = tp_mc
    out["tabpfn_temporal"]   = tp_tmp

    # ── 3-way comparisons ───────────────────────────────────
    try:
        if binary_results is not None and (ag_bin is not None or tp_bin is not None):
            out["binary_comparison"] = compare_binary(binary_results, ag_bin, tp_bin)
    except Exception as exc:
        print(f"  ✗ Binary comparison failed: {exc}")

    try:
        if multi_results is not None and (ag_mc is not None or tp_mc is not None):
            out["multiclass_comparison"] = compare_multiclass(multi_results, ag_mc, tp_mc)
    except Exception as exc:
        print(f"  ✗ Multiclass comparison failed: {exc}")

    try:
        if temporal_results is not None and (ag_tmp is not None or tp_tmp is not None):
            out["temporal_comparison"] = compare_temporal(temporal_results, ag_tmp, tp_tmp)
    except Exception as exc:
        print(f"  ✗ Temporal comparison failed: {exc}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  AutoML benchmark completed in {elapsed/60:.1f} min")
    print(f"{'='*70}")

    return out