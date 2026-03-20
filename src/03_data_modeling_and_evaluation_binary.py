# ============================================================
# 03_data_modeling_and_evaluation_binary.py – Binary classification pipelines
#
# Structure
# ---------
#   Block A — Models:        Base model factory, evaluation helpers,
#                             baseline training (Stacking + Voting)
#   Block B — Optimization:  Optuna TPE tuning (+ joint SMOTE),
#                             standalone SMOTE sweep, feature selection
#   Block C — Foundation Models: LLM-Embed (Bedrock Titan -> LogReg),
#                             LLM-Prompted (Bedrock Converse)
#   Block D — Evaluation:    Cross-experiment comparison utilities
# ============================================================
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from scipy import stats as sp_stats
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

from sklearn.base import BaseEstimator, ClassifierMixin

from . import config

import importlib as _il
_fe = _il.import_module(".02_feature_engineering", __package__)
_sanitize_features = _fe._sanitize_features
shap_feature_prefilter = _fe.shap_feature_prefilter

# FM / LLM availability (graceful degradation if boto3 is missing)
try:
    import boto3 as _boto3_mod
    _BEDROCK = True
except Exception:
    _BEDROCK = False


# ══════════════════════════════════════════════════════════════
# BLOCK A — MODELS (baseline definitions and training)
# ══════════════════════════════════════════════════════════════

def _get_base_models(y_train):
    """Return the model dict used across all experiments + Dummy floor."""
    pos_weight = min(float((y_train == 0).sum() / max(y_train.sum(), 1)), 20.0)
    models = {
        "Dummy (majority)": DummyClassifier(
            strategy="most_frequent", random_state=config.RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=config.RANDOM_STATE, class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS, max_depth=20,
            random_state=config.RANDOM_STATE,
            class_weight="balanced", n_jobs=2,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=config.N_ESTIMATORS, max_depth=5,
            learning_rate=0.05, class_weight="balanced",
            l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=config.N_ESTIMATORS, max_depth=20,
            random_state=config.RANDOM_STATE,
            class_weight="balanced", n_jobs=2,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE,
        ),  # no n_jobs param on AdaBoost
        "XGBoost": xgb.XGBClassifier(
            n_estimators=config.N_ESTIMATORS, learning_rate=0.05,
            max_depth=6, eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            scale_pos_weight=pos_weight, tree_method="hist", n_jobs=2,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=config.N_ESTIMATORS, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=config.RANDOM_STATE,
            is_unbalance=True, verbose=-1, n_jobs=2,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=config.N_ESTIMATORS, random_seed=config.RANDOM_STATE,
            auto_class_weights="Balanced", verbose=0, thread_count=2,
            train_dir=config.CATBOOST_TRAIN_DIR,
        ),
        "SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, validation_fraction=0.1,
                random_state=config.RANDOM_STATE)),
        ]),
    }
    # Remove slow / low-value models globally (same set skipped in Optuna)
    for skip in config.OPTUNA_SKIP_BINARY:
        models.pop(skip, None)
    return models


# ---- B.1 Optuna HP builder ----

def _bin_build_model(trial, name, y_train):
    """Suggest hyperparameters for *name* via Optuna trial (binary)."""
    RS = config.RANDOM_STATE
    pos_weight = float((y_train == 0).sum() / max(y_train.sum(), 1))

    if name == "Logistic Regression":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        return LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RS)

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced", random_state=RS, n_jobs=2)

    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced", random_state=RS, n_jobs=2)

    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 100, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            l2_regularization=trial.suggest_float("l2_reg", 1e-4, 10.0, log=True),
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.1,
            random_state=RS)

    if name == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 1.0, log=True),
            random_state=RS)

    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            scale_pos_weight=pos_weight,
            eval_metric="logloss", tree_method="hist",
            random_state=RS, n_jobs=2)

    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            is_unbalance=True,
            random_state=RS, verbose=-1, n_jobs=2)

    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            bootstrap_type="Bernoulli",
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            auto_class_weights="Balanced",
            random_seed=RS, verbose=0, thread_count=2,
            train_dir=config.CATBOOST_TRAIN_DIR)

    if name == "SVC":
        C = trial.suggest_float("svc_C", 0.01, 100.0, log=True)
        gamma = trial.suggest_categorical("svc_gamma", ["scale", "auto"])
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=C, gamma=gamma, kernel="rbf",
                        probability=True, class_weight="balanced",
                        random_state=RS)),
        ])

    if name == "MLP":
        n1 = trial.suggest_categorical("layer1", [64, 128, 256])
        n2 = trial.suggest_categorical("layer2", [32, 64, 128])
        lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(n1, n2), learning_rate_init=lr,
                alpha=alpha, max_iter=500, early_stopping=True,
                validation_fraction=0.1, random_state=RS)),
        ])

    raise ValueError(f"Unknown model: {name}")


def _evaluate(model, X_test, y_test):
    """Predict + compute all metrics incl. optimal threshold."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Degenerate-prediction safeguard
    p_std = float(np.std(y_proba))
    _degenerate = p_std < 1e-6
    if _degenerate:
        print(f"    ⚠ DEGENERATE: predict_proba std={p_std:.2e} "
              f"(all ≈ {y_proba.mean():.4f}) — model did not learn")
    elif float(y_proba.min()) > 0.5:
        print(f"    ⚠ ALL-POSITIVE: proba range [{y_proba.min():.3f}, {y_proba.max():.3f}] "
              f"— model biased toward class 1")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Optimal threshold: maximise F1 via precision-recall curve
    prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_proba)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
    best_idx = int(np.argmax(f1_c))
    opt_thr = float(thr_c[best_idx]) if best_idx < len(thr_c) and f1_c[best_idx] > 0 else 0.5
    y_pred_opt = (y_proba >= opt_thr).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_test, y_pred_opt).ravel()

    # Calibration table: binned predicted probability vs actual rate
    _y_arr = np.asarray(y_test)
    _bin_edges = np.linspace(0, 1, 11)  # 10 bins: [0,.1), [.1,.2), ...
    _cal_rows = []
    for i in range(len(_bin_edges) - 1):
        lo, hi = _bin_edges[i], _bin_edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi) if i < 9 else (y_proba >= lo) & (y_proba <= 1.0)
        n = int(mask.sum())
        if n > 0:
            actual_rate = float(_y_arr[mask].mean())
            mean_pred = float(y_proba[mask].mean())
            _bin_label = f"[{lo:.1f},{hi:.1f})" if i < 9 else f"[{lo:.1f},{hi:.1f}]"
            _cal_rows.append({"bin": _bin_label, "n": n,
                              "mean_pred": mean_pred, "actual_rate": actual_rate,
                              "gap": abs(mean_pred - actual_rate)})
    del _y_arr, _bin_edges
    _calibration = _cal_rows if _cal_rows else None

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_matrix_opt": {"tn": int(tn_o), "fp": int(fp_o),
                                  "fn": int(fn_o), "tp": int(tp_o)},
        "y_pred": y_pred,
        "y_pred_proba": y_proba,
        "y_test_eval": np.asarray(y_test),   # actual y used (may be subsampled)
        "y_pred_opt": y_pred_opt,
        "optimal_threshold": opt_thr,
        "f1_at_optimal": float(f1_score(y_test, y_pred_opt, zero_division=0)),
        "recall_at_optimal": float(recall_score(y_test, y_pred_opt, zero_division=0)),
        "precision_at_optimal": float(precision_score(y_test, y_pred_opt, zero_division=0)),
        "degenerate": _degenerate,
        "n_test_samples": len(y_test),
        "calibration_table": _calibration,
    }


def _print_metrics(name, m):
    cv_str = ""
    if "cv_auc_mean" in m:
        cv_str = f"  CV-AUC:{m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.4f}"
    pr_str = f" PR-AUC:{m['pr_auc']:.4f}" if 'pr_auc' in m else ""
    print(
        f"  {name:<25} Acc:{m['accuracy']:.4f} Prec:{m['precision']:.4f} "
        f"Rec:{m['recall']:.4f} F1:{m['f1_score']:.4f} AUC:{m['roc_auc']:.4f}{pr_str}{cv_str}"
    )
    if "optimal_threshold" in m:
        print(
            f"  {'':25} ↳ Opt thr={m['optimal_threshold']:.3f}: "
            f"F1={m['f1_at_optimal']:.3f} Rec={m['recall_at_optimal']:.3f} "
            f"Prec={m['precision_at_optimal']:.3f}"
        )


def _comparison_df(results):
    d = {
        "Model": list(results.keys()),
        "Accuracy": [results[m]["accuracy"] for m in results],
        "Precision": [results[m]["precision"] for m in results],
        "Recall": [results[m]["recall"] for m in results],
        "F1-Score": [results[m]["f1_score"] for m in results],
        "ROC-AUC": [results[m]["roc_auc"] for m in results],
        "PR-AUC": [results[m].get("pr_auc", np.nan) for m in results],
    }
    # Include CV columns when available
    if any("cv_auc_mean" in results[m] for m in results):
        d["CV-AUC-mean"] = [results[m].get("cv_auc_mean", np.nan) for m in results]
        d["CV-AUC-std"]  = [results[m].get("cv_auc_std", np.nan) for m in results]
    cdf = pd.DataFrame(d).sort_values("ROC-AUC", ascending=False)
    # Exclude degenerate + small-sample models from the top position
    _min_n = getattr(config, "MIN_TEST_SAMPLES_FOR_BEST", 500)
    _elig = [m for m in results
             if not results[m].get("degenerate", False)
             and results[m].get("n_test_samples", _min_n) >= _min_n]
    if _elig:
        _elig_df = cdf[cdf["Model"].isin(_elig)]
        _rest = cdf[~cdf["Model"].isin(_elig)]
        cdf = pd.concat([_elig_df, _rest]).reset_index(drop=True)
    return cdf


# ---- A.3 Baseline ----

def binary_classification(data_prep):
    """Train baseline models + Stacking & Voting ensembles with CV."""
    warnings.filterwarnings("ignore")
    N_CV = config.BASELINE_CV_FOLDS
    print("\n" + "=" * 80)
    print(f"BASELINE BINARY CLASSIFICATION ({N_CV}-fold CV + holdout, 10 models + 2 ensembles)")
    print("=" * 80 + "\n")

    X_tr, X_te = data_prep["X_train"], data_prep["X_test"]
    y_tr, y_te = data_prep["y_train"], data_prep["y_test"]
    X_tr, X_te = _sanitize_features(X_tr, X_te, label="Baseline")
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=config.RANDOM_STATE)

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    print(f"  Features: {X_tr.shape[1]}  |  Train: {len(y_tr):,} "
          f"(pos={n_pos:,}, neg={n_neg:,}, ratio=1:{n_neg // max(n_pos, 1)})")
    print(f"  Test:  {len(y_te):,} "
          f"(pos={int(y_te.sum()):,}, neg={int((y_te == 0).sum()):,})")
    print(f"  X dtype: {X_tr.dtypes.value_counts().to_dict()}\n")

    results = {}
    for name, model in _get_base_models(y_tr).items():
        try:
            print(f"  Training: {name}")
            # 3-fold CV on training set for stability estimate
            cv_scores = cross_val_score(
                model, X_tr, y_tr, cv=skf, scoring="roc_auc", n_jobs=2,
            )
            # Refit on full training set for holdout evaluation
            model.fit(X_tr, y_tr)
            results[name] = _evaluate(model, X_te, y_te)
            results[name]["cv_auc_mean"] = float(cv_scores.mean())
            results[name]["cv_auc_std"]  = float(cv_scores.std())
            _print_metrics(name, results[name])
        except Exception as exc:
            print(f"  ✗ {name} failed: {exc}")
        gc.collect()

    # ── Stacking (top-3 by AUC → LightGBM meta) ─────────────
    # Exclude: CatBoost (sklearn compat), LLMs (API-based, subsampled),
    # Dummy (trivial), ensembles themselves.
    _ENSEMBLE_SKIP = {"CatBoost", "LLM-Embed LogReg", "LLM-Prompted",
                      "Dummy (majority)", "Stacking", "Voting"}
    ranked = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    top3 = [r[0] for r in ranked if r[0] not in _ENSEMBLE_SKIP][:3]
    print(f"\n  Stacking base (top-3 AUC, excl. CatBoost): {top3}")
    try:
        stack = StackingClassifier(
            estimators=[(n, results[n]["model"]) for n in top3],
            final_estimator=LGBMClassifier(
                n_estimators=100, learning_rate=0.08, num_leaves=31,
                verbose=-1, random_state=config.RANDOM_STATE, n_jobs=2,
            ),
            cv=config.STACKING_CV_FOLDS, passthrough=False, n_jobs=1,
        )
        stack.fit(X_tr, y_tr)
        results["Stacking"] = _evaluate(stack, X_te, y_te)
        _print_metrics("Stacking", results["Stacking"])
    except Exception as exc:
        print(f"  ✗ Stacking failed: {exc}")
    gc.collect()

    # ── Soft Voting (top-3 by AUC) ───────────────────────────
    print(f"\n  Voting base (top-3 AUC, excl. CatBoost): {top3}")
    try:
        voting = VotingClassifier(
            estimators=[(n, results[n]["model"]) for n in top3],
            voting="soft", n_jobs=2,
        )
        voting.fit(X_tr, y_tr)
        results["Voting"] = _evaluate(voting, X_te, y_te)
        _print_metrics("Voting", results["Voting"])
    except Exception as exc:
        print(f"  ✗ Voting failed: {exc}")
    gc.collect()

    # ── Foundation-model classifiers (Block C) ───────────────
    _run_llm_binary(X_tr, y_tr, X_te, y_te, results,
                    X_tr_raw=data_prep.get("X_train_raw"),
                    X_te_raw=data_prep.get("X_test_raw"))

    cdf = _comparison_df(results)
    _best = cdf.iloc[0]["Model"] if not cdf.empty else None
    if _best:
        print(f"\nBest: {_best} (AUC {cdf.iloc[0]['ROC-AUC']:.4f})")
    gc.collect()
    return {
        "results": results, "comparison": cdf, "best_model": _best,
        **{k: data_prep[k] for k in ("X_train", "X_test", "y_train", "y_test",
                                       "feature_cols", "target_encoding_map", "global_mean")},
    }


# ══════════════════════════════════════════════════════════════
# BLOCK B — OPTIMIZATION (Optuna, SMOTE, feature selection)
# ══════════════════════════════════════════════════════════════

# ---- B.2 Hyperparameter tuning ----

def hyperparameter_tuning(data_prep):
    """Optuna TPE Bayesian optimisation for all 10 models (binary).

    SMOTE ratio is tuned jointly with hyperparameters: Optuna picks the
    ratio per trial and SMOTE is applied **inside** each CV fold via
    imblearn.Pipeline to prevent synthetic-sample leakage.
    Falls back to RandomizedSearchCV when Optuna is not installed.
    """
    warnings.filterwarnings("ignore")
    from imblearn.over_sampling import SMOTE as _SMOTE
    from imblearn.pipeline import Pipeline as _ImbPipeline

    X_tr, X_te = data_prep["X_train"], data_prep["X_test"]
    y_tr, y_te = data_prep["y_train"], data_prep["y_test"]

    # ── Sanitise NaN / Inf in features (root cause of XGBoost NaN AUC) ──
    _nan_tr = int(np.isnan(X_tr.values).sum()) if hasattr(X_tr, 'values') else 0
    _inf_tr = int(np.isinf(X_tr.values).sum()) if hasattr(X_tr, 'values') else 0
    if _nan_tr or _inf_tr:
        print(f"  ⚠ Sanitising features: {_nan_tr} NaN, {_inf_tr} Inf values")
        X_tr = X_tr.fillna(0).replace([np.inf, -np.inf], 0)
        X_te = X_te.fillna(0).replace([np.inf, -np.inf], 0)

    # ── SMOTE config (applied inside CV folds, not pre-computed) ──
    _SMOTE_RATIOS = ["none", "0.3", "0.5", "0.6", "0.8", "1.0"]
    min_class_n = int(y_tr.value_counts().min())
    smote_k = min(config.SMOTE_K_NEIGHBORS, min_class_n - 1)
    can_smote = smote_k >= 1
    if not can_smote:
        print("  SMOTE: skipped (minority class too small)")
        _SMOTE_RATIOS = ["none"]

    # Model names to tune (excludes Dummy + config-skipped models)
    tune_names = [n for n in [
        "Logistic Regression", "Random Forest", "Extra Trees",
        "Gradient Boosting", "AdaBoost", "XGBoost", "LightGBM",
        "CatBoost", "SVC", "MLP",
    ] if n not in config.OPTUNA_SKIP_BINARY]

    if _OPTUNA:
        # ── Optuna TPE path ──────────────────────────────────
        n_trials = config.OPTUNA_N_TRIALS
        timeout = config.OPTUNA_TIMEOUT
        cv = StratifiedKFold(n_splits=config.OPTUNA_CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE)
        print("\n" + "=" * 80)
        print(f"HYPERPARAMETER TUNING — Optuna TPE ({n_trials} trials, "
              f"{config.OPTUNA_CV_FOLDS}-fold CV, {timeout}s timeout, in-fold SMOTE)")
        print("=" * 80 + "\n")

        results = {}
        for name in tune_names:
            print(f"  Optuna tuning: {name} ({n_trials} trials) …",
                  end=" ", flush=True)

            def objective(trial, _name=name):
                # Let Optuna pick the SMOTE ratio (or no SMOTE)
                smote_key = trial.suggest_categorical(
                    "smote_ratio", _SMOTE_RATIOS)
                model = _bin_build_model(trial, _name, y_tr)
                # Wrap in imblearn Pipeline so SMOTE runs inside each fold
                if smote_key != "none" and can_smote:
                    smote_step = _SMOTE(
                        sampling_strategy=float(smote_key),
                        k_neighbors=smote_k,
                        random_state=config.RANDOM_STATE,
                    )
                    estimator = _ImbPipeline([("smote", smote_step), ("clf", model)])
                else:
                    estimator = model
                try:
                    scores = cross_val_score(
                        estimator, X_tr, y_tr, cv=cv,
                        scoring="roc_auc", n_jobs=1,
                    )
                    val = float(scores.mean())
                    if np.isnan(val) or np.isinf(val):
                        raise optuna.TrialPruned()
                    return val
                except Exception as exc:
                    if "pruned" in str(type(exc)).lower():
                        raise
                    raise optuna.TrialPruned()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                           show_progress_bar=False)

            # Guard: fall back to hand-tuned defaults if all trials failed
            # or if the best AUC is barely above random chance (≤ 0.55)
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            best_val = study.best_value if completed else 0.0
            if not completed or best_val <= 0.55:
                reason = "ALL trials failed" if not completed else f"best AUC={best_val:.4f} ≤ 0.55 (degenerate)"
                print(f"{reason} — using hand-tuned defaults")
                fallback = _get_base_models(y_tr).get(name)
                if fallback is None:
                    print(f"    ✗ No fallback for {name}, skipping"); continue
                try:
                    fallback.fit(X_tr, y_tr)
                    m = _evaluate(fallback, X_te, y_te)
                    m["best_params"] = {"fallback": True}
                    m["cv_score"] = np.nan
                    results[name] = m
                    _print_metrics(name, m)
                except Exception as exc:
                    print(f"    ✗ {name} fallback failed: {exc}")
                del study; gc.collect()
                continue

            best = study.best_trial
            best_smote = best.params.get("smote_ratio", "none")
            print(f"AUC={best.value:.4f}  (trial {best.number}, smote={best_smote})")

            # Rebuild with best params, fit with SMOTE on full training set
            try:
                best_model = _bin_build_model(best, name, y_tr)
                if best_smote != "none" and can_smote:
                    sm_final = _SMOTE(sampling_strategy=float(best_smote),
                                      k_neighbors=smote_k,
                                      random_state=config.RANDOM_STATE)
                    _X_fit, _y_fit = sm_final.fit_resample(X_tr, y_tr)
                else:
                    _X_fit, _y_fit = X_tr, y_tr
                best_model.fit(_X_fit, _y_fit)
                m = _evaluate(best_model, X_te, y_te)
                m["best_params"] = best.params
                m["cv_score"] = best.value
                results[name] = m
                _print_metrics(name, m)
            except Exception as exc:
                print(f"    ✗ {name} Optuna refit failed: {exc}")
            del study; gc.collect()

    else:
        # ── Fallback: RandomizedSearchCV ─────────────────────
        cv = StratifiedKFold(n_splits=config.HP_CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE)
        print("\n" + "=" * 80)
        print(f"HYPERPARAMETER TUNING — RandomizedSearchCV fallback "
              f"({config.HP_N_ITER} iter, {config.HP_CV_FOLDS}-fold)")
        print("=" * 80 + "\n")

        param_grids = {
            "Logistic Regression": {"C": uniform(0.1, 2), "penalty": ["l2"], "solver": ["lbfgs"], "max_iter": [1000]},
            "Random Forest": {"n_estimators": [100, 200, 400], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10], "class_weight": ["balanced"]},
            "Gradient Boosting": {"max_iter": [100, 200, 400], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 8], "class_weight": ["balanced"]},
            "Extra Trees": {"n_estimators": [100, 200, 400], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10], "class_weight": ["balanced"]},
            "AdaBoost": {"n_estimators": [50, 100, 200, 400], "learning_rate": [0.01, 0.05, 0.1, 1.0]},
            "XGBoost": {"n_estimators": [100, 200, 400], "max_depth": [3, 5, 8], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0], "tree_method": ["hist"]},
            "LightGBM": {"n_estimators": [100, 200, 400], "max_depth": [3, 5, 8], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [15, 31, 63]},
            "CatBoost": {"iterations": [100, 200, 400], "depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1], "l2_leaf_reg": [0.5, 1.0, 3.0]},
            "SVC": {"svc__C": [0.01, 0.1, 1, 10], "svc__gamma": ["scale", "auto"], "svc__kernel": ["rbf"]},
            "MLP": {"mlp__hidden_layer_sizes": [(64,), (128, 64), (256, 128)], "mlp__learning_rate_init": [0.001, 0.005, 0.01], "mlp__alpha": [1e-5, 1e-4, 1e-3]},
        }

        results = {}
        for name, base in _get_base_models(y_tr).items():
            if name not in param_grids:
                print(f"  Skipping {name} (no tunable hyperparameters)")
                continue
            print(f"  Tuning: {name}…")
            try:
                rs = RandomizedSearchCV(
                    base, param_grids[name], n_iter=config.HP_N_ITER, cv=cv,
                    scoring="roc_auc", n_jobs=2, random_state=config.RANDOM_STATE,
                    verbose=0, return_train_score=False,
                )
                rs.fit(X_tr, y_tr)
                m = _evaluate(rs.best_estimator_, X_te, y_te)
                m["best_params"] = rs.best_params_
                m["cv_score"] = rs.best_score_
                results[name] = m
                _print_metrics(name, m)
                del rs; gc.collect()
            except Exception as exc:
                print(f"  ✗ {name} RandomizedSearchCV failed: {exc}")
                gc.collect()

    # ── Foundation-model classifiers (Block C) ───────────────
    _run_llm_binary(X_tr, y_tr, X_te, y_te, results,
                    X_tr_raw=data_prep.get("X_train_raw"),
                    X_te_raw=data_prep.get("X_test_raw"))

    cdf = _comparison_df(results)
    if cdf.empty:
        print("  ✗ No models succeeded in hyperparameter_tuning — returning empty")
        _best = None
    else:
        _best = cdf.iloc[0]["Model"]
        print(f"\nBest tuned: {_best} (AUC {cdf.iloc[0]['ROC-AUC']:.4f})")
    return {
        "results": results, "comparison": cdf, "best_model": _best,
        **{k: data_prep[k] for k in ("X_train", "X_test", "y_train", "y_test",
                                       "feature_cols", "target_encoding_map", "global_mean")},
    }


# ---- B.3 SMOTE ----

def smote_classification(data_prep):
    """Train models on SMOTE-balanced data with multiple ratios.

    SMOTE is applied **inside** each CV fold via imblearn.Pipeline to
    prevent synthetic-sample leakage across folds.  The best ratio per
    model is selected by CV AUC.  Final model is refit with SMOTE on
    the full training set and evaluated on the held-out test set.
    """
    warnings.filterwarnings("ignore")
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    N_CV = config.BASELINE_CV_FOLDS
    print("\n" + "=" * 80)
    print(f"SMOTE CLASSIFICATION ({N_CV}-fold CV + holdout, multi-ratio, in-fold SMOTE)")
    print("=" * 80 + "\n")

    X_tr, X_te = data_prep["X_train"], data_prep["X_test"]
    y_tr, y_te = data_prep["y_train"], data_prep["y_test"]
    X_tr, X_te = _sanitize_features(X_tr, X_te, label="SMOTE")

    _SMOTE_RATIOS = [0.3, 0.5, 0.6, 0.8, 1.0]
    min_class_n = int(y_tr.value_counts().min())
    smote_k = min(config.SMOTE_K_NEIGHBORS, min_class_n - 1)
    can_smote = smote_k >= 1
    if not can_smote:
        print("  SMOTE: skipped (minority class too small)")
        _SMOTE_RATIOS = []

    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=config.RANDOM_STATE)

    # SMOTE models don't use class_weight (data is already balanced)
    _raw_models = {
        "Dummy (majority)": DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE),
        "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=config.N_ESTIMATORS, max_depth=20, random_state=config.RANDOM_STATE, n_jobs=2),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=config.N_ESTIMATORS, max_depth=5,
            learning_rate=0.05, l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(n_estimators=config.N_ESTIMATORS, max_depth=20, random_state=config.RANDOM_STATE, n_jobs=2),
        "AdaBoost": AdaBoostClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=config.N_ESTIMATORS, learning_rate=0.05, max_depth=6, eval_metric="logloss", random_state=config.RANDOM_STATE, tree_method="hist", n_jobs=2),
        "LightGBM": LGBMClassifier(n_estimators=config.N_ESTIMATORS, learning_rate=0.05, num_leaves=31, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0, random_state=config.RANDOM_STATE, verbose=-1, n_jobs=2),
        "CatBoost": CatBoostClassifier(iterations=config.N_ESTIMATORS, random_seed=config.RANDOM_STATE, verbose=0, thread_count=2, train_dir=config.CATBOOST_TRAIN_DIR),
    }
    # Remove slow / low-value models globally
    for skip in config.OPTUNA_SKIP_BINARY:
        _raw_models.pop(skip, None)

    results = {}
    for name, model in _raw_models.items():
        try:
            # Try each SMOTE ratio; SMOTE is applied inside each CV fold
            best_ratio, best_cv, best_cv_std = "0.6", -1.0, 0.0
            for ratio in _SMOTE_RATIOS:
                try:
                    smote_step = SMOTE(
                        sampling_strategy=ratio,
                        k_neighbors=smote_k,
                        random_state=config.RANDOM_STATE,
                    )
                    pipe = ImbPipeline([("smote", smote_step), ("clf", model)])
                    cv_scores = cross_val_score(
                        pipe, X_tr, y_tr, cv=skf, scoring="roc_auc", n_jobs=2,
                    )
                    mean_auc = float(cv_scores.mean())
                    if mean_auc > best_cv:
                        best_cv = mean_auc
                        best_cv_std = float(cv_scores.std())
                        best_ratio = str(ratio)
                except Exception:
                    continue

            # Refit on full training set with SMOTE at best ratio
            best_ratio_f = float(best_ratio) if best_ratio != "0.6" else 0.6
            sm_final = SMOTE(
                sampling_strategy=best_ratio_f,
                k_neighbors=smote_k,
                random_state=config.RANDOM_STATE,
            ) if can_smote else None
            if sm_final is not None:
                X_fit, y_fit = sm_final.fit_resample(X_tr, y_tr)
            else:
                X_fit, y_fit = X_tr, y_tr
            model.fit(X_fit, y_fit)
            results[name] = _evaluate(model, X_te, y_te)
            results[name]["cv_auc_mean"] = best_cv
            results[name]["cv_auc_std"]  = best_cv_std
            results[name]["smote_ratio"] = best_ratio
            print(f"  {name:<25} best smote={best_ratio}  CV-AUC={best_cv:.4f}")
            _print_metrics(name, results[name])
        except Exception as exc:
            print(f"  ✗ {name} failed: {exc}")
        gc.collect()

    # Track distribution of the default 0.6 variant for reporting
    if can_smote:
        sm_report = SMOTE(sampling_strategy=0.6, k_neighbors=smote_k,
                          random_state=config.RANDOM_STATE)
        _, _y_report = sm_report.fit_resample(X_tr, y_tr)
        _smote_y_counts = pd.Series(_y_report).value_counts().sort_index()
        del sm_report, _y_report
    else:
        _smote_y_counts = pd.Series(y_tr).value_counts().sort_index()
    gc.collect()

    # ── Foundation-model classifiers (Block C) ───────────────
    _run_llm_binary(X_tr, y_tr, X_te, y_te, results,
                    X_tr_raw=data_prep.get("X_train_raw"),
                    X_te_raw=data_prep.get("X_test_raw"))

    cdf = _comparison_df(results)
    _best = cdf.iloc[0]["Model"] if not cdf.empty else None
    if _best:
        print(f"\nBest SMOTE: {_best}")
    return {
        "results": results, "comparison": cdf, "best_model": _best,
        "smote_y_counts": _smote_y_counts,
        **{k: data_prep[k] for k in ("X_train", "X_test", "y_train", "y_test",
                                       "feature_cols", "target_encoding_map", "global_mean")},
    }


# ---- SHAP-based feature pre-filter (now in 02_feature_engineering.py Section 6) ----
# shap_feature_prefilter is imported from 02_feature_engineering at module top.


# ---- Feature selection (SHAP-driven) ----

# ---- B.4 Feature selection (SHAP-driven) ----

def feature_selection(data_prep, baseline_results):
    """Retrain with only top-N consensus features."""
    warnings.filterwarnings("ignore")
    print(f"\n{'=' * 80}")
    print(f"FEATURE SELECTION - Top {config.FEATURE_SELECTION_TOP_N}")
    print(f"{'=' * 80}\n")

    feature_cols = baseline_results["feature_cols"]
    all_imp = {}
    for name in ("Random Forest", "Gradient Boosting", "Extra Trees", "AdaBoost",
                  "XGBoost", "LightGBM", "CatBoost"):
        m = baseline_results["results"].get(name, {}).get("model")
        if m and hasattr(m, "feature_importances_"):
            for f, i in zip(feature_cols, m.feature_importances_):
                all_imp.setdefault(f, []).append(i)

    imp_df = pd.DataFrame([
        {"feature": f, "avg": np.mean(v)} for f, v in all_imp.items()
    ])
    if imp_df.empty:
        print("  ✗ No feature importances available — skipping feature selection")
        return {"results": {}, "comparison": pd.DataFrame(), "best_model": None, "selected_features": []}
    imp_df = imp_df.sort_values("avg", ascending=False)

    selected = imp_df.head(config.FEATURE_SELECTION_TOP_N)["feature"].tolist()
    print(f"Selected: {selected}\n")

    X_tr = data_prep["X_train"][selected]
    X_te = data_prep["X_test"][selected]
    y_tr, y_te = data_prep["y_train"], data_prep["y_test"]
    X_tr, X_te = _sanitize_features(X_tr, X_te, label="FeatSel")

    results = {}
    for name, model in _get_base_models(y_tr).items():
        try:
            model.fit(X_tr, y_tr)
            results[name] = _evaluate(model, X_te, y_te)
            base_auc = baseline_results["results"][name]["roc_auc"]
            cur_auc = results[name]["roc_auc"]
            print(f"  {name:<25} AUC: {cur_auc:.4f} (was {base_auc:.4f}, {cur_auc - base_auc:+.4f})")
        except Exception as exc:
            print(f"  ✗ {name} failed: {exc}")
        gc.collect()

    # ── Foundation-model classifiers (Block C) ───────────────
    _run_llm_binary(X_tr, y_tr, X_te, y_te, results,
                    X_tr_raw=data_prep.get("X_train_raw"),
                    X_te_raw=data_prep.get("X_test_raw"))

    cdf = _comparison_df(results)
    print(f"\nFeatures: {len(feature_cols)} -> {config.FEATURE_SELECTION_TOP_N}")
    gc.collect()
    _best = cdf.iloc[0]["Model"] if not cdf.empty else None
    return {"results": results, "comparison": cdf, "best_model": _best, "selected_features": selected}

# ══════════════════════════════════════════════════════════════════════
# BLOCK C — FOUNDATION-MODEL CLASSIFIERS (LLMs, embeddings, prompting)
# ══════════════════════════════════════════════════════════════════════

# ── LLM Embedding classifier (Bedrock Titan embeddings + LogReg) ─

class _LLMEmbeddingClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible classifier: Bedrock Titan embeddings → LogReg.

    Each row of tabular features is serialised as
    ``col1=val1; col2=val2; …`` text, embedded via Amazon Bedrock
    Titan Text Embeddings V2 (1 024 dims), then classified by a
    Logistic Regression head.

    This is the tabular-LLM baseline recommended by the thesis
    supervisor.  Embeddings are cached to avoid redundant API calls.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._logreg = None
        self._scaler = None
        self._col_names = None
        # Stored for RAG few-shot & stacking reuse
        self._train_embeddings = None   # raw embeddings (n, 1024)
        self._train_texts = None        # serialised row strings
        self._train_labels = None       # y array matching embeddings

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _rows_to_texts(X, col_names=None):
        """Serialise each feature row as a semicolon-delimited string.

        Handles both numeric-only (encoded) and mixed-type (raw) DataFrames.
        """
        if isinstance(X, pd.DataFrame):
            cols = X.columns.tolist()
            arr = X.values
        else:
            arr = np.asarray(X)
            cols = col_names or [f"f{i}" for i in range(arr.shape[1])]
        texts = []
        for row in arr:
            parts = []
            for j, c in enumerate(cols):
                v = row[j]
                if isinstance(v, str):
                    parts.append(f"{c}={v}")
                elif v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                    parts.append(f"{c}=NA")
                elif isinstance(v, (float, np.floating)):
                    parts.append(f"{c}={v:.4g}")
                else:
                    parts.append(f"{c}={v}")
            texts.append("; ".join(parts))
        return texts

    @staticmethod
    def _embed(texts):
        """Call Amazon Bedrock Titan Text Embeddings V2 (concurrent)."""
        import boto3, json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_workers = getattr(config, "LLM_EMBED_WORKERS", 10)
        region    = config.BEDROCK_REGION
        model_id  = config.LLM_EMBEDDING_MODEL
        dims      = config.LLM_EMBEDDING_DIMS
        total     = len(texts)
        results   = [None] * total
        done      = [0]  # mutable counter for threads

        def _call(idx_text):
            idx, text = idx_text
            # Each thread gets its own client (boto3 clients aren't thread-safe)
            client = boto3.client("bedrock-runtime", region_name=region)
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "inputText": text[:8_000],
                    "dimensions": dims,
                    "normalize": True,
                }),
                contentType="application/json",
            )
            emb = json.loads(resp["body"].read())["embedding"]
            results[idx] = emb
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"      embedded {done[0]}/{total} rows", flush=True)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_call, enumerate(texts)))

        print(f"      embedded {total}/{total} rows (done)", flush=True)
        return np.array(results, dtype=np.float32)

    # ── sklearn interface ─────────────────────────────────────

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self._col_names = X.columns.tolist()
        # Subsample train rows to cap Bedrock API calls
        _max = getattr(config, "LLM_EMBED_MAX_SAMPLES", 5_000)
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if len(X_arr) > _max:
            from sklearn.model_selection import train_test_split as _tts
            X_arr, _, y_arr, _ = _tts(
                X_arr, y_arr, train_size=_max,
                random_state=self.random_state, stratify=y_arr,
            )
            print(f"    [LLM-Embed] subsampled {len(X):,} \u2192 {_max:,} train rows")
        texts = self._rows_to_texts(X_arr, self._col_names)
        print(f"    [LLM-Embed] embedding {len(texts):,} train rows \u2026", flush=True)
        emb = self._embed(texts)
        # Store for RAG few-shot & stacking reuse
        self._train_embeddings = emb
        self._train_texts = texts
        self._train_labels = np.asarray(y_arr)
        y = y_arr
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(emb)
        self._logreg = LogisticRegression(
            max_iter=5000, C=1.0,
            random_state=self.random_state, class_weight="balanced",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._logreg.fit(X_sc, np.asarray(y))
        self.classes_ = self._logreg.classes_
        return self

    def predict(self, X):
        emb = self._embed(self._rows_to_texts(X, self._col_names))
        return self._logreg.predict(self._scaler.transform(emb))

    def predict_proba(self, X):
        emb = self._embed(self._rows_to_texts(X, self._col_names))
        return self._logreg.predict_proba(self._scaler.transform(emb))

    def __reduce__(self):
        from . import utils
        return (utils._reconstruct_clf,
                (self.__class__.__module__, type(self).__name__,
                 self.__dict__.copy()))

    # get_params / set_params inherited from BaseEstimator

# ── LLM Prompted classifier (Bedrock Converse — domain-guided) ───

class _LLMPromptedClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible binary classifier: Bedrock Converse with
    domain-specific prompts, statistical context, and few-shot examples.

    Unlike LLM-Embed (which converts rows → embeddings → LogReg),
    this model asks the LLM to **directly predict** whether a vehicle
    will come for retrofit, guided by:

      1. **Domain context** — what each feature means in the vehicle
         retrofit domain (emission class, vehicle age, etc.)
      2. **Statistical summaries** — class-conditional feature means
         computed from training data so the LLM knows what "typical"
         positive and negative cases look like.
      3. **Few-shot examples** — K real training rows (stratified) with
         their labels, so the LLM sees concrete patterns.
      4. **Structured output** — the LLM returns JSON with a probability
         and a short natural-language explanation (built-in XAI).

    The prompt is fully deterministic (temperature=0) and the JSON
    output is parsed into ``predict`` / ``predict_proba`` arrays.

    This is deliberately slow (1 Bedrock call per row) so it is
    subsampled to ``LLM_PROMPTED_MAX_SAMPLES`` rows at test time.
    """

    # ── Feature descriptions for the system prompt ────────────
    # Maps encoded column names → human-readable descriptions.
    # Covers the most common features after target-encoding + cleansing.
    _FEATURE_DESCRIPTIONS = {
        # Categorical (target-encoded → numeric)
        config.COL_DERIVAT_CLEAN: "Vehicle derivative / model variant",
        "basistyp_clean":       "Base vehicle type (platform)",
        "antriebsart_clean":    "Powertrain type (diesel, petrol, hybrid, electric)",
        "motortyp_clean":       "Engine type code",
        "basisausfuehrung_clean": "Base trim / equipment level",
        "pep_bauphase_clean":   "PEP build phase (development stage of the model)",
        "owner_fleet_clean": "Owner / fleet operator",
        "hybrid_clean":         "Hybrid flag (yes/no → encoded)",
        "prio_clean":           "Retrofit priority level assigned by planning",
        "umr_paket_name_clean": "Retrofit package name / campaign",
        "werkstattgruppe_clean": "Workshop group (region / capacity cluster)",
        "werkstattname_clean":  "Workshop name (specific service location)",
        "laendervariante_clean": "Country variant of the vehicle",
        # Numeric / date-derived
        "vehicle_age_days":     "Vehicle age in days since first registration",
        "days_since_sop":       "Days since Start of Production (SOP) of this model",
        "days_to_deadline":     "Days remaining until the regulatory retrofit deadline",
        "month_sin":            "Sine of calendar month (seasonality feature)",
        "month_cos":            "Cosine of calendar month (seasonality feature)",
        "year":                 "Calendar year of the observation",
    }

    # ── Domain context for the system prompt ────────────────
    _DOMAIN_CONTEXT_BINARY = (
        "DOMAIN BACKGROUND — Vehicle Retrofits\n"
        "A retrofit is a mandatory post-sale modification to vehicles\n"
        "already in the field, triggered by regulatory, safety, or quality reasons.\n"
        "The OEM tracks these through two main systems:\n"
        "  • Source A — planning system that defines retrofit campaigns and deadlines.\n"
        "  • Source B — dealer workshop system that records which vehicles actually\n"
        "    came in and had the retrofit performed.\n\n"
        "There are several main retrofit types (software updates, hardware replacements,\n"
        "mechatronic actions, type-approval retrofits, etc.).\n\n"
        "Key domain knowledge:\n"
        "  • A single vehicle can be subject to multiple campaigns simultaneously.\n"
        "  • 'days_to_deadline' is among the strongest predictors: vehicles near\n"
        "    their regulatory deadline are much more likely to come in.\n"
        "  • Fleet / corporate-owned vehicles tend to\n"
        "    have higher retrofit compliance than private owners.\n"
        "  • Seasonal patterns exist: workshop visits peak before winter and\n"
        "    around annual vehicle inspections.\n"
        "  • The retrofit priority level (prio_clean) set by planning reflects\n"
        "    urgency: higher priority → more dealer outreach → higher show-up rate.\n"
    )

    def __init__(self, random_state=42, embed_clf=None):
        self.random_state = random_state
        self.embed_clf = embed_clf   # trained _LLMEmbeddingClassifier for RAG
        self._col_names = None
        self._top_feature_idx = None # indices of selected features (or None = all)
        self._few_shot_pos = []      # K positive examples (text + label)
        self._few_shot_neg = []      # K negative examples
        self._class_stats = ""       # statistical summary string
        self._feature_desc = ""      # feature description block
        self.classes_ = np.array([0, 1])

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _row_to_text(row_vals, col_names):
        """Serialise one feature row as 'col = value' lines.

        Handles both numeric (encoded) and string (raw) values.
        """
        parts = []
        for c, v in zip(col_names, row_vals):
            if isinstance(v, str):
                parts.append(f"  {c} = {v}")
            elif v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                parts.append(f"  {c} = NA")
            elif isinstance(v, (float, np.floating)):
                parts.append(f"  {c} = {v:.4g}")
            else:
                parts.append(f"  {c} = {v}")
        return "\n".join(parts)

    def _build_feature_descriptions(self, col_names):
        """Build a human-readable block describing each feature."""
        lines = []
        for c in col_names:
            desc = self._FEATURE_DESCRIPTIONS.get(c)
            if desc:
                lines.append(f"  • {c}: {desc}")
            else:
                if c.endswith("_clean"):
                    base = c.replace("_clean", "")
                    lines.append(f"  • {c}: Categorical feature '{base}'")
                elif c.endswith("_freq"):
                    base = c.replace("_freq", "")
                    lines.append(f"  • {c}: Frequency encoding of '{base}'")
                else:
                    lines.append(f"  • {c}: Numeric feature")
        return "\n".join(lines)

    def _build_class_stats(self, X_arr, y_arr, col_names):
        """Compute per-class feature summaries for context.

        Numeric columns → mean; string columns → most common value.
        """
        df = pd.DataFrame(X_arr, columns=col_names)
        df["_label"] = y_arr
        pos = df[df["_label"] == 1].drop(columns="_label")
        neg = df[df["_label"] == 0].drop(columns="_label")

        lines = [f"Training set: {len(df):,} rows "
                 f"({int(y_arr.sum()):,} positive = {y_arr.mean():.1%} retrofit rate)\n"]
        lines.append("Feature summaries by class:")
        lines.append(f"  {'Feature':<35} {'Came (1)':>14}  {'Did not (0)':>14}")
        lines.append(f"  {'─'*35} {'─'*14}  {'─'*14}")
        for c in col_names:
            if df[c].dtype == object:
                pm = pos[c].mode().iloc[0] if len(pos) and len(pos[c].mode()) else "?"
                nm = neg[c].mode().iloc[0] if len(neg) and len(neg[c].mode()) else "?"
                lines.append(f"  {c:<35} {str(pm):>14}  {str(nm):>14}")
            else:
                pm = pos[c].mean() if len(pos) else np.nan
                nm = neg[c].mean() if len(neg) else np.nan
                lines.append(f"  {c:<35} {pm:>14.3f}  {nm:>14.3f}")
        return "\n".join(lines)

    def _select_few_shot(self, X_arr, y_arr, col_names, k=None):
        """Select K stratified few-shot examples per class."""
        if k is None:
            k = getattr(config, "LLM_PROMPTED_FEW_SHOT_K", 3)
        rng = np.random.RandomState(self.random_state)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        pos_examples = []
        for i in pos_idx[:k]:
            txt = self._row_to_text(X_arr[i], col_names)
            pos_examples.append(txt)

        neg_examples = []
        for i in neg_idx[:k]:
            txt = self._row_to_text(X_arr[i], col_names)
            neg_examples.append(txt)

        return pos_examples, neg_examples

    def _select_few_shot_rag(self, row_text, k=None):
        """RAG few-shot: pick K nearest training examples per class via
        cosine similarity on Titan embeddings stored in embed_clf.

        Falls back to the static (random) few-shot if embed_clf is
        unavailable or hasn't stored embeddings yet.
        """
        if k is None:
            k = getattr(config, "LLM_PROMPTED_FEW_SHOT_K", 3)
        ec = self.embed_clf
        if ec is None or not hasattr(ec, "_train_embeddings") or ec._train_embeddings is None:
            return self._few_shot_pos, self._few_shot_neg

        from sklearn.metrics.pairwise import cosine_similarity

        # Embed the query row (single text)
        query_emb = ec._embed([row_text])                # (1, d)
        sims = cosine_similarity(query_emb, ec._train_embeddings)[0]  # (N,)

        labels = ec._train_labels
        texts  = ec._train_texts

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        # Top-k most similar positives
        pos_sims = np.where(pos_mask, sims, -2.0)
        pos_top  = np.argsort(pos_sims)[-k:][::-1]
        pos_examples = [self._row_to_text_from_str(texts[i]) for i in pos_top]

        # Top-k most similar negatives
        neg_sims = np.where(neg_mask, sims, -2.0)
        neg_top  = np.argsort(neg_sims)[-k:][::-1]
        neg_examples = [self._row_to_text_from_str(texts[i]) for i in neg_top]

        return pos_examples, neg_examples

    def _select_top_features(self, X_arr, y_arr, col_names, top_k=None):
        """Pick top-K features by mutual information.

        String (object) columns are always kept because they carry
        human-readable categories valuable for the LLM.  Remaining
        budget is filled with the highest-MI numeric columns.

        Returns (selected_idx, selected_col_names).
        """
        if top_k is None:
            top_k = getattr(config, "LLM_TOP_FEATURES", 15)
        if len(col_names) <= top_k:
            return list(range(len(col_names))), list(col_names)

        from sklearn.feature_selection import mutual_info_classif

        df = pd.DataFrame(X_arr, columns=col_names)

        str_cols = [c for c in col_names if df[c].dtype == object]
        num_cols = [c for c in col_names if df[c].dtype != object]

        if len(str_cols) >= top_k:
            sel_names = str_cols[:top_k]
        else:
            remaining = top_k - len(str_cols)
            if num_cols:
                X_num = df[num_cols].values.astype(np.float64)
                X_num = np.nan_to_num(X_num, nan=0.0)
                mi = mutual_info_classif(
                    X_num, y_arr, random_state=self.random_state)
                top_idx = np.argsort(mi)[-remaining:][::-1]
                top_num = [num_cols[i] for i in top_idx]
            else:
                top_num = []
            sel_names = str_cols + top_num

        sel_idx = [col_names.index(c) for c in sel_names]
        return sel_idx, sel_names

    @staticmethod
    def _row_to_text_from_str(semicolon_text):
        """Convert 'col=val; col2=val2' semicolon text → 'col = val' line format."""
        parts = semicolon_text.split("; ")
        lines = []
        for p in parts:
            eq = p.split("=", 1)
            if len(eq) == 2:
                lines.append(f"  {eq[0]} = {eq[1]}")
            else:
                lines.append(f"  {p}")
        return "\n".join(lines)

    def _build_prompt(self, row_text, few_shot_pos=None, few_shot_neg=None):
        """Assemble the full prompt for one vehicle.

        When *few_shot_pos*/*few_shot_neg* are supplied (RAG per-row),
        they override the static examples stored during fit().
        """
        pos = few_shot_pos if few_shot_pos is not None else self._few_shot_pos
        neg = few_shot_neg if few_shot_neg is not None else self._few_shot_neg
        k = len(pos)

        # Few-shot block
        fs_lines = []
        for i, txt in enumerate(pos):
            fs_lines.append(f"Example {i+1} (CAME for retrofit, label=1):\n{txt}")
        for i, txt in enumerate(neg):
            fs_lines.append(f"Example {k+i+1} (DID NOT come, label=0):\n{txt}")
        few_shot_block = "\n\n".join(fs_lines)

        prompt = (
            "You are an expert vehicle-retrofit analyst at an automotive OEM.\n"
            "Your task: predict whether a vehicle will come for a retrofit\n"
            "based on its features.\n\n"
            "=== DOMAIN CONTEXT ===\n"
            f"{self._DOMAIN_CONTEXT_BINARY}\n\n"
            "=== FEATURE DEFINITIONS ===\n"
            f"{self._feature_desc}\n\n"
            "=== TRAINING DATA STATISTICS ===\n"
            f"{self._class_stats}\n\n"
            "=== FEW-SHOT EXAMPLES (most similar to the vehicle below) ===\n"
            f"{few_shot_block}\n\n"
            "=== VEHICLE TO CLASSIFY ===\n"
            f"{row_text}\n\n"
            "=== INSTRUCTIONS ===\n"
            "Think step-by-step before answering:\n"
            "  1. Identify the most predictive features for this vehicle.\n"
            "  2. Compare them against the training statistics and few-shot examples.\n"
            "  3. Consider domain knowledge (deadline proximity, fleet vs private,\n"
            "     retrofit priority, seasonality).\n"
            "  4. Arrive at a probability and binary prediction.\n\n"
            "Return ONLY valid JSON (no markdown, no text outside the JSON):\n"
            '{"reasoning": "<step-by-step analysis, 3-5 sentences>", '
            '"probability": <float 0-1>, "prediction": <0 or 1>, '
            '"explanation": "<1-sentence summary of key drivers>"}\n'
        )
        return prompt

    def _call_bedrock(self, prompt, temperature=None):
        """Single Bedrock Converse call → parsed JSON dict."""
        import boto3, json

        _temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        client = boto3.client("bedrock-runtime",
                              region_name=config.BEDROCK_REGION)
        resp = client.converse(
            modelId=config.LLM_CHAT_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": _temp,
                             "maxTokens": 1024},
        )
        # Robust extraction: iterate content blocks for a "text" key
        content_blocks = (resp.get("output", {})
                              .get("message", {})
                              .get("content", []))
        raw = None
        for block in content_blocks:
            if isinstance(block, dict) and "text" in block:
                raw = block["text"].strip()
                break
        if raw is None:
            if content_blocks:
                raw = str(content_blocks[0]).strip()
            else:
                raise ValueError("Empty response from Bedrock Converse")

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)
        return result

    def _predict_batch(self, X):
        """Concurrent prediction for a batch of rows."""
        import json
        from concurrent.futures import ThreadPoolExecutor

        if isinstance(X, pd.DataFrame):
            arr = X.values
            cols = X.columns.tolist()
        else:
            arr = np.asarray(X)
            cols = [f"f{i}" for i in range(arr.shape[1])]

        # Always apply feature selection when indices were computed during fit
        if self._top_feature_idx is not None:
            arr = arr[:, self._top_feature_idx]
            cols = self._col_names

        n = len(arr)
        results = [None] * n
        done = [0]
        n_workers = getattr(config, "LLM_EMBED_WORKERS", 10)

        _use_rag = (self.embed_clf is not None
                    and hasattr(self.embed_clf, "_train_embeddings")
                    and self.embed_clf._train_embeddings is not None)

        _sc_n = getattr(config, "LLM_SELF_CONSISTENCY_N", 1)
        _sc_temp = getattr(config, "LLM_CONSISTENCY_TEMP", 0.7)

        def _call_one(idx):
            row_text = self._row_to_text(arr[idx], cols)
            # RAG: select per-row few-shot examples via embedding similarity
            if _use_rag:
                row_semi = "; ".join(
                    f"{c}={arr[idx][j]}" for j, c in enumerate(cols))
                pos_fs, neg_fs = self._select_few_shot_rag(row_semi)
                prompt = self._build_prompt(row_text, pos_fs, neg_fs)
            else:
                prompt = self._build_prompt(row_text)
            try:
                if _sc_n <= 1:
                    parsed = self._call_bedrock(prompt)
                else:
                    # Self-consistency: N calls with higher temp → majority vote
                    votes = []
                    for _ in range(_sc_n):
                        votes.append(self._call_bedrock(prompt, temperature=_sc_temp))
                    preds = [v.get("prediction", 0) for v in votes]
                    probs = [max(0.0, min(1.0, float(v.get("probability", 0.5)))) for v in votes]
                    majority = max(set(preds), key=preds.count)
                    parsed = {
                        "probability": float(np.mean(probs)),
                        "prediction": majority,
                        "explanation": votes[0].get("explanation", ""),
                        "reasoning": votes[0].get("reasoning", ""),
                    }
                results[idx] = parsed
            except Exception as exc:
                # Fallback: 50/50 if LLM fails to parse
                results[idx] = {"probability": 0.5, "prediction": 0,
                                "explanation": f"Parse error: {exc}"}
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == n:
                print(f"      [LLM-Prompted] {done[0]}/{n} rows classified",
                      flush=True)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_call_one, range(n)))

        return results

    # ── sklearn interface ─────────────────────────────────────

    def fit(self, X, y):
        """Learn domain context from training data (no LLM calls needed)."""
        if isinstance(X, pd.DataFrame):
            self._col_names = X.columns.tolist()
            # Preserve mixed types (strings + numerics) for the LLM prompt
            X_arr = X.values  # object array keeps string categories
        else:
            X_arr = np.asarray(X)
            self._col_names = self._col_names or [f"f{i}" for i in range(X_arr.shape[1])]
        y_arr = np.asarray(y)

        # Subsample for stats computation (fast, local — no API calls)
        _max = getattr(config, "LLM_EMBED_MAX_SAMPLES", 5_000)
        if len(X_arr) > _max:
            from sklearn.model_selection import train_test_split as _tts
            X_arr, _, y_arr, _ = _tts(
                X_arr, y_arr, train_size=_max,
                random_state=self.random_state, stratify=y_arr,
            )
            print(f"    [LLM-Prompted] subsampled {len(X):,} → {_max:,} for context")

        # Feature selection — keep only the most informative columns
        all_cols = list(self._col_names)
        self._top_feature_idx, sel_cols = self._select_top_features(
            X_arr, y_arr, all_cols)
        X_sel = X_arr[:, self._top_feature_idx]
        print(f"    [LLM-Prompted] selected {len(sel_cols)}/{len(all_cols)} "
              f"features by mutual information")

        # Build the three prompt components from selected features
        self._feature_desc = self._build_feature_descriptions(sel_cols)
        self._class_stats = self._build_class_stats(X_sel, y_arr, sel_cols)
        self._few_shot_pos, self._few_shot_neg = self._select_few_shot(
            X_sel, y_arr, sel_cols)

        # Store selected column names for prediction-time filtering
        self._col_names = sel_cols

        k = len(self._few_shot_pos)
        print(f"    [LLM-Prompted] fit complete: {len(sel_cols)} features, "
              f"{k}+{k} few-shot examples, "
              f"{y_arr.mean():.1%} positive rate")
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        results = self._predict_batch(X)
        preds = np.array([r.get("prediction", 0) for r in results], dtype=int)
        return preds

    def predict_proba(self, X):
        results = self._predict_batch(X)
        probs = np.array([
            max(0.0, min(1.0, float(r.get("probability", 0.5))))
            for r in results
        ], dtype=np.float64)
        return np.column_stack([1 - probs, probs])

    def predict_with_explanations(self, X):
        """Like predict_proba but also returns per-row explanations.

        Returns
        -------
        proba : ndarray (n, 2)
        explanations : list[str]
        """
        results = self._predict_batch(X)
        probs = np.array([
            max(0.0, min(1.0, float(r.get("probability", 0.5))))
            for r in results
        ], dtype=np.float64)
        explanations = [r.get("explanation", "") for r in results]
        return np.column_stack([1 - probs, probs]), explanations

    def __reduce__(self):
        from . import utils
        return (utils._reconstruct_clf,
                (self.__class__.__module__, type(self).__name__,
                 self.__dict__.copy()))

    # get_params / set_params inherited from BaseEstimator


# ── C helper: train & evaluate both LLM classifiers ─────────

def _run_llm_binary(X_tr, y_tr, X_te, y_te, results,
                    X_tr_raw=None, X_te_raw=None):
    """Train & evaluate both FM classifiers, appending to *results*.

    When *X_tr_raw* / *X_te_raw* are provided (pre-encoding DataFrames
    with original text categories), the LLM sees human-readable values
    like ``antriebsart_clean=Diesel`` instead of target-encoded floats.

    Handles _BEDROCK guard, test-set subsampling, and graceful failure.
    Called from every public training function (A.3, B.2–B.4).
    """
    if not _BEDROCK:
        return

    # Prefer raw (text) data when available — better for LLMs
    _X_tr = X_tr_raw if X_tr_raw is not None else X_tr
    _X_te = X_te_raw if X_te_raw is not None else X_te

    # ── LLM-Embed (Bedrock Titan embeddings → LogReg) ────────
    print(f"\n  LLM-Embed LogReg (Bedrock {config.LLM_EMBEDDING_MODEL}) …")
    try:
        llm_clf = _LLMEmbeddingClassifier(random_state=config.RANDOM_STATE)
        llm_clf.fit(_X_tr, y_tr)
        _mx = getattr(config, "LLM_EMBED_MAX_SAMPLES", 5_000)
        if len(_X_te) > _mx:
            from sklearn.model_selection import train_test_split as _tts
            _X_te_s, _, _y_te_s, _ = _tts(
                _X_te, y_te, train_size=_mx,
                random_state=config.RANDOM_STATE, stratify=y_te)
            print(f"    [LLM-Embed] test subset {len(_X_te):,} → {_mx:,}")
        else:
            _X_te_s, _y_te_s = _X_te, y_te
        results["LLM-Embed LogReg"] = _evaluate(llm_clf, _X_te_s, _y_te_s)
        _print_metrics("LLM-Embed LogReg", results["LLM-Embed LogReg"])
    except Exception as exc:
        print(f"  ✗ LLM-Embed LogReg failed: {exc}")
    gc.collect()

    # ── LLM-Prompted (Bedrock Converse, domain-guided + RAG) ─
    print(f"\n  LLM-Prompted (Bedrock {config.LLM_CHAT_MODEL}) …")
    try:
        llm_p = _LLMPromptedClassifier(
            random_state=config.RANDOM_STATE,
            embed_clf=llm_clf if 'llm_clf' in dir() else None,
        )
        llm_p.fit(_X_tr, y_tr)
        _mx_p = getattr(config, "LLM_PROMPTED_MAX_SAMPLES", 200)
        if len(_X_te) > _mx_p:
            from sklearn.model_selection import train_test_split as _tts
            _X_te_p, _, _y_te_p, _ = _tts(
                _X_te, y_te, train_size=_mx_p,
                random_state=config.RANDOM_STATE, stratify=y_te)
            print(f"    [LLM-Prompted] test subset {len(_X_te):,} → {_mx_p:,}")
        else:
            _X_te_p, _y_te_p = _X_te, y_te
        results["LLM-Prompted"] = _evaluate(llm_p, _X_te_p, _y_te_p)
        _print_metrics("LLM-Prompted", results["LLM-Prompted"])
    except Exception as exc:
        print(f"  ✗ LLM-Prompted failed: {exc}")
    gc.collect()

    # ── LLM-Prompted-Calibrated (Platt scaling) ─────────────
    if "LLM-Prompted" in results and not results["LLM-Prompted"].get("degenerate", False):
        print(f"\n  LLM-Prompted-Calibrated (Platt scaling) …")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import train_test_split as _tts

            _cal_n = getattr(config, "LLM_CALIBRATION_SAMPLES", 50)

            # Split a small calibration set from training data
            _X_cal, _, _y_cal, _ = _tts(
                _X_tr, y_tr, train_size=min(_cal_n, len(_X_tr) // 2),
                random_state=config.RANDOM_STATE, stratify=y_tr,
            )

            # Wrap the already-fitted LLM-Prompted classifier with Platt scaling
            # sklearn >= 1.6 removed cv="prefit"; use FrozenEstimator instead
            try:
                from sklearn.frozen import FrozenEstimator
                cal_clf = CalibratedClassifierCV(
                    FrozenEstimator(llm_p), method="sigmoid", cv=2,
                )
            except ImportError:
                cal_clf = CalibratedClassifierCV(
                    llm_p, method="sigmoid", cv="prefit",
                )
            cal_clf.fit(_X_cal, _y_cal)

            # Evaluate on the same test subset used for LLM-Prompted
            _X_te_cal = _X_te_p if '_X_te_p' in dir() else _X_te
            _y_te_cal = _y_te_p if '_y_te_p' in dir() else y_te
            results["LLM-Prompted-Calibrated"] = _evaluate(
                cal_clf, _X_te_cal, _y_te_cal)
            _print_metrics("LLM-Prompted-Calibrated",
                           results["LLM-Prompted-Calibrated"])
        except Exception as exc:
            print(f"  ✗ LLM-Prompted-Calibrated failed: {exc}")
        gc.collect()

    # ── LLM-Stacked (best ML + LLM → LogReg meta-learner) ───
    if "LLM-Prompted" in results:
        print(f"\n  LLM-Stacked (best ML + LLM-Prompted → LogReg) …")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split as _tts

            # Find best traditional ML model (exclude LLM-* models)
            _ml_names = [k for k in results
                         if not k.startswith("LLM") and "model" in results[k]]
            if not _ml_names:
                raise RuntimeError("No traditional ML model available")
            _best_ml_name = max(
                _ml_names, key=lambda k: results[k].get("roc_auc", 0))
            _best_ml = results[_best_ml_name]["model"]
            print(f"    [LLM-Stacked] base ML model: {_best_ml_name}")

            # --- Meta-training set (held-out slice of training data) ---
            _stack_n = min(
                getattr(config, "LLM_STACKED_MAX_SAMPLES", 2_000),
                len(y_tr) // 2)
            _idx_tr = np.arange(len(y_tr))
            _, _idx_meta, _, _y_meta = _tts(
                _idx_tr, y_tr, test_size=_stack_n,
                random_state=config.RANDOM_STATE, stratify=y_tr)

            _X_meta_enc = (X_tr.iloc[_idx_meta] if isinstance(X_tr, pd.DataFrame)
                           else X_tr[_idx_meta])
            _X_meta_raw = (_X_tr.iloc[_idx_meta] if isinstance(_X_tr, pd.DataFrame)
                           else _X_tr[_idx_meta])

            _prob_ml_meta = _best_ml.predict_proba(_X_meta_enc)[:, 1]
            print(f"    [LLM-Stacked] LLM inference on {_stack_n} meta rows …")
            _prob_llm_meta = llm_p.predict_proba(_X_meta_raw)[:, 1]

            _meta_X = np.column_stack([_prob_ml_meta, _prob_llm_meta])
            _meta_lr = LogisticRegression(random_state=config.RANDOM_STATE)
            _meta_lr.fit(_meta_X, _y_meta)
            _w = _meta_lr.coef_[0]
            print(f"    [LLM-Stacked] meta weights: ML={_w[0]:.3f}, "
                  f"LLM={_w[1]:.3f}")

            # --- Test evaluation (reuse stored LLM predictions) ------
            _llm_proba_te = results["LLM-Prompted"]["y_pred_proba"]
            _y_te_stack = results["LLM-Prompted"]["y_test_eval"]

            # Reproduce same test subsampling on encoded data
            _mx_p = getattr(config, "LLM_PROMPTED_MAX_SAMPLES", 200)
            if len(X_te) > _mx_p:
                _X_te_enc_s, _, _, _ = _tts(
                    X_te, y_te, train_size=_mx_p,
                    random_state=config.RANDOM_STATE, stratify=y_te)
            else:
                _X_te_enc_s = X_te
            _ml_proba_te = _best_ml.predict_proba(_X_te_enc_s)[:, 1]

            _stack_te = np.column_stack([_ml_proba_te, _llm_proba_te])
            _stack_proba = _meta_lr.predict_proba(_stack_te)[:, 1]
            _stack_pred = (_stack_proba >= 0.5).astype(int)

            # Build result dict (same schema as _evaluate)
            _tn, _fp, _fn, _tp = confusion_matrix(
                _y_te_stack, _stack_pred).ravel()
            _prec_c, _rec_c, _thr_c = precision_recall_curve(
                _y_te_stack, _stack_proba)
            _f1_c = 2 * _prec_c * _rec_c / (_prec_c + _rec_c + 1e-8)
            _best_idx = int(np.argmax(_f1_c))
            _opt_thr = (float(_thr_c[_best_idx])
                        if _best_idx < len(_thr_c) and _f1_c[_best_idx] > 0
                        else 0.5)
            _y_pred_opt = (_stack_proba >= _opt_thr).astype(int)
            _tn_o, _fp_o, _fn_o, _tp_o = confusion_matrix(
                _y_te_stack, _y_pred_opt).ravel()

            results["LLM-Stacked"] = {
                "model": _meta_lr,
                "accuracy": accuracy_score(_y_te_stack, _stack_pred),
                "precision": precision_score(
                    _y_te_stack, _stack_pred, zero_division=0),
                "recall": recall_score(
                    _y_te_stack, _stack_pred, zero_division=0),
                "f1_score": f1_score(
                    _y_te_stack, _stack_pred, zero_division=0),
                "roc_auc": roc_auc_score(_y_te_stack, _stack_proba),
                "pr_auc": average_precision_score(
                    _y_te_stack, _stack_proba),
                "confusion_matrix": {"tn": int(_tn), "fp": int(_fp),
                                     "fn": int(_fn), "tp": int(_tp)},
                "confusion_matrix_opt": {
                    "tn": int(_tn_o), "fp": int(_fp_o),
                    "fn": int(_fn_o), "tp": int(_tp_o)},
                "y_pred": _stack_pred,
                "y_pred_proba": _stack_proba,
                "y_test_eval": _y_te_stack,
                "y_pred_opt": _y_pred_opt,
                "optimal_threshold": _opt_thr,
                "f1_at_optimal": float(f1_score(
                    _y_te_stack, _y_pred_opt, zero_division=0)),
                "recall_at_optimal": float(recall_score(
                    _y_te_stack, _y_pred_opt, zero_division=0)),
                "precision_at_optimal": float(precision_score(
                    _y_te_stack, _y_pred_opt, zero_division=0)),
                "base_ml_model": _best_ml_name,
            }
            _print_metrics("LLM-Stacked", results["LLM-Stacked"])
        except Exception as exc:
            print(f"  ✗ LLM-Stacked failed: {exc}")
        gc.collect()


# ══════════════════════════════════════════════════════════════
# BLOCK D — EVALUATION (bootstrap CIs, McNemar, ensemble, final ranking)
# ══════════════════════════════════════════════════════════════

# ── Bootstrap Confidence Intervals ───────────────────────────

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000,
                 ci=0.95, seed=None):
    """
    Non-parametric bootstrap confidence interval for any metric.

    Parameters
    ----------
    y_true   : array-like – true binary labels
    y_score  : array-like – predicted probabilities or binary preds
    metric_fn: callable(y_true, y_score) → float
    n_boot   : int   – bootstrap iterations (default 2 000)
    ci       : float – confidence level (default 0.95)
    seed     : int   – random seed for reproducibility

    Returns
    -------
    dict  with keys 'point', 'ci_lo', 'ci_hi', 'std', 'samples'
    """
    rng = np.random.RandomState(seed or config.RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        # Ensure both classes present in bootstrap sample
        if len(np.unique(y_true[idx])) < 2:
            scores[b] = np.nan
            continue
        scores[b] = metric_fn(y_true[idx], y_score[idx])
    scores = scores[~np.isnan(scores)]
    alpha = 1 - ci
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return {
        "point": float(metric_fn(y_true, y_score)),
        "ci_lo": lo,
        "ci_hi": hi,
        "std": float(np.std(scores)),
        "samples": scores,
    }


def bootstrap_model_metrics(y_true, y_proba, n_boot=2000, ci=0.95):
    """
    Bootstrap CIs for both AUC and F1 (at default 0.5 threshold).

    Returns dict with 'auc' and 'f1' sub-dicts, each containing
    point, ci_lo, ci_hi, std, samples.
    """
    y_pred_binary = (np.asarray(y_proba) >= 0.5).astype(int)

    auc_ci = bootstrap_ci(
        y_true, y_proba,
        metric_fn=roc_auc_score,
        n_boot=n_boot, ci=ci,
    )
    f1_ci = bootstrap_ci(
        y_true, y_pred_binary,
        metric_fn=lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n_boot=n_boot, ci=ci,
    )
    pr_auc_ci = bootstrap_ci(
        y_true, y_proba,
        metric_fn=average_precision_score,
        n_boot=n_boot, ci=ci,
    )
    return {"auc": auc_ci, "f1": f1_ci, "pr_auc": pr_auc_ci}


# ── McNemar's Test (paired model comparison) ────────────────

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's exact test comparing two classifiers on the same test set.

    H0: both classifiers have the same error rate.

    Parameters
    ----------
    y_true   : true labels
    y_pred_a : predictions from model A
    y_pred_b : predictions from model B

    Returns
    -------
    dict with 'b' (A wrong, B right), 'c' (A right, B wrong),
    'statistic', 'p_value', 'significant' (at α = 0.05).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A wrong, B right;  c = A right, B wrong
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))

    # Use exact binomial test (more reliable than chi² for small b+c)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0,
                "p_value": 1.0, "significant": False}

    # Two-sided binomial test: under H0, b ~ Binom(n, 0.5)
    try:
        # scipy ≥ 1.7
        p_value = float(sp_stats.binomtest(b, n, 0.5).pvalue)
    except AttributeError:
        # scipy < 1.7 fallback
        p_value = float(sp_stats.binom_test(b, n, 0.5))

    # Also compute chi² statistic for reporting
    chi2 = (abs(b - c) - 1) ** 2 / max(b + c, 1)

    return {
        "b": b, "c": c,
        "statistic": float(chi2),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def pairwise_mcnemar(y_true, results_dict):
    """
    Run McNemar's test for all pairs of models.

    Parameters
    ----------
    y_true       : true labels (shared test set)
    results_dict : {model_name: {... 'y_pred': array ...}}

    Returns
    -------
    pd.DataFrame with columns Model_A, Model_B, b, c, chi2, p_value, significant
    """
    n_full = len(np.asarray(y_true))
    names = [n for n in results_dict
             if results_dict[n].get("y_pred") is not None
             and len(np.asarray(results_dict[n]["y_pred"])) == n_full]
    rows = []
    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            res = mcnemar_test(
                y_true,
                results_dict[a]["y_pred"],
                results_dict[b_name]["y_pred"],
            )
            rows.append({
                "Model_A": a, "Model_B": b_name,
                "b (A✗ B✓)": res["b"], "c (A✓ B✗)": res["c"],
                "χ²": res["statistic"],
                "p_value": res["p_value"],
                "significant": res["significant"],
            })
    return pd.DataFrame(rows)


def compare_results(results_a, results_b, label_a="Baseline", label_b="Tuned", metric="roc_auc"):
    """Print side-by-side metric comparison."""
    print(f"\n{'=' * 80}")
    print(f"{label_a} vs {label_b}")
    print(f"{'=' * 80}\n")
    for name in results_a["results"]:
        if name in results_b["results"]:
            a = results_a["results"][name][metric]
            b = results_b["results"][name][metric]
            sym = "+" if b > a else "-" if b < a else "="
            print(f"  {name:<25} {a:.4f} -> {b:.4f}  ({sym}{abs(b - a):.4f})")


def ensemble_voting(source_results, top_n=None):
    """Weighted soft-voting ensemble of top-N models."""
    top_n = top_n or config.ENSEMBLE_TOP_N
    print(f"\n{'=' * 80}")
    print(f"ENSEMBLE VOTING (soft) - Top {top_n}")
    print(f"{'=' * 80}\n")

    y_test = source_results["y_test"]
    n_full = len(y_test)
    # Exclude subsampled models (e.g. LLM-Embed with 5k test) — their
    # y_pred_proba arrays are shorter than y_test and cannot be summed.
    top_models = [m for m in source_results["comparison"].head(top_n + 5)["Model"].tolist()
                  if len(np.asarray(source_results["results"][m].get("y_pred_proba", []))) == n_full][:top_n]

    if not top_models:
        print("  ✗ No full-test models available for ensemble — skipping")
        return None
    scores = [source_results["results"][m]["roc_auc"] for m in top_models]
    total = sum(scores)
    if total == 0:
        weights = [1.0 / len(scores)] * len(scores)
    else:
        weights = [s / total for s in scores]

    for name, w in zip(top_models, weights):
        print(f"  {name:<25} weight: {w:.3f}")

    proba = np.zeros(len(y_test))
    for m, w in zip(top_models, weights):
        proba += w * source_results["results"][m]["y_pred_proba"]

    pred = (proba > 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    prec = precision_score(y_test, pred, zero_division=0)
    acc = accuracy_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    # Optimal threshold (maximise F1)
    prec_c, rec_c, thr_c = precision_recall_curve(y_test, proba)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
    best_idx = int(np.argmax(f1_c))
    opt_thr = float(thr_c[best_idx]) if best_idx < len(thr_c) and f1_c[best_idx] > 0 else 0.5
    y_pred_opt = (proba >= opt_thr).astype(int)

    print(f"\n  Ensemble AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f}")
    pr_auc_val = average_precision_score(y_test, proba)
    print(f"  Optimal thr={opt_thr:.3f}: F1={f1_score(y_test, y_pred_opt, zero_division=0):.4f} "
          f"Recall={recall_score(y_test, y_pred_opt, zero_division=0):.4f}")
    print(f"  vs best individual: {auc - max(scores):+.4f}")

    return {
        "ensemble_model": "Weighted Voting Ensemble",
        "component_models": top_models, "weights": weights,
        "roc_auc": auc, "pr_auc": pr_auc_val, "f1_score": f1, "recall": rec,
        "precision": prec, "accuracy": acc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "optimal_threshold": opt_thr,
        "f1_at_optimal": float(f1_score(y_test, y_pred_opt, zero_division=0)),
        "recall_at_optimal": float(recall_score(y_test, y_pred_opt, zero_division=0)),
        "precision_at_optimal": float(precision_score(y_test, y_pred_opt, zero_division=0)),
        "y_pred_proba": proba,
    }


def final_model_comparison(*result_dicts, n_boot=2000):
    """
    Rank all approaches with bootstrap CIs and McNemar significance.

    Steps:
      1. Collect point metrics from every (approach, model) pair.
      2. Compute bootstrap 95 % CI on AUC and F1 for each.
      3. Composite score = 0.5 × AUC + 0.5 × F1@optimal.
      4. Run pairwise McNemar between the top-ranked model and every
         other model to test whether the ranking is statistically
         significant.
      5. Print a rich comparison table with CIs and significance flags.

    Returns
    -------
    dict  with 'best_approach', 'best_model', 'best_metrics',
          'comparison' (DataFrame), 'optimal_threshold',
          'bootstrap_cis' (per-model dict), 'mcnemar_results' (DataFrame).
    """
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON  (with bootstrap CIs & McNemar)")
    print("=" * 80 + "\n")

    rows = []
    boot_cis = {}
    all_y_test = None   # shared test set (must be identical across approaches)
    n_full = 0           # fallback test-set size (updated once all_y_test is set)

    for res in result_dicts:
        approach = res.get("approach_name", "Unknown")
        y_te = res.get("y_test")
        if y_te is not None:
            all_y_test = y_te

        # Fall back to shared test set when this dict has no y_test
        _y = y_te if y_te is not None else all_y_test
        if all_y_test is not None:
            n_full = len(np.asarray(all_y_test))

        if "ensemble_model" in res:
            # Ensemble row (single model)
            y_proba = res.get("y_pred_proba")
            ci = bootstrap_model_metrics(_y, y_proba, n_boot=n_boot) if (y_proba is not None and _y is not None) else None
            boot_cis[f"{approach}|{res['ensemble_model']}"] = ci
            rows.append({
                "Approach": approach, "Model": res["ensemble_model"],
                "ROC-AUC": res["roc_auc"], "PR-AUC": res.get("pr_auc", np.nan),
                "F1-Score": res["f1_score"],
                "Recall": res["recall"], "Precision": res["precision"],
                "Accuracy": res["accuracy"],
                "Opt-Thr": res.get("optimal_threshold", 0.5),
                "F1-Opt": res.get("f1_at_optimal", res["f1_score"]),
                "AUC-CI": f"[{ci['auc']['ci_lo']:.3f}, {ci['auc']['ci_hi']:.3f}]" if ci else "",
                "F1-CI": f"[{ci['f1']['ci_lo']:.3f}, {ci['f1']['ci_hi']:.3f}]" if ci else "",
                "PR-AUC-CI": f"[{ci['pr_auc']['ci_lo']:.3f}, {ci['pr_auc']['ci_hi']:.3f}]" if ci and 'pr_auc' in ci else "",
                "Degenerate": False,
                "N-Test": len(np.asarray(_y)) if _y is not None else n_full,
            })
        elif "results" in res:
            for name, m in res["results"].items():
                y_proba = m.get("y_pred_proba")
                # Use per-model y_test when available (subsampled models like LLM-Embed)
                _y_m = m.get("y_test_eval")
                if _y_m is None:
                    # Cached results may lack y_test_eval — fall back only if sizes match
                    _y_m = _y if (y_proba is None or _y is None or len(np.asarray(y_proba)) == len(np.asarray(_y))) else None
                ci = bootstrap_model_metrics(_y_m, y_proba, n_boot=n_boot) if (y_proba is not None and _y_m is not None) else None
                boot_cis[f"{approach}|{name}"] = ci
                rows.append({
                    "Approach": approach, "Model": name,
                    "ROC-AUC": m["roc_auc"], "PR-AUC": m.get("pr_auc", np.nan),
                    "F1-Score": m["f1_score"],
                    "Recall": m["recall"], "Precision": m["precision"],
                    "Accuracy": m["accuracy"],
                    "Opt-Thr": m.get("optimal_threshold", 0.5),
                    "F1-Opt": m.get("f1_at_optimal", m["f1_score"]),
                    "AUC-CI": f"[{ci['auc']['ci_lo']:.3f}, {ci['auc']['ci_hi']:.3f}]" if ci else "",
                    "F1-CI": f"[{ci['f1']['ci_lo']:.3f}, {ci['f1']['ci_hi']:.3f}]" if ci else "",
                    "PR-AUC-CI": f"[{ci['pr_auc']['ci_lo']:.3f}, {ci['pr_auc']['ci_hi']:.3f}]" if ci and 'pr_auc' in ci else "",
                    "Degenerate": m.get("degenerate", False),
                    "N-Test": m.get("n_test_samples", len(np.asarray(_y_m)) if _y_m is not None else n_full),
                })

    df = pd.DataFrame(rows)
    # Composite: 50% AUC + 50% F1 at optimal threshold
    df["Composite"] = 0.5 * df["ROC-AUC"] + 0.5 * df["F1-Opt"]
    df = df.sort_values("Composite", ascending=False).reset_index(drop=True)

    # ── Exclude degenerate / small-sample models from WINNER ─
    # They stay in the table for comparison but cannot be crowned best.
    _min_test = getattr(config, "MIN_TEST_SAMPLES_FOR_BEST", 500)
    _eligible = df[
        (~df["Degenerate"]) & (df["N-Test"] >= _min_test)
    ] if "Degenerate" in df.columns else df
    if _eligible.empty:
        _eligible = df  # fallback: keep all if none eligible
    df = pd.concat([_eligible, df[~df.index.isin(_eligible.index)]]).reset_index(drop=True)

    print("── Ranked models with 95 % bootstrap CIs ──\n")
    show_cols = ["Approach", "Model", "ROC-AUC", "AUC-CI",
                 "PR-AUC", "PR-AUC-CI", "F1-Opt", "F1-CI", "Composite"]
    print(df[show_cols].to_string(index=False))

    # ── McNemar pairwise: winner vs every other ──────────────
    mcnemar_df = pd.DataFrame()
    best = df.iloc[0]
    best_key = f"{best['Approach']}|{best['Model']}"

    n_full = len(all_y_test) if all_y_test is not None else 0
    if all_y_test is not None:
        # Collect y_pred from each result dict (only full-test-set models)
        all_preds = {}
        for res in result_dicts:
            approach = res.get("approach_name", "Unknown")
            if "ensemble_model" in res:
                y_proba = res.get("y_pred_proba")
                if y_proba is not None and len(y_proba) == n_full:
                    all_preds[f"{approach}|{res['ensemble_model']}"] = (y_proba >= 0.5).astype(int)
            elif "results" in res:
                for name, m in res["results"].items():
                    yp = m.get("y_pred")
                    if yp is not None and len(yp) == n_full:
                        all_preds[f"{approach}|{name}"] = np.asarray(yp)

        if best_key in all_preds:
            mc_rows = []
            best_pred = all_preds[best_key]
            for key, other_pred in all_preds.items():
                if key == best_key:
                    continue
                mc = mcnemar_test(all_y_test, best_pred, other_pred)
                mc_rows.append({
                    "Winner vs": key.split("|", 1)[1],
                    "Approach": key.split("|", 1)[0],
                    "b (W✗ O✓)": mc["b"], "c (W✓ O✗)": mc["c"],
                    "χ²": mc["statistic"], "p-value": mc["p_value"],
                    "sig (α=.05)": "✓" if mc["significant"] else "—",
                })
            mcnemar_df = pd.DataFrame(mc_rows)
            if not mcnemar_df.empty:
                print("\n── McNemar's test: winner vs each model ──\n")
                print(mcnemar_df.to_string(index=False))
                n_sig = mcnemar_df["sig (α=.05)"].eq("✓").sum()
                n_total = len(mcnemar_df)
                print(f"\n  Significantly different from winner: {n_sig}/{n_total}")

    opt_thr = float(best.get("Opt-Thr", 0.5))
    best_ci = boot_cis.get(best_key)
    ci_str = ""
    if best_ci:
        ci_str = (f", AUC 95% CI [{best_ci['auc']['ci_lo']:.3f}, "
                  f"{best_ci['auc']['ci_hi']:.3f}]")

    print(f"\nWINNER: {best['Approach']} - {best['Model']} "
          f"(AUC: {best['ROC-AUC']:.4f}, PR-AUC: {best.get('PR-AUC', 0):.4f}{ci_str}, "
          f"F1@opt: {best['F1-Opt']:.4f}, threshold: {opt_thr:.3f})")

    return {
        "best_approach": best["Approach"], "best_model": best["Model"],
        "best_metrics": dict(best), "comparison": df,
        "optimal_threshold": opt_thr,
        "bootstrap_cis": boot_cis,
        "mcnemar_results": mcnemar_df,
    }