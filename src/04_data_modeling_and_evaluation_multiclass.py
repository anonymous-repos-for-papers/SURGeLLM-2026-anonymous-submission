# ============================================================
# 04_data_modeling_and_evaluation_multiclass.py – Multiclass retrofit-type prediction
#
# Structure
# ---------
#   Block A — Models:        Base model factory, evaluation helpers,
#                             baseline training (Stacking)
#   Block B — Optimization:  Focal loss, Optuna TPE tuning (+ joint
#                             SMOTE ratio selection)
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               ExtraTreesClassifier,
                               AdaBoostClassifier,
                               StackingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
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
select_features = _fe.select_features
prepare_multiclass = _fe.prepare_multiclass
_smoothed_target_encode = _fe._smoothed_target_encode
_frequency_encode = _fe._frequency_encode

# FM / LLM availability (graceful degradation if boto3 is missing)
try:
    import boto3 as _boto3_mod
    _BEDROCK = True
except Exception:
    _BEDROCK = False


# Target column for multiclass (Source A retrofit type)
MULTICLASS_TARGET = config.COL_MULTICLASS_TARGET


# ════════════════════════════════════════════════════════════
# BLOCK A — MODELS (baseline definitions and training)
# ════════════════════════════════════════════════════════════

# ── Focal Loss for XGBoost and LightGBM ─────────────────────
# γ controls how much easy examples are down-weighted.
# α is handled by class_weight / SMOTE; focal loss adds the (1-p)^γ factor.
_FOCAL_GAMMA = config.FOCAL_GAMMA


def _focal_loss_xgb(y_true, y_pred_raw, n_classes, gamma=_FOCAL_GAMMA):
    """Focal loss gradient & hessian for XGBoost multiclass (softprob)."""
    y_pred_raw = y_pred_raw.reshape(-1, n_classes)
    # softmax
    exp_p = np.exp(y_pred_raw - y_pred_raw.max(axis=1, keepdims=True))
    p = exp_p / exp_p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-7, 1 - 1e-7)

    # one-hot
    y_oh = np.zeros_like(p)
    y_oh[np.arange(len(y_true)), y_true.astype(int)] = 1.0

    # focal weight: (1 - p_t)^gamma
    p_t = (p * y_oh).sum(axis=1, keepdims=True)
    focal_w = (1 - p_t) ** gamma

    grad = focal_w * (p - y_oh)
    hess = focal_w * p * (1 - p)  # approximate Hessian
    return grad.ravel(), hess.ravel()


def _focal_loss_lgbm(y_true, y_pred_raw, n_classes, gamma=_FOCAL_GAMMA):
    """Focal loss gradient & hessian for LightGBM multiclass."""
    y_pred_raw = y_pred_raw.reshape(-1, n_classes)
    exp_p = np.exp(y_pred_raw - y_pred_raw.max(axis=1, keepdims=True))
    p = exp_p / exp_p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-7, 1 - 1e-7)

    y_oh = np.zeros_like(p)
    y_oh[np.arange(len(y_true)), y_true.astype(int)] = 1.0

    p_t = (p * y_oh).sum(axis=1, keepdims=True)
    focal_w = (1 - p_t) ** gamma

    grad = focal_w * (p - y_oh)
    hess = focal_w * p * (1 - p)
    return grad.ravel(), hess.ravel()


# ════════════════════════════════════════════════════════════
# BLOCK B — OPTIMIZATION (Optuna, SMOTE, focal loss)
# ════════════════════════════════════════════════════════════

# ---- B.1 Optuna HP builder ----

# ── Optuna Bayesian hyperparameter tuning (multiclass) ───────

def _mc_build_model(trial, name, n_classes):
    """Suggest hyperparameters for *name* via Optuna trial and return model."""
    RS = config.RANDOM_STATE

    if name == "Logistic Regression":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        cw = trial.suggest_categorical("cw", ["balanced", "none"])
        return LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000,
            class_weight=(cw if cw != "none" else None),
            random_state=RS, n_jobs=2)

    if name == "Random Forest":
        cw = trial.suggest_categorical("cw", ["balanced", "none"])
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight=(cw if cw != "none" else None),
            random_state=RS, n_jobs=2)

    if name == "Extra Trees":
        cw = trial.suggest_categorical("cw", ["balanced", "none"])
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight=(cw if cw != "none" else None),
            random_state=RS, n_jobs=2)

    if name == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 1.0, log=True),
            random_state=RS)

    if name == "XGBoost":
        use_focal = trial.suggest_categorical("focal", [True, False])
        if use_focal:
            _nc = n_classes
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                num_class=n_classes,
                objective=lambda y, p: _focal_loss_xgb(y, p, _nc),
                disable_default_eval_metric=True, tree_method="hist",
                random_state=RS, n_jobs=2)
        return xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            num_class=n_classes, objective="multi:softmax",
            eval_metric="mlogloss", tree_method="hist",
            random_state=RS, n_jobs=2)

    if name == "LightGBM":
        cw = trial.suggest_categorical("cw", ["balanced", "none"])
        use_focal = trial.suggest_categorical("focal", [True, False])
        if use_focal:
            _nc = n_classes
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                objective=lambda y, p: _focal_loss_lgbm(y, p, _nc),
                class_weight=(cw if cw != "none" else None),
                random_state=RS, verbose=-1, n_jobs=2)
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            class_weight=(cw if cw != "none" else None),
            random_state=RS, verbose=-1, n_jobs=2)

    if name == "CatBoost":
        acw = trial.suggest_categorical("acw", ["Balanced", "None"])
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            bootstrap_type="Bernoulli",
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            auto_class_weights=(acw if acw != "None" else None),
            random_seed=RS, verbose=0, thread_count=2,
            train_dir=config.CATBOOST_TRAIN_DIR)

    if name == "SVC":
        C = trial.suggest_float("svc_C", 0.01, 100.0, log=True)
        gamma = trial.suggest_categorical("svc_gamma", ["scale", "auto"])
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=C, gamma=gamma, kernel="rbf",
                        probability=True, class_weight="balanced",
                        decision_function_shape="ovr", random_state=RS)),
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


def _optuna_tune_multiclass(X_tr, y_tr, n_classes, smote_k, can_smote):
    """
    Run Optuna TPE Bayesian optimisation for each multiclass model.

    SMOTE is applied **inside** each CV fold via imblearn.Pipeline
    to prevent synthetic-sample leakage across folds.

    Returns
    -------
    dict[str, (estimator, str)]  – model name → (fitted model, smote_key)
    """
    warnings.filterwarnings("ignore")
    from imblearn.pipeline import Pipeline as _ImbPipeline
    if not _OPTUNA:
        print("  ⚠ Optuna not installed — using hand-tuned defaults")
        return None

    _SMOTE_RATIOS = ["none", "0.3", "0.5", "0.6", "0.8", "1.0"]
    if not can_smote:
        _SMOTE_RATIOS = ["none"]
    max_class_n = int(y_tr.value_counts().max())

    n_trials = config.OPTUNA_MC_N_TRIALS
    timeout = config.OPTUNA_TIMEOUT
    cv = StratifiedKFold(n_splits=config.OPTUNA_CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    model_names = [n for n in [
        "Logistic Regression", "Random Forest", "Extra Trees",
        "AdaBoost", "XGBoost", "LightGBM", "CatBoost",
        "SVC", "MLP",
    ] if n not in config.OPTUNA_SKIP_MULTICLASS]

    tuned = {}
    for name in model_names:
        print(f"  Optuna tuning: {name} ({n_trials} trials) …", end=" ", flush=True)

        def objective(trial, _name=name):
            try:
                smote_key = trial.suggest_categorical(
                    "smote_ratio", _SMOTE_RATIOS)
                model = _mc_build_model(trial, _name, n_classes)
                # Wrap in imblearn Pipeline so SMOTE runs inside each fold
                if smote_key != "none" and can_smote:
                    ratio = float(smote_key)
                    target_n = int(max_class_n * ratio)
                    _cc = y_tr.value_counts().to_dict()
                    _strat = {c: target_n for c, n in _cc.items()
                              if n < target_n}
                    if _strat:
                        smote_step = SMOTE(
                            sampling_strategy=_strat,
                            k_neighbors=smote_k,
                            random_state=config.RANDOM_STATE,
                        )
                        estimator = _ImbPipeline([("smote", smote_step),
                                                   ("clf", model)])
                    else:
                        estimator = model
                else:
                    estimator = model
                scores = cross_val_score(
                    estimator, X_tr, y_tr, cv=cv,
                    scoring="f1_weighted", n_jobs=1,
                )
                return scores.mean()
            except Exception:
                raise optuna.TrialPruned()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False)

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f"  ⚠ All trials failed for {name} — skipping")
            del study; gc.collect()
            continue
        best = study.best_trial
        best_smote = best.params.get("smote_ratio", "none")
        print(f"F1w={best.value:.4f}  (trial {best.number}, smote={best_smote})")

        # Rebuild model with best hyperparameters
        tuned[name] = (_mc_build_model(best, name, n_classes), best_smote)
        del study; gc.collect()

    return tuned


def multiclass_retrofit_types(df_multi):
    """
    Predict which retrofit type a vehicle will receive (multiclass).

    v2 improvements over v1:
      - Frequency encoding alongside target encoding
      - SMOTE oversampling for minority classes
      - StratifiedKFold 3-fold CV for robust model selection
      - Higher n_estimators + lower learning rates
      - StackingClassifier (LightGBM meta-learner) instead of VotingClassifier

    Parameters
    ----------
    df_multi : DataFrame from merge_inner_join (Source A ⋈ Source B).
               All rows are retrofit vehicles — no filtering needed.
    """
    warnings.filterwarnings("ignore")
    print("\n" + "=" * 80)
    print("MULTICLASS CLASSIFICATION - Predict Retrofit Type (v3 + Optuna)")
    print("=" * 80 + "\n")

    # ── Feature engineering (delegated to 02_feature_engineering Section 7) ──
    dp = prepare_multiclass(df_multi)
    if dp is None:
        return None

    X_tr_f   = dp["X_train"]
    X_te_f   = dp["X_test"]
    y_tr     = dp["y_train"]
    y_te     = dp["y_test"]
    y_tr_num = dp["y_train_num"]
    y_te_num = dp["y_test_num"]
    le       = dp["label_encoder"]
    cat_cols = dp["categorical_cols_clean"]
    num_cols = dp["num_cols"]
    enc_map  = dp["encoding_map"]
    g_mean   = dp["global_mean"]
    freq_map = dp["freq_map"]
    n_classes = dp["n_classes"]
    common    = dp["class_names"]
    _top_feats = dp["feature_cols"]

    # ── SMOTE config (applied inside CV folds, not pre-computed) ─
    class_counts = y_tr_num.value_counts()
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    smote_k = min(config.SMOTE_K_NEIGHBORS, min_class_count - 1)
    can_smote = smote_k >= 1
    if not can_smote:
        print("  SMOTE: skipped (minority class too small)")

    # ── Optuna Bayesian hyperparameter tuning ──────────────────
    tuned_models = _optuna_tune_multiclass(
        X_tr_f, y_tr_num, n_classes, smote_k, can_smote)

    if tuned_models is not None:
        # tuned_models is dict[name -> (model, smote_key)]
        models = {n: m for n, (m, _) in tuned_models.items()}
        smote_choice = {n: s for n, (_, s) in tuned_models.items()}
    else:
        # Fallback: hand-tuned defaults (when Optuna is not available)
        smote_choice = {}
        models = {
            "Logistic Regression": LogisticRegression(
                solver="lbfgs", C=1.0,
                max_iter=1000, class_weight="balanced",
                random_state=config.RANDOM_STATE, n_jobs=2),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=20,
                min_samples_leaf=2, min_samples_split=5,
                class_weight="balanced",
                random_state=config.RANDOM_STATE, n_jobs=2),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, max_depth=20,
                min_samples_leaf=2, min_samples_split=5,
                class_weight="balanced",
                random_state=config.RANDOM_STATE, n_jobs=2),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.08, max_depth=6,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
                num_class=n_classes, objective="multi:softmax",
                eval_metric="mlogloss", tree_method="hist",
                random_state=config.RANDOM_STATE, n_jobs=2),
            "LightGBM": LGBMClassifier(
                n_estimators=300, learning_rate=0.08, max_depth=-1,
                num_leaves=63, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
                random_state=config.RANDOM_STATE, verbose=-1, n_jobs=2),
            "CatBoost": CatBoostClassifier(
                iterations=300, learning_rate=0.08, depth=6,
                l2_leaf_reg=3.0, subsample=0.8,
                bootstrap_type="Bernoulli",
                random_seed=config.RANDOM_STATE, verbose=0,
                thread_count=2, train_dir=config.CATBOOST_TRAIN_DIR),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=200, learning_rate=0.08,
                random_state=config.RANDOM_STATE),
            "SVC": Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", probability=True,
                            class_weight="balanced", decision_function_shape="ovr",
                            random_state=config.RANDOM_STATE)),
            ]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=300,
                    early_stopping=True, validation_fraction=0.1,
                    random_state=config.RANDOM_STATE)),
            ]),
        }

    # ── Helper: apply SMOTE once on full training set for final refit ──
    def _smote_refit(X, y, ratio_key):
        """Return (X_sm, y_sm) for final refit with the chosen SMOTE ratio."""
        if ratio_key == "none" or not can_smote:
            return X, y
        ratio = float(ratio_key)
        target_n = int(max_class_count * ratio)
        _cc = y.value_counts().to_dict()
        _strat = {c: target_n for c, n in _cc.items() if n < target_n}
        if not _strat:
            return X, y
        sm = SMOTE(sampling_strategy=_strat, k_neighbors=smote_k,
                   random_state=config.RANDOM_STATE)
        X_sm, y_sm = sm.fit_resample(X, y)
        print(f"    SMOTE {ratio_key}: {len(y):,} → {len(y_sm):,}")
        return X_sm, y_sm

    # ── StratifiedKFold test evaluation (skip redundant CV — Optuna
    #    already did 3-fold CV internally; just fit + test eval) ───
    results = {}

    for name, model in models.items():
        try:
            print(f"Training: {name}")

            # Apply SMOTE on full training set for final refit only
            sm_key = smote_choice.get(name, "0.6")
            _X_fit, _y_fit = _smote_refit(X_tr_f, y_tr_num, sm_key)
            model.fit(_X_fit, _y_fit)
            y_pred = le.inverse_transform(model.predict(X_te_f))
            y_proba = model.predict_proba(X_te_f)
            acc = accuracy_score(y_te, y_pred)
            f1w = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            f1m = f1_score(y_te, y_pred, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_te_num, y_proba,
                                    multi_class="ovr", average="weighted")
            except Exception:
                auc = np.nan
            print(f"  Test  Acc: {acc:.4f} | F1w: {f1w:.4f} | F1m: {f1m:.4f} | AUC: {auc:.4f}\n")
            results[name] = {
                "model": model, "accuracy": acc, "f1_weighted": f1w,
                "f1_macro": f1m, "roc_auc": auc, "y_pred": y_pred,
                "cv_f1_mean": f1w, "cv_f1_std": 0.0,
            }
        except Exception as exc:
            print(f"  ✗ {name} failed: {exc}")
        gc.collect()  # free CV copies between models

    # ── StackingClassifier (LightGBM meta-learner) ───────────
    # CatBoost uses deprecated sklearn internals (force_all_finite)
    # that crash with sklearn ≥ 1.6 inside Stacking/Voting refits.
    _ENSEMBLE_SKIP = {"CatBoost"}
    ranked = sorted(results.items(),
                    key=lambda x: x[1]["cv_f1_mean"], reverse=True)
    top3_names = [r[0] for r in ranked if r[0] not in _ENSEMBLE_SKIP][:3]
    print(f"Stacking base estimators (top-3 by CV, excl. CatBoost): {top3_names}")

    try:
        stack = StackingClassifier(
            estimators=[(n, results[n]["model"]) for n in top3_names],
            final_estimator=LGBMClassifier(
                n_estimators=100, learning_rate=0.08,
                num_leaves=31, verbose=-1,
                random_state=config.RANDOM_STATE, n_jobs=2,
            ),
            cv=config.STACKING_CV_FOLDS, passthrough=False, n_jobs=1,
        )
        # Apply SMOTE on full training set for stacking refit
        _X_stk, _y_stk = _smote_refit(X_tr_f, y_tr_num, "0.6")
        stack.fit(_X_stk, _y_stk)

        y_stk_pred = le.inverse_transform(stack.predict(X_te_f))
        y_stk_proba = stack.predict_proba(X_te_f)
        acc_s = accuracy_score(y_te, y_stk_pred)
        f1w_s = f1_score(y_te, y_stk_pred, average="weighted", zero_division=0)
        f1m_s = f1_score(y_te, y_stk_pred, average="macro", zero_division=0)
        try:
            auc_s = roc_auc_score(y_te_num, y_stk_proba,
                                  multi_class="ovr", average="weighted")
        except Exception:
            auc_s = np.nan
        print(f"  Stacking Acc: {acc_s:.4f} | F1w: {f1w_s:.4f} | F1m: {f1m_s:.4f} | AUC: {auc_s:.4f}\n")
        results["Stacking"] = {
            "model": stack, "accuracy": acc_s, "f1_weighted": f1w_s,
            "f1_macro": f1m_s, "roc_auc": auc_s, "y_pred": y_stk_pred,
            "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
        }
    except Exception as exc:
        print(f"  ✗ Stacking failed: {exc}")

    # VotingClassifier skipped — Stacking already combines top-3 models
    # and typically matches or outperforms soft voting.

    # ── Foundation-model classifiers (Block C) ───────────────
    _run_llm_multiclass(X_tr_f, y_tr_num, smote_k, can_smote,
                        max_class_count, X_te_f, y_te, y_te_num, le, results,
                        X_tr_raw=dp.get("X_train_raw"),
                        X_te_raw=dp.get("X_test_raw"))

    # ── Comparison table ─────────────────────────────────────
    cdf = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [results[m]["accuracy"] for m in results],
        "F1_weighted": [results[m]["f1_weighted"] for m in results],
        "F1_macro": [results[m].get("f1_macro", np.nan) for m in results],
        "ROC_AUC": [results[m]["roc_auc"] for m in results],
        "CV_F1_mean": [results[m].get("cv_f1_mean", np.nan) for m in results],
    }).sort_values("F1_weighted", ascending=False)

    print(cdf.to_string(index=False))
    if cdf.empty:
        print("  ✗ No models succeeded — returning empty results")
        return {
            "results": results, "comparison": cdf, "best_model": None,
            "y_test": y_te, "min_support_sensitivity": None,
            "agreement_matrix": None, "bootstrap_ci": None,
            "label_encoder": le, "encoding_map": enc_map,
            "global_mean": g_mean, "freq_map": freq_map,
            "cat_cols": cat_cols, "num_cols": num_cols,
            "selected_features": _top_feats,
        }
    best = cdf.iloc[0]["Model"]
    print(f"\nBest: {best}")
    _best_yt = results[best].get("y_test_eval", y_te)
    print(f"\n{classification_report(_best_yt, results[best]['y_pred'])}")

    # ── Min-support sensitivity analysis ─────────────────────
    # Show how metrics change when excluding classes with few
    # training samples.  Maps each test label back to its training
    # count, then filters by increasing thresholds.
    _train_counts = pd.Series(y_tr).value_counts().to_dict()
    _test_support = np.array([_train_counts.get(t, 0) for t in _best_yt])
    _thresholds = [1, 2, 3, 5, 10, 20, 50]
    _best_yp = results[best]["y_pred"]
    print(f"\n  Min-support sensitivity (best = {best}):")
    print(f"    {'Min-N':>6s}  {'N-test':>7s}  {'F1w':>6s}  {'F1m':>6s}  {'Acc':>6s}")
    _sens_rows = []
    for thr in _thresholds:
        mask = _test_support >= thr
        n_keep = int(mask.sum())
        if n_keep < 2:
            break
        _yt = np.asarray(_best_yt)[mask]
        _yp = np.asarray(_best_yp)[mask]
        _f1w = f1_score(_yt, _yp, average="weighted", zero_division=0)
        _f1m = f1_score(_yt, _yp, average="macro", zero_division=0)
        _acc = accuracy_score(_yt, _yp)
        print(f"    {thr:6d}  {n_keep:7d}  {_f1w:.4f}  {_f1m:.4f}  {_acc:.4f}")
        _sens_rows.append({"min_support": thr, "n_test": n_keep,
                           "f1_weighted": _f1w, "f1_macro": _f1m,
                           "accuracy": _acc})
    _sensitivity_df = pd.DataFrame(_sens_rows) if _sens_rows else None

    # ── Agreement matrix between models ──────────────────────
    # Only compare models whose prediction arrays have the same length
    # (LLM models use a subsampled test set, so they differ from ML models).
    _full_n = len(y_te)
    _full_models = [m for m in results if len(results[m]["y_pred"]) == _full_n]
    _preds = {m: np.asarray(results[m]["y_pred"]) for m in _full_models}
    _n_models = len(_full_models)
    _agree_mat = np.zeros((_n_models, _n_models))
    for i in range(_n_models):
        for j in range(_n_models):
            _agree_mat[i, j] = float((_preds[_full_models[i]] == _preds[_full_models[j]]).mean())
    _agreement_df = pd.DataFrame(_agree_mat, index=_full_models, columns=_full_models)
    print(f"\n  Pairwise prediction agreement (fraction, {_n_models} full-test models):")
    print(_agreement_df.round(3).to_string())

    # ── Bootstrap confidence intervals (best model) ──────────
    _best_yp = results[best]["y_pred"]
    _n_boot = 1000
    _boot_f1w, _boot_f1m, _boot_acc = [], [], []
    rng = np.random.RandomState(config.RANDOM_STATE)
    _yt_arr = np.asarray(_best_yt)
    _yp_arr = np.asarray(_best_yp)
    for _ in range(_n_boot):
        idx = rng.choice(len(_yt_arr), len(_yt_arr), replace=True)
        _yt_b = _yt_arr[idx]
        _yp_b = _yp_arr[idx]
        _boot_f1w.append(f1_score(_yt_b, _yp_b, average="weighted", zero_division=0))
        _boot_f1m.append(f1_score(_yt_b, _yp_b, average="macro", zero_division=0))
        _boot_acc.append(accuracy_score(_yt_b, _yp_b))
    _ci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    _f1w_ci = _ci(_boot_f1w)
    _f1m_ci = _ci(_boot_f1m)
    _acc_ci = _ci(_boot_acc)
    print(f"\n  Bootstrap 95% CI (best = {best}, n_boot={_n_boot}):")
    print(f"    F1-weighted : [{_f1w_ci[0]:.4f}, {_f1w_ci[1]:.4f}]")
    print(f"    F1-macro    : [{_f1m_ci[0]:.4f}, {_f1m_ci[1]:.4f}]")
    print(f"    Accuracy    : [{_acc_ci[0]:.4f}, {_acc_ci[1]:.4f}]")
    _bootstrap_ci = {
        "f1_weighted": _f1w_ci, "f1_macro": _f1m_ci, "accuracy": _acc_ci,
    }

    gc.collect()
    return {
        "results": results,
        "comparison": cdf,
        "best_model": best,
        "y_test": y_te,
        "min_support_sensitivity": _sensitivity_df,
        "agreement_matrix": _agreement_df,
        "bootstrap_ci": _bootstrap_ci,
        "label_encoder": le,
        "encoding_map": enc_map,
        "global_mean": g_mean,
        "freq_map": freq_map,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "selected_features": _top_feats,
    }

# ══════════════════════════════════════════════════════════════════
# BLOCK C — FOUNDATION-MODEL CLASSIFIERS (LLM-Embed + LLM-Prompted)
# ══════════════════════════════════════════════════════════════════

# ── LLM Embedding classifier (Bedrock Titan embeddings + LogReg) ─

class _LLMEmbeddingClassifier(BaseEstimator, ClassifierMixin):
    """Bedrock Titan embeddings → LogReg for multiclass classification."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._logreg = None
        self._scaler = None
        self._col_names = None

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
        done      = [0]

        def _call(idx_text):
            idx, text = idx_text
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
            print(f"    [LLM-Embed] subsampled {len(X):,} → {_max:,} train rows")
        texts = self._rows_to_texts(X_arr, self._col_names)
        print(f"    [LLM-Embed] embedding {len(texts):,} train rows …", flush=True)
        emb = self._embed(texts)

        # Store for RAG reuse by _LLMPromptedClassifier
        self._train_embeddings = emb
        self._train_texts = texts
        self._train_labels = y_arr

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
    """Sklearn-compatible multiclass classifier: Bedrock Converse with
    domain-specific prompts, statistical context, and few-shot examples.
    """

    # ── Feature descriptions (example, extend as needed)
    _FEATURE_DESCRIPTIONS = {
        config.COL_DERIVAT_CLEAN: "Vehicle derivative / model variant",
        "basistyp_clean": "Base vehicle type (platform)",
        "antriebsart_clean": "Powertrain type (diesel, petrol, hybrid, electric)",
        "motortyp_clean": "Engine type code",
        "basisausfuehrung_clean": "Base trim / equipment level",
        "pep_bauphase_clean": "PEP build phase (development stage of the model)",
        "owner_fleet_clean": "Owner / fleet operator",
        "hybrid_clean": "Hybrid flag (yes/no → encoded)",
        "prio_clean": "Retrofit priority level assigned by planning",
        "umr_paket_name_clean": "Retrofit package name / campaign",
        "werkstattgruppe_clean": "Workshop group (region / capacity cluster)",
        "werkstattname_clean": "Workshop name (specific service location)",
        "laendervariante_clean": "Country variant of the vehicle",
        "vehicle_age_days": "Vehicle age in days since first registration",
        "days_since_sop": "Days since Start of Production (SOP) of this model",
        "days_to_deadline": "Days remaining until the regulatory retrofit deadline",
        "month_sin": "Sine of calendar month (seasonality feature)",
        "month_cos": "Cosine of calendar month (seasonality feature)",
        "year": "Calendar year of the observation",
    }

    # ── Domain context for the system prompt ────────────────
    _DOMAIN_CONTEXT_MULTICLASS = (
        "DOMAIN BACKGROUND — Vehicle Retrofits\n"
        "A retrofit is a mandatory post-sale modification to vehicles\n"
        "already in the field, triggered by regulatory, safety, or quality reasons.\n"
        "The OEM tracks these through two main systems:\n"
        "  • Source A — planning system that defines retrofit campaigns and deadlines.\n"
        "  • Source B — dealer workshop system that records which vehicles actually\n"
        "    came in and had the retrofit performed.\n\n"
        "There are several main retrofit types (software updates, hardware replacements,\n"
        "mechatronic actions, type-approval retrofits, etc.).\n\n"
        "Key domain knowledge for TYPE prediction:\n"
        "  • The retrofit package name (umr_paket_name_clean) is the strongest\n"
        "    indicator of retrofit type — each campaign targets a specific type.\n"
        "  • Powertrain type (antriebsart_clean) correlates with retrofit type:\n"
        "    diesel vehicles are more likely to need emission-related retrofits,\n"
        "    while hybrids/EVs tend toward software updates.\n"
        "  • Vehicle age and platform (basistyp_clean) determine which hardware\n"
        "    or mechatronic components are affected.\n"
        "  • Priority level (prio_clean) differs by type: high-priority mechatronic\n"
        "    actions are always urgent, while software updates are typically lower.\n"
        "  • Seasonal patterns and deadline proximity affect WHEN vehicles come,\n"
        "    but the retrofit TYPE is determined by the campaign / package.\n"
    )

    def __init__(self, random_state=42, embed_clf=None):
        self.random_state = random_state
        self.embed_clf = embed_clf   # trained _LLMEmbeddingClassifier for RAG
        self._col_names = None
        self._top_feature_idx = None # indices of selected features (or None = all)
        self._few_shot_examples = {}  # dict[class_label] -> list[texts]
        self._class_stats = ""       # statistical summary string
        self._feature_desc = ""      # feature description block
        self.classes_ = None

    # ── Helpers ──────────────────────────────

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
        lines = [f"Training set: {len(df):,} rows"]
        labels = np.unique(y_arr)
        for lbl in labels:
            subset = df[df["_label"] == lbl].drop(columns="_label")
            lines.append(f"\nClass {lbl}: {len(subset):,} examples")
            lines.append("Feature summaries:")
            for c in col_names:
                if df[c].dtype == object:
                    mode_val = subset[c].mode().iloc[0] if len(subset) and len(subset[c].mode()) else "?"
                    lines.append(f"  {c}: {mode_val} (most common)")
                else:
                    lines.append(f"  {c}: {subset[c].mean():.3f}")
        return "\n".join(lines)

    def _select_few_shot(self, X_arr, y_arr, col_names, k=None):
        if k is None:
            from . import config
            k = getattr(config, "LLM_PROMPTED_FEW_SHOT_K", 3)
        rng = np.random.RandomState(self.random_state)
        self._few_shot_examples = {}
        for lbl in np.unique(y_arr):
            idxs = np.where(y_arr == lbl)[0]
            rng.shuffle(idxs)
            texts = [self._row_to_text(X_arr[i], col_names) for i in idxs[:k]]
            self._few_shot_examples[lbl] = texts

    def _select_few_shot_rag(self, row_text, k=None):
        """RAG few-shot: pick K nearest training examples per class via
        cosine similarity on Titan embeddings stored in embed_clf.

        Returns dict[class_label] → list[texts], falling back to the
        static few-shot examples if embed_clf is unavailable.
        """
        if k is None:
            k = getattr(config, "LLM_PROMPTED_FEW_SHOT_K", 3)
        ec = self.embed_clf
        if ec is None or not hasattr(ec, "_train_embeddings") or ec._train_embeddings is None:
            return dict(self._few_shot_examples)

        from sklearn.metrics.pairwise import cosine_similarity

        query_emb = ec._embed([row_text])                # (1, d)
        sims = cosine_similarity(query_emb, ec._train_embeddings)[0]  # (N,)

        labels = ec._train_labels
        texts  = ec._train_texts

        result = {}
        for lbl in np.unique(labels):
            mask = (labels == lbl)
            lbl_sims = np.where(mask, sims, -2.0)
            top_idx = np.argsort(lbl_sims)[-k:][::-1]
            result[lbl] = [self._row_to_text_from_str(texts[i]) for i in top_idx]
        return result

    def _select_top_features(self, X_arr, y_arr, col_names, top_k=None):
        """Pick top-K features by mutual information.

        String (object) columns are always kept; remaining budget is
        filled with the highest-MI numeric columns.

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

    def _build_prompt(self, row_text, target_classes, few_shot_override=None):
        """Assemble the full prompt for one vehicle.

        When *few_shot_override* is supplied (RAG per-row), it overrides
        the static examples stored during fit().
        """
        fs_data = few_shot_override if few_shot_override is not None else self._few_shot_examples
        fs_lines = []
        ex_num = 0
        for lbl, examples in fs_data.items():
            for txt in examples:
                ex_num += 1
                fs_lines.append(f"Example {ex_num} (class={lbl}):\n{txt}")
        few_shot_block = "\n\n".join(fs_lines)

        class_list = ", ".join(str(c) for c in target_classes)
        example_probs = ", ".join(f'"{c}": 0.0' for c in target_classes)

        prompt = (
            "You are an expert vehicle-retrofit analyst at an automotive OEM.\n"
            "Your task: predict the retrofit type of a vehicle.\n"
            f"Valid classes: [{class_list}]\n\n"
            "=== DOMAIN CONTEXT ===\n"
            f"{self._DOMAIN_CONTEXT_MULTICLASS}\n\n"
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
            "  3. Consider domain knowledge (package name, powertrain, platform,\n"
            "     retrofit priority, deadline proximity).\n"
            "  4. Arrive at a class probability distribution and final prediction.\n\n"
            "Return ONLY valid JSON (no markdown, no text outside the JSON):\n"
            '{"reasoning": "<step-by-step analysis, 3-5 sentences>", '
            '"predictions": {' + example_probs + '}, '
            '"prediction": <int — one of the valid classes>, '
            '"explanation": "<1-sentence summary of key drivers>"}\n'
        )
        return prompt

    def _call_bedrock(self, prompt, temperature=None):
        import boto3, json
        _temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)
        resp = client.converse(
            modelId=config.LLM_CHAT_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": _temp, "maxTokens": 1024},
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
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    def _predict_batch(self, X):
        """Concurrent prediction for a batch of rows."""
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
                fs_override = self._select_few_shot_rag(row_semi)
                prompt = self._build_prompt(row_text, self.classes_, fs_override)
            else:
                prompt = self._build_prompt(row_text, self.classes_)
            try:
                if _sc_n <= 1:
                    parsed = self._call_bedrock(prompt)
                else:
                    # Self-consistency: N calls with higher temp → majority vote
                    votes = []
                    for _ in range(_sc_n):
                        votes.append(self._call_bedrock(prompt, temperature=_sc_temp))
                    preds = [v.get("prediction", self.classes_[0]) for v in votes]
                    majority = max(set(preds), key=preds.count)
                    # Average probability distributions
                    avg_preds = {}
                    for c_label in self.classes_:
                        c_str = str(c_label)
                        vals = [float(v.get("predictions", {}).get(c_str, 1.0/len(self.classes_))) for v in votes]
                        avg_preds[c_str] = float(np.mean(vals))
                    parsed = {
                        "predictions": avg_preds,
                        "prediction": majority,
                        "explanation": votes[0].get("explanation", ""),
                        "reasoning": votes[0].get("reasoning", ""),
                    }
                results[idx] = parsed
            except Exception as exc:
                results[idx] = {
                    "predictions": {str(c): 1.0/len(self.classes_) for c in self.classes_},
                    "prediction": self.classes_[0],
                    "explanation": f"Parse error: {exc}"
                }
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == n:
                print(f"      [LLM-Prompted] {done[0]}/{n} rows classified", flush=True)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_call_one, range(n)))
        return results

    # ── Sklearn interface ─────────────────────

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self._col_names = X.columns.tolist()
            # Preserve mixed types (strings + numerics) for the LLM prompt
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            self._col_names = self._col_names or [f"f{i}" for i in range(X_arr.shape[1])]
        y_arr = np.asarray(y)

        # Subsample for stats
        _max = getattr(config, "LLM_EMBED_MAX_SAMPLES", 5_000)
        if len(X_arr) > _max:
            from sklearn.model_selection import train_test_split as _tts
            X_arr, _, y_arr, _ = _tts(X_arr, y_arr, train_size=_max,
                                      random_state=self.random_state, stratify=y_arr)
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
        self._select_few_shot(X_sel, y_arr, sel_cols)

        # Store selected column names for prediction-time filtering
        self._col_names = sel_cols

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        k = len(next(iter(self._few_shot_examples.values()), []))
        print(f"    [LLM-Prompted] fit complete: {len(sel_cols)} features, "
              f"{k} few-shot examples × {n_classes} classes")
        return self

    def predict(self, X):
        results = self._predict_batch(X)
        preds = np.array([r.get("prediction", self.classes_[0]) for r in results], dtype=int)
        return preds

    def predict_proba(self, X):
        results = self._predict_batch(X)
        probs = []
        for r in results:
            pred_dict = r.get("predictions", {})
            row_probs = [float(pred_dict.get(str(c), 0.0)) for c in self.classes_]
            row_sum = sum(row_probs)
            if row_sum > 0:
                row_probs = [v/row_sum for v in row_probs]
            else:
                row_probs = [1.0/len(self.classes_)]*len(self.classes_)
            probs.append(row_probs)
        return np.array(probs, dtype=np.float64)

    def predict_with_explanations(self, X):
        results = self._predict_batch(X)
        probs = []
        explanations = []
        for r in results:
            pred_dict = r.get("predictions", {})
            row_probs = [float(pred_dict.get(str(c), 0.0)) for c in self.classes_]
            row_sum = sum(row_probs)
            if row_sum > 0:
                row_probs = [v/row_sum for v in row_probs]
            else:
                row_probs = [1.0/len(self.classes_)]*len(self.classes_)
            probs.append(row_probs)
            explanations.append(r.get("explanation", ""))
        return np.array(probs, dtype=np.float64), explanations

    def __reduce__(self):
        from . import utils
        return (utils._reconstruct_clf,
                (self.__class__.__module__, type(self).__name__,
                 self.__dict__.copy()))

    # get_params / set_params inherited from BaseEstimator


# ── C helper: train & evaluate both LLM classifiers ─────────

def _run_llm_multiclass(X_tr, y_tr, smote_k, can_smote, max_class_count,
                        X_te, y_te, y_te_num, le, results,
                        X_tr_raw=None, X_te_raw=None):
    """Train & evaluate both FM classifiers for multiclass, appending to *results*.

    Handles _BEDROCK guard, SMOTE refit, test-set subsampling,
    label decoding, and graceful failure.

    If *X_tr_raw* / *X_te_raw* are provided (pre-encoding DataFrames with
    original text categories), LLM classifiers see those instead of the
    target-encoded floats.
    """
    _X_tr = X_tr_raw if X_tr_raw is not None else X_tr
    _X_te = X_te_raw if X_te_raw is not None else X_te
    if not _BEDROCK:
        return

    # SMOTE requires numeric data; raw DataFrames may contain string
    # categorical columns (e.g. derivat_clean='G60').  LLM classifiers
    # convert rows to text anyway, so skip SMOTE when strings are present.
    _has_strings = (isinstance(_X_tr, pd.DataFrame)
                    and _X_tr.select_dtypes(include="object").shape[1] > 0)

    def _smote_once(X, y, ratio=0.6):
        if _has_strings or not can_smote:
            return X, y
        target_n = int(max_class_count * ratio)
        _cc = y.value_counts().to_dict()
        _strat = {c: target_n for c, n in _cc.items() if n < target_n}
        if not _strat:
            return X, y
        sm = SMOTE(sampling_strategy=_strat, k_neighbors=smote_k,
                   random_state=config.RANDOM_STATE)
        return sm.fit_resample(X, y)

    # ── LLM-Embed (Bedrock Titan embeddings → LogReg) ────────
    print(f"\n  LLM-Embed LogReg (Bedrock {config.LLM_EMBEDDING_MODEL}) …")
    try:
        llm_clf = _LLMEmbeddingClassifier(random_state=config.RANDOM_STATE)
        _X_llm, _y_llm = _smote_once(_X_tr, y_tr)
        llm_clf.fit(_X_llm, _y_llm)
        _mx = getattr(config, "LLM_EMBED_MAX_SAMPLES", 5_000)
        if len(_X_te) > _mx:
            from sklearn.model_selection import train_test_split as _tts
            _X_te_s, _, _y_te_s_num, _ = _tts(
                _X_te, y_te_num, train_size=_mx,
                random_state=config.RANDOM_STATE, stratify=y_te_num)
            _y_te_s = le.inverse_transform(_y_te_s_num)
            print(f"    [LLM-Embed] test subset {len(_X_te):,} → {_mx:,}")
        else:
            _X_te_s, _y_te_s_num, _y_te_s = _X_te, y_te_num, y_te
        y_llm_pred_num = llm_clf.predict(_X_te_s)
        y_llm_pred = le.inverse_transform(y_llm_pred_num)
        y_llm_proba = llm_clf.predict_proba(_X_te_s)
        acc_l = accuracy_score(_y_te_s, y_llm_pred)
        f1w_l = f1_score(_y_te_s, y_llm_pred, average="weighted", zero_division=0)
        f1m_l = f1_score(_y_te_s, y_llm_pred, average="macro", zero_division=0)
        try:
            auc_l = roc_auc_score(_y_te_s_num, y_llm_proba,
                                  multi_class="ovr", average="weighted")
        except Exception:
            auc_l = np.nan
        print(f"  LLM-Embed Acc: {acc_l:.4f} | F1w: {f1w_l:.4f} | F1m: {f1m_l:.4f} | AUC: {auc_l:.4f}")
        results["LLM-Embed LogReg"] = {
            "model": llm_clf, "accuracy": acc_l, "f1_weighted": f1w_l,
            "f1_macro": f1m_l, "roc_auc": auc_l, "y_pred": y_llm_pred,
            "y_test_eval": _y_te_s,
            "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
        }
    except Exception as exc:
        print(f"  ✗ LLM-Embed LogReg failed: {exc}")
    gc.collect()

    # ── LLM-Prompted (Bedrock Converse, domain-guided) ───────
    print(f"\n  LLM-Prompted (Bedrock {config.LLM_CHAT_MODEL}) …")
    try:
        llm_p = _LLMPromptedClassifier(
            random_state=config.RANDOM_STATE,
            embed_clf=llm_clf if 'llm_clf' in dir() else None,
        )
        _X_llmp, _y_llmp = _smote_once(_X_tr, y_tr)
        llm_p.fit(_X_llmp, _y_llmp)
        _mx_p = getattr(config, "LLM_PROMPTED_MAX_SAMPLES", 200)
        if len(_X_te) > _mx_p:
            from sklearn.model_selection import train_test_split as _tts
            _X_te_p, _, _y_te_p_num, _ = _tts(
                _X_te, y_te_num, train_size=_mx_p,
                random_state=config.RANDOM_STATE, stratify=y_te_num)
            _y_te_p = le.inverse_transform(_y_te_p_num)
            print(f"    [LLM-Prompted] test subset {len(_X_te):,} → {_mx_p:,}")
        else:
            _X_te_p, _y_te_p_num, _y_te_p = _X_te, y_te_num, y_te
        y_lp_pred_num = llm_p.predict(_X_te_p)
        y_lp_pred = le.inverse_transform(y_lp_pred_num)
        y_lp_proba = llm_p.predict_proba(_X_te_p)
        acc_p = accuracy_score(_y_te_p, y_lp_pred)
        f1w_p = f1_score(_y_te_p, y_lp_pred, average="weighted", zero_division=0)
        f1m_p = f1_score(_y_te_p, y_lp_pred, average="macro", zero_division=0)
        try:
            auc_p = roc_auc_score(_y_te_p_num, y_lp_proba,
                                  multi_class="ovr", average="weighted")
        except Exception:
            auc_p = np.nan
        print(f"  LLM-Prompted Acc: {acc_p:.4f} | F1w: {f1w_p:.4f} | F1m: {f1m_p:.4f} | AUC: {auc_p:.4f}")
        results["LLM-Prompted"] = {
            "model": llm_p, "accuracy": acc_p, "f1_weighted": f1w_p,
            "f1_macro": f1m_p, "roc_auc": auc_p, "y_pred": y_lp_pred,
            "y_test_eval": _y_te_p,
            "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
        }
    except Exception as exc:
        print(f"  ✗ LLM-Prompted failed: {exc}")
    gc.collect()

    # ── LLM-Prompted-Calibrated (Platt scaling) ─────────────
    if "LLM-Prompted" in results:
        print(f"\n  LLM-Prompted-Calibrated (Platt scaling) …")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import train_test_split as _tts

            _cal_n = getattr(config, "LLM_CALIBRATION_SAMPLES", 50)
            _X_cal, _, _y_cal, _ = _tts(
                _X_tr, y_tr, train_size=min(_cal_n, len(_X_tr) // 2),
                random_state=config.RANDOM_STATE, stratify=y_tr,
            )
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

            _X_te_cal = _X_te_p if '_X_te_p' in dir() else _X_te
            _y_te_cal = _y_te_p if '_y_te_p' in dir() else y_te
            _y_te_cal_num = _y_te_p_num if '_y_te_p_num' in dir() else y_te_num

            y_cal_pred_num = cal_clf.predict(_X_te_cal)
            y_cal_pred = le.inverse_transform(y_cal_pred_num)
            y_cal_proba = cal_clf.predict_proba(_X_te_cal)
            acc_c = accuracy_score(_y_te_cal, y_cal_pred)
            f1w_c = f1_score(_y_te_cal, y_cal_pred, average="weighted", zero_division=0)
            f1m_c = f1_score(_y_te_cal, y_cal_pred, average="macro", zero_division=0)
            try:
                auc_c = roc_auc_score(_y_te_cal_num, y_cal_proba,
                                      multi_class="ovr", average="weighted")
            except Exception:
                auc_c = np.nan
            print(f"  LLM-Prompted-Calibrated Acc: {acc_c:.4f} | F1w: {f1w_c:.4f} | F1m: {f1m_c:.4f} | AUC: {auc_c:.4f}")
            results["LLM-Prompted-Calibrated"] = {
                "model": cal_clf, "accuracy": acc_c, "f1_weighted": f1w_c,
                "f1_macro": f1m_c, "roc_auc": auc_c, "y_pred": y_cal_pred,
                "y_test_eval": _y_te_cal,
                "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
            }
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

            _prob_ml_meta = _best_ml.predict_proba(_X_meta_enc)   # (n, C)
            print(f"    [LLM-Stacked] LLM inference on {_stack_n} meta rows …")
            _prob_llm_meta = llm_p.predict_proba(_X_meta_raw)     # (n, C)

            _meta_X = np.hstack([_prob_ml_meta, _prob_llm_meta])  # (n, 2C)
            _y_meta_num = np.asarray(_y_meta)
            _meta_lr = LogisticRegression(
                multi_class="multinomial", max_iter=1000,
                random_state=config.RANDOM_STATE)
            _meta_lr.fit(_meta_X, _y_meta_num)
            print(f"    [LLM-Stacked] meta-learner fitted "
                  f"({_meta_X.shape[1]} features = 2×{_prob_ml_meta.shape[1]} classes)")

            # --- Test evaluation (reuse stored LLM predictions) ------
            _y_te_stack = results["LLM-Prompted"]["y_test_eval"]
            _y_te_stack_num = np.array([
                np.where(le.classes_ == lbl)[0][0] for lbl in _y_te_stack])

            # Reproduce same test subsampling on encoded data
            _mx_p = getattr(config, "LLM_PROMPTED_MAX_SAMPLES", 200)
            if len(X_te) > _mx_p:
                _X_te_enc_s, _, _, _ = _tts(
                    X_te, y_te_num, train_size=_mx_p,
                    random_state=config.RANDOM_STATE, stratify=y_te_num)
            else:
                _X_te_enc_s = X_te
            _ml_proba_te = _best_ml.predict_proba(_X_te_enc_s)    # (n, C)
            _llm_proba_te = y_lp_proba                             # stored from LLM-Prompted block

            _stack_te = np.hstack([_ml_proba_te, _llm_proba_te])
            _stack_proba = _meta_lr.predict_proba(_stack_te)       # (n, C)
            _stack_pred_num = _meta_lr.predict(_stack_te)
            _stack_pred = le.inverse_transform(_stack_pred_num)

            acc_s = accuracy_score(_y_te_stack, _stack_pred)
            f1w_s = f1_score(_y_te_stack, _stack_pred, average="weighted", zero_division=0)
            f1m_s = f1_score(_y_te_stack, _stack_pred, average="macro", zero_division=0)
            try:
                auc_s = roc_auc_score(_y_te_stack_num, _stack_proba,
                                      multi_class="ovr", average="weighted")
            except Exception:
                auc_s = np.nan
            print(f"  LLM-Stacked Acc: {acc_s:.4f} | F1w: {f1w_s:.4f} | F1m: {f1m_s:.4f} | AUC: {auc_s:.4f}")
            results["LLM-Stacked"] = {
                "model": _meta_lr, "accuracy": acc_s, "f1_weighted": f1w_s,
                "f1_macro": f1m_s, "roc_auc": auc_s, "y_pred": _stack_pred,
                "y_test_eval": _y_te_stack,
                "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
                "base_ml_model": _best_ml_name,
            }
        except Exception as exc:
            print(f"  ✗ LLM-Stacked failed: {exc}")
        gc.collect()


# ══════════════════════════════════════════════════════════════════
# BLOCK D — EVALUATION (cross-experiment comparison)
# ══════════════════════════════════════════════════════════════════

def compare_all_multiclass(result):
    """Print a summary comparison table from multiclass_retrofit_types().

    Parameters
    ----------
    result : dict — return value of ``multiclass_retrofit_types()``

    Returns
    -------
    pd.DataFrame sorted by F1-weighted descending.
    """
    cdf = result.get("comparison")
    if cdf is None or not len(cdf):
        return pd.DataFrame()
    cdf = cdf.copy()
    cdf.sort_values("F1_weighted", ascending=False, inplace=True)
    cdf.reset_index(drop=True, inplace=True)
    print(f"\n{'═'*60}")
    print("BLOCK D — Multiclass cross-experiment comparison")
    print(f"{'═'*60}")
    print(cdf.to_string(index=False))
    print(f"\nBest overall: {cdf.iloc[0]['Model']} "
          f"(F1w {cdf.iloc[0]['F1_weighted']:.4f})")
    return cdf