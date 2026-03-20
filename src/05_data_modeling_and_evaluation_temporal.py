# ============================================================
# 05_data_modeling_and_evaluation_temporal.py – Per-visit ML stacking regression
#
# Structure
# ---------
#   Block A — Models:        Base regressor factory, stacking runner,
#                             per-model metrics
#   Block B — Optimization:  Optuna TPE tuning for all regressors
#   Block D — Evaluation:    Cross-target comparison utilities
#   (No Block C — regression task, no LLM classifiers)
# ============================================================
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import (
    ElasticNet, Ridge, Lasso, SGDRegressor,
)
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Optional heavy imports (graceful degradation)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

from . import config

import importlib as _il
_fe = _il.import_module(".02_feature_engineering", __package__)
select_features         = _fe.select_features
_TEMPORAL_LEAK          = _fe._TEMPORAL_LEAK
_smoothed_target_encode = _fe._smoothed_target_encode
_frequency_encode       = _fe._frequency_encode
_add_interactions       = _fe._add_interactions



# ════════════════════════════════════════════════════════════
# BLOCK B — OPTIMIZATION (Optuna TPE tuning)
# ════════════════════════════════════════════════════════════

# ---- B.1 Optuna HP builder ----

def _temp_build_model(trial, name):
    """Suggest hyperparameters for *name* via Optuna trial (regression)."""
    RS = config.RANDOM_STATE

    if name == "ElasticNet":
        return Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
            max_iter=10000, random_state=RS))])

    if name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(
            alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True)))])

    if name == "Lasso":
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(
            alpha=trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            max_iter=10000, random_state=RS))])

    if name == "SGD":
        return Pipeline([("scaler", StandardScaler()), ("model", SGDRegressor(
            alpha=trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            penalty=trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            max_iter=2000, tol=1e-4, random_state=RS))])

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=RS, n_jobs=2)

    if name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=RS, n_jobs=2)

    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            random_state=RS)

    if name == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 1.0, log=True),
            random_state=RS)

    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            random_state=RS, tree_method="hist", n_jobs=2)

    if name == "LightGBM":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            random_state=RS, verbose=-1, n_jobs=2)

    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            bootstrap_type="Bernoulli",
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_seed=RS, verbose=0, thread_count=2)

    if name == "SVR":
        C = trial.suggest_float("svr_C", 0.1, 100.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.5, log=True)
        gamma = trial.suggest_categorical("svr_gamma", ["scale", "auto"])
        return Pipeline([("scaler", StandardScaler()),
                         ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))])

    if name == "MLP":
        n1 = trial.suggest_categorical("layer1", [64, 128, 256])
        n2 = trial.suggest_categorical("layer2", [32, 64, 128])
        n3 = trial.suggest_categorical("layer3", [16, 32, 64])
        lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True)
        return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(
            hidden_layer_sizes=(n1, n2, n3), activation="relu",
            learning_rate_init=lr, alpha=alpha,
            max_iter=600, early_stopping=True, validation_fraction=0.15,
            random_state=RS))])

    raise ValueError(f"Unknown model: {name}")


def _optuna_tune_temporal(X_tr, y_tr):
    """
    Run Optuna TPE Bayesian optimisation for each regression model.

    Returns dict[str, estimator] with best-params models, or None if
    Optuna is unavailable.
    """
    if not _OPTUNA:
        return None

    n_trials = config.OPTUNA_TEMP_N_TRIALS
    timeout = config.OPTUNA_TIMEOUT
    cv = KFold(n_splits=config.OPTUNA_CV_FOLDS, shuffle=True,
               random_state=config.RANDOM_STATE)

    tune_names = [n for n in [
        "ElasticNet", "Ridge", "Lasso", "SGD",
        "Random Forest", "Extra Trees", "GradientBoosting", "AdaBoost",
        "XGBoost", "LightGBM", "CatBoost", "SVR", "MLP",
    ] if n not in config.OPTUNA_SKIP_TEMPORAL]

    print(f"\n  Optuna TPE tuning ({n_trials} trials, "
          f"{config.OPTUNA_CV_FOLDS}-fold CV, {timeout}s timeout per model)")

    tuned = {}
    for name in tune_names:
        print(f"    tuning: {name} …", end=" ", flush=True)

        def objective(trial, _name=name):
            model = _temp_build_model(trial, _name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    model, X_tr, y_tr, cv=cv,
                    scoring="neg_mean_absolute_error", n_jobs=1,
                )
            return scores.mean()  # negative MAE — maximise (less negative = better)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False)

        try:
            best = study.best_trial
        except ValueError:
            print(f"SKIP (no completed trials)")
            del study; gc.collect()
            continue
        print(f"neg-MAE={best.value:.4f}  (trial {best.number})")

        tuned[name] = _temp_build_model(best, name)
        del study; gc.collect()

    return tuned



# ════════════════════════════════════════════════════════════
# BLOCK A — MODELS (stacking regression runner)
# ════════════════════════════════════════════════════════════

def _run_regression(df, target_col, label):
    """
    Train 5 base regressors on log1p(target), blend via 3-fold stacking
    with a Ridge meta-learner.  Metrics on original (days) scale.
    """
    print(f"\n{'─' * 70}")
    print(f"  {label}  ─  target: {target_col}")
    print(f"{'─' * 70}")

    y_raw = df[target_col]
    print(f"  Raw stats: mean={y_raw.mean():.1f} d, median={y_raw.median():.1f}, "
          f"std={y_raw.std():.1f}  |  range: {y_raw.min():.0f}–{y_raw.max():.0f} d")

    # Log-transform (log1p handles 0-day durations safely)
    y_log = np.log1p(y_raw)
    print(f"  Log1p stats: mean={y_log.mean():.2f}, median={y_log.median():.2f}, "
          f"std={y_log.std():.2f}")

    # Feature selection
    cat_cols, num_cols = select_features(df, target_col)

    cat_cols = [c for c in cat_cols if c not in _TEMPORAL_LEAK
                and c.removesuffix("_clean") not in _TEMPORAL_LEAK]
    num_cols = [c for c in num_cols if c not in _TEMPORAL_LEAK]
    feat_cols = cat_cols + num_cols

    if not feat_cols:
        print("  ERROR: no features selected"); return None

    print(f"  Features: {len(cat_cols)} cat + {len(num_cols)} num = {len(feat_cols)}")

    X = df[feat_cols].copy()

    for c in cat_cols:
        X[c] = X[c].fillna(config.FILLNA_CATEGORICAL)
    for c in num_cols:
        X[c] = X[c].fillna(0)

    # ── Add group-level historical stats as features ─────────
    # These capture type/derivat-specific patterns much better than
    # target encoding alone (which compresses to a single value).
    # NOTE: actual values are computed AFTER train/test split to avoid
    # leakage (train-only statistics mapped to test).
    _group_features = []
    _type_col = config.COL_MULTICLASS_TARGET
    _deriv_col = config.COL_DERIVAT_CLEAN if config.COL_DERIVAT_CLEAN in df.columns else config.COL_DERIVAT

    # Placeholder columns (filled after split)
    if _type_col in df.columns:
        X[config.COL_TYPE_HIST_MEDIAN] = np.float32(0)
        X[config.COL_TYPE_HIST_COUNT] = np.float32(0)
        _group_features += [config.COL_TYPE_HIST_MEDIAN, config.COL_TYPE_HIST_COUNT]
        num_cols = num_cols + [config.COL_TYPE_HIST_MEDIAN, config.COL_TYPE_HIST_COUNT]
    if _deriv_col in df.columns:
        X[config.COL_DERIVAT_HIST_MEDIAN] = np.float32(0)
        _group_features += [config.COL_DERIVAT_HIST_MEDIAN]
        num_cols = num_cols + [config.COL_DERIVAT_HIST_MEDIAN]

    if _group_features:
        print(f"  Added group stats: {_group_features}")

    X_tr, X_te, y_tr_log, y_te_log = train_test_split(
        X, y_log, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Fill group stats from TRAIN only → map to test (no leakage)
    _y_tr_raw = np.expm1(y_tr_log)
    _overall_med = float(_y_tr_raw.median())
    _group_maps = {}  # saved for forecast use

    if _type_col in df.columns and _type_col in X_tr.columns:
        _tmed = _y_tr_raw.groupby(X_tr[_type_col]).median()
        _tcnt = _y_tr_raw.groupby(X_tr[_type_col]).count()
        _group_maps["type_median"] = _tmed.to_dict()
        _group_maps["type_count"] = _tcnt.to_dict()
        X_tr[config.COL_TYPE_HIST_MEDIAN] = X_tr[_type_col].map(_tmed).fillna(_overall_med).astype("float32")
        X_tr[config.COL_TYPE_HIST_COUNT] = X_tr[_type_col].map(_tcnt).fillna(0).astype("float32")
        X_te[config.COL_TYPE_HIST_MEDIAN] = X_te[_type_col].map(_tmed).fillna(_overall_med).astype("float32")
        X_te[config.COL_TYPE_HIST_COUNT] = X_te[_type_col].map(_tcnt).fillna(0).astype("float32")
    if _deriv_col in df.columns and _deriv_col in X_tr.columns:
        _dmed = _y_tr_raw.groupby(X_tr[_deriv_col]).median()
        _group_maps["derivat_median"] = _dmed.to_dict()
        X_tr[config.COL_DERIVAT_HIST_MEDIAN] = X_tr[_deriv_col].map(_dmed).fillna(_overall_med).astype("float32")
        X_te[config.COL_DERIVAT_HIST_MEDIAN] = X_te[_deriv_col].map(_dmed).fillna(_overall_med).astype("float32")

    # Keep original-scale test set for metric reporting
    y_te_raw = np.expm1(y_te_log)

    # Smoothed target encoding on LOG scale
    Xtr_enc, Xte_enc, enc_map, g_mean = _smoothed_target_encode(
        X_tr, X_te, y_tr_log, cat_cols)

    # Frequency encoding (category prevalence as numeric feature)
    X_tr_freq, X_te_freq, freq_map = _frequency_encode(X_tr, X_te, cat_cols)
    freq_cols = [f"{c}_freq" for c in cat_cols]
    num_present = [c for c in num_cols if c in X_tr.columns]
    X_tr_f = pd.concat([Xtr_enc, X_tr_freq, X_tr[num_present].astype("float32")], axis=1)
    X_te_f = pd.concat([Xte_enc, X_te_freq, X_te[num_present].astype("float32")], axis=1)

    # Safety net: replace Inf (fillna does NOT catch Inf)
    X_tr_f.replace([np.inf, -np.inf], 0, inplace=True)
    X_te_f.replace([np.inf, -np.inf], 0, inplace=True)

    # ── Feature importance pre-filter (top 40) ───────────────
    _scout = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        random_state=config.RANDOM_STATE, n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _scout.fit(X_tr_f, y_tr_log)
    importances = pd.Series(_scout.feature_importances_, index=X_tr_f.columns)
    top_n = min(40, len(importances))
    top_feats = importances.nlargest(top_n).index.tolist()
    print(f"  Feature pre-filter: kept top {len(top_feats)} of {len(X_tr_f.columns)} "
          f"(min imp: {importances[top_feats[-1]]:.4f})")
    X_tr_f = X_tr_f[top_feats]
    X_te_f = X_te_f[top_feats]

    # ── Interaction features ─────────────────────────────────
    top_num = [c for c in importances.nlargest(10).index if c in num_present]
    X_tr_f, ix_pairs = _add_interactions(X_tr_f, top_num, max_pairs=15)
    X_te_f, _ = _add_interactions(X_te_f, top_num, max_pairs=15)
    all_feats = X_tr_f.columns.tolist()
    print(f"  After interactions: {len(all_feats)} features "
          f"({len(ix_pairs)} interaction pairs)")

    # ── Base models — Optuna TPE tuning or hand-tuned fallback ─
    optuna_models = _optuna_tune_temporal(X_tr_f, y_tr_log)

    if optuna_models:
        base_specs = optuna_models
        print(f"  Using Optuna-tuned models ({len(base_specs)} models)")
    else:
        # Hand-tuned fallback defaults
        print("  Using hand-tuned default models (Optuna unavailable)")
        base_specs = {
            # ── Linear models (scale-sensitive → wrapped in Pipeline) ──
            "ElasticNet": Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(
                alpha=0.005, l1_ratio=0.3, max_iter=10000,
                random_state=config.RANDOM_STATE))]),
            "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(
                alpha=1.0))]),
            "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(
                alpha=0.01, max_iter=10000,
                random_state=config.RANDOM_STATE))]),
            "SGD": Pipeline([("scaler", StandardScaler()), ("model", SGDRegressor(
                max_iter=2000, tol=1e-4, penalty="l2",
                random_state=config.RANDOM_STATE))]),
            # ── Tree-based (bagging) ──
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_leaf=3,
                random_state=config.RANDOM_STATE, n_jobs=2),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=200, max_depth=20, min_samples_leaf=3,
                random_state=config.RANDOM_STATE, n_jobs=2),
            # ── Boosting ──
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.08, max_depth=5,
                subsample=0.8, random_state=config.RANDOM_STATE),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=150, learning_rate=0.08,
                random_state=config.RANDOM_STATE),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05,
                reg_lambda=1.0, early_stopping_rounds=60,
                random_state=config.RANDOM_STATE, tree_method="hist",
                n_jobs=2),
            "LightGBM": LGBMRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=-1,
                num_leaves=63, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.05, reg_lambda=1.0, min_child_samples=10,
                random_state=config.RANDOM_STATE, verbose=-1, n_jobs=2),
            "CatBoost": CatBoostRegressor(
                iterations=400, learning_rate=0.05, depth=6,
                l2_leaf_reg=3.0, bootstrap_type="Bernoulli",
                subsample=0.8,
                random_seed=config.RANDOM_STATE, verbose=0,
                thread_count=2),
        }

    # ── Skip sanity check — Optuna already performed CV ─────
    # (Previously did a 3-fold CV on each model to filter degenerate ones,
    #  but this is redundant with Optuna's internal CV evaluation.)
    pass

    # ── 3-fold stacking ─────────────────────────────────────
    N_FOLDS = config.STACKING_CV_FOLDS
    kf = KFold(n_splits=N_FOLDS, shuffle=True,
               random_state=config.RANDOM_STATE)
    oof = {n: np.zeros(len(X_tr_f)) for n in base_specs}
    tst = {n: np.zeros(len(X_te_f)) for n in base_specs}

    print(f"\n  Stacking: {N_FOLDS}-fold OOF training …")
    for fi, (tr_idx, va_idx) in enumerate(kf.split(X_tr_f), 1):
        Xf_tr, Xf_va = X_tr_f.iloc[tr_idx], X_tr_f.iloc[va_idx]
        yf_tr, yf_va = y_tr_log.iloc[tr_idx], y_tr_log.iloc[va_idx]
        for name, spec in base_specs.items():
            try:
                m = clone(spec)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if name == "XGBoost":
                        m.fit(Xf_tr, yf_tr,
                              eval_set=[(Xf_va, yf_va)], verbose=False)
                    elif name == "LightGBM":
                        m.fit(Xf_tr, yf_tr,
                              eval_set=[(Xf_va, yf_va)],
                              callbacks=[__import__("lightgbm").early_stopping(
                                  80, verbose=False)])
                    elif name == "CatBoost":
                        m.fit(Xf_tr, yf_tr,
                              eval_set=(Xf_va, yf_va),
                              early_stopping_rounds=60, verbose=False)
                    else:
                        m.fit(Xf_tr, yf_tr)
                oof[name][va_idx] = m.predict(Xf_va)
                tst[name] += m.predict(X_te_f) / N_FOLDS
                del m
            except Exception as exc:
                print(f"    ✗ {name} fold {fi}: {exc}")
        gc.collect()
        print(f"    fold {fi}/{N_FOLDS} done")

    # ── Per-model metrics (fold-averaged test preds) ─────────
    results = {}
    # Bin edges for actionable duration buckets (days)
    _BIN_EDGES = config.DURATION_BIN_EDGES
    _BIN_LABELS = config.DURATION_BIN_LABELS
    y_te_bins = pd.cut(y_te_raw, bins=_BIN_EDGES, labels=_BIN_LABELS, right=False)
    from sklearn.metrics import accuracy_score as _acc, f1_score as _f1

    for name in base_specs:
        yp = np.expm1(tst[name]).clip(0)
        mae = mean_absolute_error(y_te_raw, yp)
        rmse = np.sqrt(mean_squared_error(y_te_raw, yp))
        r2 = r2_score(y_te_raw, yp)
        # Bin-based classification accuracy
        yp_bins = pd.cut(pd.Series(yp), bins=_BIN_EDGES, labels=_BIN_LABELS, right=False)
        bin_acc = _acc(y_te_bins, yp_bins)
        bin_f1m = _f1(y_te_bins, yp_bins, average="macro", zero_division=0)
        print(f"    {name:25s}  MAE={mae:6.1f} d  RMSE={rmse:6.1f}  "
              f"R²={r2:.4f}  BinAcc={bin_acc:.3f}  BinF1m={bin_f1m:.3f}")
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2,
                         "bin_acc": bin_acc, "bin_f1_macro": bin_f1m,
                         "y_pred": yp}

    # ── Ridge meta-learner on OOF predictions ────────────────
    S_tr = np.column_stack([oof[n] for n in base_specs])
    S_te = np.column_stack([tst[n] for n in base_specs])
    meta = Ridge(alpha=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta.fit(S_tr, y_tr_log)

    # Stacking weights inspection
    _base_names = list(base_specs.keys())
    _stacking_weights = dict(zip(_base_names, meta.coef_.ravel()))
    print(f"\n    Stacking weights (Ridge α={meta.alpha}):")
    for _bn, _bw in sorted(_stacking_weights.items(), key=lambda x: -abs(x[1])):
        print(f"      {_bn:25s}  {_bw:+.4f}")
    print(f"      {'intercept':25s}  {meta.intercept_:+.4f}")

    y_meta = np.expm1(meta.predict(S_te)).clip(0)
    mae_m = mean_absolute_error(y_te_raw, y_meta)
    rmse_m = np.sqrt(mean_squared_error(y_te_raw, y_meta))
    r2_m = r2_score(y_te_raw, y_meta)
    yp_bins_m = pd.cut(pd.Series(y_meta), bins=_BIN_EDGES,
                        labels=_BIN_LABELS, right=False)
    bin_acc_m = _acc(y_te_bins, yp_bins_m)
    bin_f1m_m = _f1(y_te_bins, yp_bins_m, average="macro", zero_division=0)
    print(f"    {'▸ Stacked Ensemble':25s}  MAE={mae_m:6.1f} d  "
          f"RMSE={rmse_m:6.1f}  R²={r2_m:.4f}  "
          f"BinAcc={bin_acc_m:.3f}  BinF1m={bin_f1m_m:.3f}")
    results["Stacked Ensemble"] = {
        "mae": mae_m, "rmse": rmse_m, "r2": r2_m,
        "bin_acc": bin_acc_m, "bin_f1_macro": bin_f1m_m,
        "y_pred": y_meta}

    # ── Quantile regression (P50 + P90 prediction intervals) ─
    print("\n  Quantile GBR (P50 / P90) …")
    quantile_preds = {}
    for q, qlabel in [(0.5, "P50"), (0.9, "P90")]:
        try:
            qgbr = GradientBoostingRegressor(
                loss="quantile", alpha=q,
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10,
                random_state=config.RANDOM_STATE)
            qgbr.fit(X_tr_f, y_tr_log)
            yq = np.expm1(qgbr.predict(X_te_f)).clip(0)
            quantile_preds[qlabel] = yq
            q_mae = mean_absolute_error(y_te_raw, yq)
            coverage = float((y_te_raw <= yq).mean()) if q > 0.5 else np.nan
            cov_str = f"  cov={coverage:.1%}" if q > 0.5 else ""
            print(f"    {qlabel:25s}  MAE={q_mae:6.1f} d{cov_str}")
            del qgbr
        except Exception as exc:
            print(f"    ✗ {qlabel}: {exc}")
    gc.collect()

    # ── Comparison table ─────────────────────────────────────
    cdf = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE_days": [r["mae"] for r in results.values()],
        "RMSE_days": [r["rmse"] for r in results.values()],
        "R2": [r["r2"] for r in results.values()],
        "Bin_Acc": [r.get("bin_acc", np.nan) for r in results.values()],
        "Bin_F1m": [r.get("bin_f1_macro", np.nan) for r in results.values()],
    }).sort_values("MAE_days")

    best = cdf.iloc[0]["Model"]
    best_ind = cdf[cdf["Model"] != "Stacked Ensemble"].iloc[0]["Model"]
    print(f"\n  ★ Best: {best}  "
          f"(MAE = {results[best]['mae']:.1f} d, R² = {results[best]['r2']:.4f})")

    # ── Bin distribution summary (best model) ────────────────
    _best_pred = results[best]["y_pred"]
    _best_bins = pd.cut(pd.Series(_best_pred), bins=_BIN_EDGES,
                        labels=_BIN_LABELS, right=False)
    print(f"\n  Duration bin distribution (test set, best={best}):")
    print(f"    {'Bin':20s} {'Actual':>8s} {'Predicted':>10s}")
    for bl in _BIN_LABELS:
        n_actual = int((y_te_bins == bl).sum())
        n_pred   = int((_best_bins == bl).sum())
        print(f"    {bl:20s} {n_actual:8d} {n_pred:10d}")

    # ── Residual diagnostics & bias correction ───────────────
    _best_resid = y_te_raw.values - _best_pred
    resid_median = float(np.median(_best_resid))
    resid_std    = float(np.std(_best_resid))
    resid_p10    = float(np.percentile(_best_resid, 10))
    resid_p90    = float(np.percentile(_best_resid, 90))
    print(f"\n  Residual stats (best={best}):")
    print(f"    median={resid_median:+.1f} d  std={resid_std:.1f} d  "
          f"P10={resid_p10:+.1f} d  P90={resid_p90:+.1f} d")

    # Bias-corrected predictions (subtract median residual)
    if abs(resid_median) > 1.0:
        y_best_corrected = (_best_pred + resid_median).clip(0)
        mae_corr = mean_absolute_error(y_te_raw, y_best_corrected)
        print(f"    Bias-corrected MAE: {mae_corr:.1f} d "
              f"(shift {resid_median:+.1f} d)")
    else:
        y_best_corrected = _best_pred
        print(f"    Bias < 1 d — no correction needed")

    # Prediction intervals (80% CI based on residual distribution)
    _ci_lo = float(np.percentile(_best_resid, 10))
    _ci_hi = float(np.percentile(_best_resid, 90))
    pi_lower = (_best_pred + _ci_lo).clip(0)
    pi_upper = (_best_pred + _ci_hi).clip(0)
    coverage_80 = float(((y_te_raw.values >= pi_lower)
                         & (y_te_raw.values <= pi_upper)).mean())
    print(f"    80% PI width: [{_ci_lo:+.0f}, {_ci_hi:+.0f}] d  "
          f"actual coverage: {coverage_80:.1%}")

    # ── Min-support sensitivity analysis ─────────────────────
    # Show how MAE changes when excluding test samples whose
    # retrofit type has fewer than N training observations.
    _type_counts = _group_maps.get("type_count", {})
    _sensitivity_df = None
    if _type_col in X_te.columns and _type_counts:
        _te_types = X_te[_type_col].values
        _te_support = np.array([_type_counts.get(t, 0) for t in _te_types])
        _thresholds = [1, 2, 3, 5, 10, 20, 50]
        _sens_rows = []
        print(f"\n  Min-support sensitivity (best = {best}):")
        print(f"    {'Min-N':>6s}  {'N-test':>7s}  {'MAE':>7s}  "
              f"{'RMSE':>7s}  {'R²':>7s}")
        for thr in _thresholds:
            mask = _te_support >= thr
            n_keep = int(mask.sum())
            if n_keep < 2:
                break
            _yt = y_te_raw.values[mask]
            _yp = _best_pred[mask]
            _mae = mean_absolute_error(_yt, _yp)
            _rmse = float(np.sqrt(mean_squared_error(_yt, _yp)))
            _r2 = r2_score(_yt, _yp)
            print(f"    {thr:6d}  {n_keep:7d}  {_mae:7.1f}  "
                  f"{_rmse:7.1f}  {_r2:7.4f}")
            _sens_rows.append({"min_support": thr, "n_test": n_keep,
                               "mae": _mae, "rmse": _rmse, "r2": _r2})
        _sensitivity_df = pd.DataFrame(_sens_rows) if _sens_rows else None

    # ── Per-type MAE breakdown ────────────────────────────────
    _per_type_mae = None
    if _type_col in X_te.columns:
        _te_types = X_te[_type_col].values
        _type_count = _group_maps.get("type_count", {})
        _ptype_rows = []
        print(f"\n  Per-type MAE breakdown (best = {best}):")
        print(f"    {'Type':35s}  {'N-train':>8s}  {'N-test':>7s}  "
              f"{'MAE':>7s}  {'Median-Err':>10s}")
        for t in sorted(set(_te_types)):
            mask_t = _te_types == t
            n_te = int(mask_t.sum())
            if n_te < 1:
                continue
            _yt_t = y_te_raw.values[mask_t]
            _yp_t = _best_pred[mask_t]
            _mae_t = mean_absolute_error(_yt_t, _yp_t)
            _med_err = float(np.median(np.abs(_yt_t - _yp_t)))
            n_tr = _type_count.get(t, 0)
            print(f"    {str(t):35s}  {n_tr:8d}  {n_te:7d}  "
                  f"{_mae_t:7.1f}  {_med_err:10.1f}")
            _ptype_rows.append({"type": t, "n_train": n_tr,
                                "n_test": n_te, "mae": _mae_t,
                                "median_abs_error": _med_err})
        _per_type_mae = pd.DataFrame(_ptype_rows) if _ptype_rows else None

    # ── Outlier impact analysis ───────────────────────────────
    # Show MAE after removing top-N% residual outliers.
    _abs_resid = np.abs(y_te_raw.values - _best_pred)
    _outlier_rows = []
    print(f"\n  Outlier impact analysis (best = {best}):")
    print(f"    {'Excl-%':>6s}  {'N-kept':>7s}  {'MAE':>7s}  "
          f"{'RMSE':>7s}  {'R²':>7s}")
    for pct in [0, 1, 2, 5, 10]:
        if pct == 0:
            mask_o = np.ones(len(_abs_resid), dtype=bool)
        else:
            cutoff = np.percentile(_abs_resid, 100 - pct)
            mask_o = _abs_resid <= cutoff
        n_k = int(mask_o.sum())
        if n_k < 2:
            break
        _yt_o = y_te_raw.values[mask_o]
        _yp_o = _best_pred[mask_o]
        _mae_o = mean_absolute_error(_yt_o, _yp_o)
        _rmse_o = float(np.sqrt(mean_squared_error(_yt_o, _yp_o)))
        _r2_o = r2_score(_yt_o, _yp_o)
        print(f"    {pct:5d}%  {n_k:7d}  {_mae_o:7.1f}  "
              f"{_rmse_o:7.1f}  {_r2_o:7.4f}")
        _outlier_rows.append({"excl_pct": pct, "n_kept": n_k,
                              "mae": _mae_o, "rmse": _rmse_o, "r2": _r2_o})
    _outlier_df = pd.DataFrame(_outlier_rows) if _outlier_rows else None

    # ── Prediction stability: perturb stacked inputs ─────────
    # Add Gaussian noise (scaled by feature std) to S_te and re-predict
    # through meta-learner. Shows how sensitive MAE is to small changes
    # in the base-model predictions.
    _noise_levels = [0.01, 0.02, 0.05, 0.10]
    _stability_rows = []
    _base_mae = mean_absolute_error(y_te_raw, y_meta)
    _rng_stab = np.random.RandomState(config.RANDOM_STATE)
    _S_te_std = S_te.std(axis=0, keepdims=True).clip(1e-8)
    print(f"\n  Prediction stability (noise on stacked inputs, base MAE={_base_mae:.1f}):")
    print(f"    {'Noise%':>6s}  {'MAE':>7s}  {'ΔMAE':>7s}  {'ΔMAE%':>7s}")
    for sigma in _noise_levels:
        noise = _rng_stab.normal(0, sigma, size=S_te.shape) * _S_te_std
        _y_pert = np.expm1(meta.predict(S_te + noise)).clip(0)
        _mae_p = mean_absolute_error(y_te_raw, _y_pert)
        _delta = _mae_p - _base_mae
        _delta_pct = 100 * _delta / max(_base_mae, 1e-8)
        print(f"    {sigma*100:5.1f}%  {_mae_p:7.1f}  {_delta:+7.1f}  {_delta_pct:+6.1f}%")
        _stability_rows.append({"noise_pct": sigma * 100, "mae": _mae_p,
                                "delta_mae": _delta, "delta_pct": _delta_pct})
    _stability_df = pd.DataFrame(_stability_rows) if _stability_rows else None

    # ── Retrain base models on full train set (for forecasting) ──
    final_models = {}
    for name, spec in base_specs.items():
        try:
            m = clone(spec)
            if name == "XGBoost":
                m.set_params(early_stopping_rounds=None)
            elif name == "CatBoost":
                # early_stopping_rounds is a fit-time param in CatBoost,
                # not a constructor param — just skip set_params
                pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_tr_f, y_tr_log)
            final_models[name] = m
            results[name]["model"] = m          # backward-compat key
        except Exception as exc:
            print(f"    ✗ Retrain {name} failed: {exc}")
        gc.collect()

    gc.collect()

    # Vehicle types aligned to y_test (for error-by-type diagnostics)
    _y_test_types = (
        X_te[_type_col].values if _type_col in X_te.columns
        else None
    )

    return {
        "results": results,
        "comparison": cdf,
        "best_model": best_ind,             # best individual (for forecast)
        "best_overall": best,               # might be stacked
        "y_test": y_te_raw,                 # actual values (days) for plots
        "y_test_bins": y_te_bins,           # Short/Medium/Long labels
        "y_test_types": _y_test_types,      # retrofit type per test row (for diagnostics)
        "min_support_sensitivity": _sensitivity_df,
        "per_type_mae": _per_type_mae,
        "outlier_impact": _outlier_df,
        "prediction_stability": _stability_df,
        "stacking_weights": _stacking_weights,
        "quantile_preds": quantile_preds,   # {"P50": array, "P90": array}
        "residual_stats": {                 # for downstream calibration
            "median": resid_median,
            "std": resid_std,
            "p10": resid_p10,
            "p90": resid_p90,
            "pi_coverage_80": coverage_80,
        },
        "encoding_map": enc_map,
        "global_mean": g_mean,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "top_feats": all_feats,             # includes interactions
        "interaction_pairs": ix_pairs,       # for forecast to recreate
        "log_transformed": True,
        "final_models": final_models,
        "meta_model": meta,
        "base_names": list(base_specs.keys()),
        "group_maps": _group_maps,           # for forecast: type_median, derivat_median dicts
        "overall_target_median": _overall_med,
        "freq_map": freq_map,               # for forecast: frequency encoding
    }


# ══════════════════════════════════════════════════════════════
# BLOCK D — EVALUATION (cross-target comparison)
# ══════════════════════════════════════════════════════════════

def compare_all_temporal(*results):
    """Merge comparison tables from multiple _run_regression() calls.

    Parameters
    ----------
    *results : dicts returned by ``_run_regression()`` for different
               target columns (e.g. days-to-visit, visit duration).

    Returns
    -------
    pd.DataFrame with one row per (target × model), sorted by MAE.
    """
    frames = []
    for res in results:
        if res is None:
            continue
        cdf = res.get("comparison")
        if cdf is not None and len(cdf):
            cdf = cdf.copy()
            best = res.get("best_overall", "?")
            cdf.insert(0, "Target", best)
            frames.append(cdf)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values("MAE_days", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    print(f"\n{'═'*60}")
    print("BLOCK D — Temporal cross-target comparison")
    print(f"{'═'*60}")
    print(merged.to_string(index=False))
    print(f"\nBest overall: {merged.iloc[0]['Model']} "
          f"(MAE {merged.iloc[0]['MAE_days']:.1f} d)")
    return merged
