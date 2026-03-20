# ============================================================
# 02_feature_engineering.py - Feature engineering for all model families
#
# KDD-structured feature engineering pipeline:
#   Section 5 - Shared Helpers (constants, select_features, encodings)
#   Section 6 - Feature Engineering: Binary Classification
#   Section 7 - Feature Engineering: Multiclass Classification
#   Section 8 - Feature Engineering: Temporal Regression
#   Section 9 - Feature Engineering: Time Series
# ============================================================
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from . import config


# ══════════════════════════════════════════════════════════════
# SECTION 5 — Shared Feature Engineering Helpers
# ══════════════════════════════════════════════════════════════

# These constants and functions are shared across Sections 5-8.
# They define which columns to exclude and how to select features.

# Columns to never use as features (IDs, merge artifacts)
_EXCLUDE = {config.COL_V_NUMMER, config.COL_JOIN_KEY_SOURCE_A, "bt_nummer", config.COL_JOIN_KEY_SOURCE_B,
            "_merge", config.COL_VNUMBER_SOURCE_B,
            config.COL_JOIN_KEY}   # composite fallback key
# Also exclude _clean versions of ID columns (high-cardinality, overfit risk)
_EXCLUDE_CLEAN = {f"{c}_clean" for c in _EXCLUDE} | {
    "v_nummer_clean", "at_nummer_clean", "bt_nummer_clean",
    "_join_key_clean"}

# Columns that leak temporal targets (used by Sections 7-8)
_TEMPORAL_LEAK = {
    # Raw date columns (direct leakage)
    config.COL_START_UR, config.COL_AV_UR, config.COL_AV_FIRST_ISTUFE, config.COL_ZIEL_SAB,
    # Features derived from umr_start in cleanse_source_a
    "month", "quarter", "year", "day_of_week", "week_of_year",
    "monthly_retrofit_count",
    "lead_time_category", "lead_time_category_clean",
    # Derived from umr_av/av_fzg in _engineer_features
    "car_ready_age_months", "retrofit_ready_age_months",
    "car_ready_category", "retrofit_ready_category",
    "is_recent_av_fzg", "is_recent_av_retrofit",
}


def select_features(df, target):
    """
    Select features for ML (shared by Binary, Multiclass, Temporal):
    - Categorical: columns ending in '_clean' (already NaN-safe)
    - Numeric:     all numeric columns
    - Exclude:     target, IDs, dates, raw versions of _clean cols
    """
    cat_cols = [c for c in df.columns
                if c.endswith("_clean") and c != target
                and c not in _EXCLUDE and c not in _EXCLUDE_CLEAN]

    raw_of_clean = {c.removesuffix("_clean") for c in cat_cols}

    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and c != target
                and c not in _EXCLUDE
                and c not in raw_of_clean
                and not pd.api.types.is_datetime64_any_dtype(df[c])]

    print(f"  Features: {len(cat_cols)} categorical (_clean) + {len(num_cols)} numeric = {len(cat_cols)+len(num_cols)} total")
    return cat_cols, num_cols


def _sanitize_features(X_tr, X_te, label=""):
    """Replace NaN/Inf in feature matrices (returns clean copies)."""
    n_nan = int(np.isnan(X_tr.values).sum()) + int(np.isnan(X_te.values).sum())
    n_inf = int(np.isinf(X_tr.values).sum()) + int(np.isinf(X_te.values).sum())
    if n_nan or n_inf:
        print(f"  ⚠ [{label}] Sanitising features: {n_nan} NaN, {n_inf} Inf")
        X_tr = X_tr.fillna(0).replace([np.inf, -np.inf], 0)
        X_te = X_te.fillna(0).replace([np.inf, -np.inf], 0)
    return X_tr, X_te


def _smoothed_target_encode(X_tr, X_te, y_tr, cat_cols, min_samples=20):
    """Bayesian-smoothed target encoding (reduces overfitting on rare cats)."""
    g_mean = float(y_tr.mean())
    enc_map = {}
    Xtr_e = pd.DataFrame(index=X_tr.index)
    Xte_e = pd.DataFrame(index=X_te.index)
    for col in cat_cols:
        agg = y_tr.groupby(X_tr[col]).agg(["mean", "count"])
        w = agg["count"] / (agg["count"] + min_samples)
        vals = w * agg["mean"] + (1 - w) * g_mean
        m = vals.to_dict()
        enc_map[col] = m
        Xtr_e[col] = X_tr[col].map(m).fillna(g_mean).astype("float32")
        Xte_e[col] = X_te[col].map(m).fillna(g_mean).astype("float32")
    return Xtr_e, Xte_e, enc_map, g_mean


def _frequency_encode(X_tr, X_te, cat_cols):
    """Frequency encoding: proportion of each category in training set."""
    freq_map = {}
    Xtr_f = pd.DataFrame(index=X_tr.index)
    Xte_f = pd.DataFrame(index=X_te.index)
    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True).to_dict()
        freq_map[col] = freq
        fname = f"{col}_freq"
        Xtr_f[fname] = X_tr[col].map(freq).fillna(0).astype("float32")
        Xte_f[fname] = X_te[col].map(freq).fillna(0).astype("float32")
    return Xtr_f, Xte_f, freq_map


def _add_interactions(X, top_num, max_pairs=15):
    """Pairwise products of top numeric features — captures non-linear signal."""
    pairs, new = [], {}
    for i, a in enumerate(top_num):
        if a not in X.columns:
            continue
        for b in top_num[i + 1:]:
            if b not in X.columns or len(pairs) >= max_pairs:
                break
            col_name = f"ix_{a[:12]}_{b[:12]}"
            new[col_name] = (X[a] * X[b]).astype("float32")
            pairs.append((a, b, col_name))
    if new:
        return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1), pairs
    return X, pairs


# ══════════════════════════════════════════════════════════════
# SECTION 6 — Feature Engineering: Binary Classification
# ══════════════════════════════════════════════════════════════

def prepare_binary(df_binary):
    """
    Build feature matrix, apply target encoding, split into train/test.
    Returns a dict with everything needed for binary model training.
    """
    target = config.COL_CAME_FOR_RETROFIT

    # Select features: _clean (categorical) + numeric
    cat_cols, num_cols = select_features(df_binary, target)
    feature_cols = cat_cols + num_cols

    df_model = df_binary[feature_cols + [target]].copy()

    # Fill remaining NaN
    for col in cat_cols:
        df_model[col] = df_model[col].fillna(config.FILLNA_CATEGORICAL)
    for col in num_cols:
        _med = df_model[col].median()
        df_model[col] = df_model[col].fillna(_med if pd.notna(_med) else 0)

    # Only drop rows where target is missing
    df_model = df_model.dropna(subset=[target])

    # Safety check: both classes must exist
    vc = df_model[target].value_counts()
    print(f"Target distribution:\n{vc.to_string()}")
    if len(vc) < 2:
        raise ValueError(
            f"Only 1 class found ({vc.index[0]}). "
            f"Re-run the merge cell to regenerate df_merged."
        )

    X, y = df_model[feature_cols], df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # Target encoding (fit on train only to prevent leakage)
    encoding_map = {}
    global_mean = y_train.mean()

    X_tr_enc = pd.DataFrame(index=X_train.index)
    X_te_enc = pd.DataFrame(index=X_test.index)
    for col in cat_cols:
        col_map = y_train.groupby(X_train[col]).mean().to_dict()
        encoding_map[col] = col_map
        X_tr_enc[col] = X_train[col].map(col_map).fillna(global_mean).astype("float32")
        X_te_enc[col] = X_test[col].map(col_map).fillna(global_mean).astype("float32")

    X_train_final = pd.concat(
        [X_tr_enc, X_train[num_cols].astype("float32")], axis=1
    )
    X_test_final = pd.concat(
        [X_te_enc, X_test[num_cols].astype("float32")], axis=1
    )

    # Safety net: replace any residual Inf with 0 (fillna does NOT catch Inf)
    X_train_final.replace([np.inf, -np.inf], 0, inplace=True)
    X_test_final.replace([np.inf, -np.inf], 0, inplace=True)

    print(f"Samples: {len(df_model):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Features: {X_train_final.shape[1]} | Positive: {100*y.mean():.1f}%")

    # Train/test feature drift check
    _tr_mean = X_train_final.mean()
    _tr_std = X_train_final.std().clip(1e-8)
    _te_mean = X_test_final.mean()
    _drift_z = ((_te_mean - _tr_mean) / _tr_std).abs().sort_values(ascending=False)
    _drifted = _drift_z[_drift_z > 1.0]
    if len(_drifted):
        print(f"  Train/test drift warning ({len(_drifted)} features with |z|>1):")
        for _feat, _z in _drifted.head(10).items():
            print(f"    {_feat:40s}  z={_z:+.2f}  "
                  f"train_μ={_tr_mean[_feat]:.4f}  test_μ={_te_mean[_feat]:.4f}")
    else:
        print("  Train/test drift: no features with |z|>1 — looks stable.")

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": list(X_train_final.columns),
        "categorical_cols_clean": cat_cols,
        "target_encoding_map": encoding_map,
        "global_mean": global_mean,
    }


def shap_feature_prefilter(data_prep, top_n=None):
    """Quick RF → SHAP → rank features → return reduced data_prep.

    Two-pass approach:
      Pass 1: Train a quick Random Forest on ALL features.
      Pass 2: Compute SHAP values → rank by mean |SHAP| → keep top N.

    Returns
    -------
    dict  – same structure as data_prep but with reduced X_train / X_test,
            plus 'shap_ranking' (DataFrame) and 'selected_features' (list).
    """
    import shap
    warnings.filterwarnings("ignore")

    top_n = top_n or config.SHAP_PREFILTER_TOP_N
    X_tr, X_te = data_prep["X_train"], data_prep["X_test"]
    y_tr, y_te = data_prep["y_train"], data_prep["y_test"]
    X_tr, X_te = _sanitize_features(X_tr, X_te, label="SHAP-prefilter")
    feature_cols = list(X_tr.columns)

    print(f"\n{'='*80}")
    print(f"SHAP FEATURE PRE-FILTER — RF → SHAP → top {top_n} of {len(feature_cols)}")
    print(f"{'='*80}\n")

    # Pass 1: Quick Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=2,
    )
    rf.fit(X_tr, y_tr)
    rf_auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    print(f"  RF (all {len(feature_cols)} features): AUC = {rf_auc:.4f}")

    # Pass 2: SHAP values (TreeExplainer — fast for RF)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer(X_te)
    vals = shap_values
    if hasattr(vals, "values") and vals.values.ndim == 3:
        vals = vals[..., 1]  # positive-class slice

    mean_abs_shap = np.abs(vals.values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    selected = ranking.head(top_n)["feature"].tolist()
    dropped = ranking.tail(len(feature_cols) - top_n)["feature"].tolist()

    print(f"\n  Top {top_n} features (by mean |SHAP|):")
    for i, row in ranking.head(top_n).iterrows():
        print(f"    {i+1:2d}. {row['feature']:<30s} {row['mean_abs_shap']:.4f}")
    print(f"\n  Dropped {len(dropped)} low-SHAP features: "
          f"{dropped[:5]}{'…' if len(dropped) > 5 else ''}")

    # Verify with reduced RF
    rf2 = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=2,
    )
    rf2.fit(X_tr[selected], y_tr)
    rf2_auc = roc_auc_score(y_te, rf2.predict_proba(X_te[selected])[:, 1])
    print(f"\n  RF (top {top_n} features): AUC = {rf2_auc:.4f}  "
          f"(delta: {rf2_auc - rf_auc:+.4f})")
    del rf, rf2, explainer, shap_values; gc.collect()

    print(f"  Features: {len(feature_cols)} → {len(selected)}")
    return {
        **data_prep,
        "X_train": X_tr[selected],
        "X_test": X_te[selected],
        "feature_cols": selected,
        "shap_ranking": ranking,
        "selected_features": selected,
        "dropped_features": dropped,
        "all_feature_cols": feature_cols,
    }


def visualize_binary(data_prep):
    """Visualise binary feature engineering: class balance, correlations, SHAP ranking."""
    X_tr, y_tr = data_prep["X_train"], data_prep["y_train"]

    # 1 — Target class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = y_tr.value_counts()
    ax.bar(vc.index.astype(str), vc.values, color=["#4c72b0", "#dd8452"])
    ax.set_title("Binary target distribution (train)")
    ax.set_ylabel("Count")
    plt.tight_layout(); plt.show()

    # 2 — Feature correlation heatmap (top 20 features)
    top_cols = list(X_tr.columns[:20])
    corr = X_tr[top_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Feature correlation (top 20)")
    plt.tight_layout(); plt.show()

    # 3 — SHAP ranking bar chart (if SHAP pre-filter was run)
    if "shap_ranking" in data_prep:
        ranking = data_prep["shap_ranking"]
        top = ranking.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                color="#4c72b0")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title("Top 20 features by SHAP importance")
        plt.tight_layout(); plt.show()

    print("  ✓ Binary FE visualisation complete")



# ══════════════════════════════════════════════════════════════
# SECTION 7 — Feature Engineering: Multiclass Classification
# ══════════════════════════════════════════════════════════════

def prepare_multiclass(df_multi):
    """
    Prepare features for multiclass retrofit-type prediction.

    Pipeline:
      1. Group rare multi-type combinations into primary type
      2. Filter classes with too few samples
      3. select_features (shared)
      4. Fill NaN
      5. Train / test split (stratified)
      6. Label encoding
      7. Smoothed target encoding + frequency encoding
      8. Combine all encoded features
      9. RF importance pre-filter → top 50

    Returns dict consumed by 04_data_modeling_and_evaluation_multiclass.py.
    """
    target = config.COL_MULTICLASS_TARGET

    if target not in df_multi.columns:
        print(f"ERROR: target column '{target}' not found.")
        print(f"  Available _clean cols: "
              f"{[c for c in df_multi.columns if c.endswith('_clean')]}")
        return None

    df = df_multi.copy()

    # ── 0. Drop rows with missing target ──────────────────
    _before = len(df)
    df = df[df[target].notna() & (df[target].astype(str) != "nan") & (df[target].astype(str).str.strip() != "")]
    _dropped = _before - len(df)
    if _dropped:
        print(f"  Dropped {_dropped:,} rows with missing/empty target '{target}'")

    # ── 1. Simplify rare multi-type combinations ─────────────
    # Map each rare combo → its rarest standalone component
    # (avoids inflating already-large categories)
    _MIN_COMBO_FREQ = max(50, int(0.005 * len(df)))
    counts_raw = df[target].value_counts()

    # Standalone frequencies: count only non-combo occurrences
    _standalone = {
        cls: cnt for cls, cnt in counts_raw.items()
        if isinstance(cls, str) and ", " not in cls
    }

    _remap = {}
    for cls, cnt in counts_raw.items():
        if cnt < _MIN_COMBO_FREQ and isinstance(cls, str) and ", " in cls:
            parts = [p.strip() for p in cls.split(", ")]
            # Pick the component with the smallest standalone count
            best = min(parts, key=lambda p: _standalone.get(p, 0))
            _remap[cls] = best
    if _remap:
        df[target] = df[target].replace(_remap)
        print(f"  Grouped {len(_remap)} rare multi-type combos "
              f"into rarest component:")
        for old, new in sorted(_remap.items()):
            print(f"    {old} ({counts_raw[old]:,}) → {new}")

    # ── 2. Filter rare classes ───────────────────────────────
    counts = df[target].value_counts()
    min_n = max(10, int(0.001 * len(df)))
    common = counts[counts >= min_n].index.tolist()
    df = df[df[target].isin(common)]
    print(f"Samples: {len(df):,} | Retrofit types kept: {len(common)}")
    print(f"  Types: {common}")

    # ── 3. Select features ───────────────────────────────────
    cat_cols, num_cols = select_features(df, target)
    cat_cols = [c for c in cat_cols if c != target]
    feat_cols = cat_cols + num_cols

    if not feat_cols:
        print("ERROR: no features selected"); return None

    X, y = df[feat_cols].copy(), df[target]

    # ── 4. Fill NaN ──────────────────────────────────────────
    for c in cat_cols:
        X[c] = X[c].fillna(config.FILLNA_CATEGORICAL)
    for c in num_cols:
        _med = X[c].median()
        X[c] = X[c].fillna(_med if pd.notna(_med) else 0)

    # ── 5. Train / test split ────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )

    # ── 6. Label encoding ────────────────────────────────────
    le = LabelEncoder()
    y_tr_num = pd.Series(le.fit_transform(y_tr), index=X_tr.index)
    y_te_num = le.transform(y_te)

    # ── 7. Smoothed target encoding + frequency encoding ─────
    X_tr_enc, X_te_enc, enc_map, g_mean = _smoothed_target_encode(
        X_tr, X_te, y_tr_num, cat_cols)
    X_tr_freq, X_te_freq, freq_map = _frequency_encode(X_tr, X_te, cat_cols)

    # ── 8. Combine all features ──────────────────────────────
    num_present = [c for c in num_cols if c in X_tr.columns]
    X_tr_f = pd.concat([X_tr_enc, X_tr_freq,
                         X_tr[num_present].astype("float32")], axis=1)
    X_te_f = pd.concat([X_te_enc, X_te_freq,
                         X_te[num_present].astype("float32")], axis=1)
    X_tr_f.replace([np.inf, -np.inf], 0, inplace=True)
    X_te_f.replace([np.inf, -np.inf], 0, inplace=True)

    n_classes = len(common)
    print(f"  Features: {len(cat_cols)} target-enc + "
          f"{len(X_tr_freq.columns)} freq-enc + "
          f"{len(num_present)} num = {X_tr_f.shape[1]} total")

    # ── 9. RF importance pre-filter (top 50) ─────────────────
    _scout = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _scout.fit(X_tr_f, y_tr_num)
    _importances = pd.Series(_scout.feature_importances_, index=X_tr_f.columns)
    _top_n = min(50, len(_importances))
    _top_feats = _importances.nlargest(_top_n).index.tolist()
    print(f"  Feature pre-filter: kept top {len(_top_feats)} of {X_tr_f.shape[1]} "
          f"(min imp: {_importances[_top_feats[-1]]:.4f})")
    X_tr_f = X_tr_f[_top_feats]
    X_te_f = X_te_f[_top_feats]
    del _scout; gc.collect()

    print(f"  Train: {len(X_tr_f):,} | Test: {len(X_te_f):,} | "
          f"Classes: {n_classes} | Features: {X_tr_f.shape[1]}")

    return {
        "X_train": X_tr_f,
        "X_test": X_te_f,
        "X_train_raw": X_tr,
        "X_test_raw": X_te,
        "y_train": y_tr,
        "y_test": y_te,
        "y_train_num": y_tr_num,
        "y_test_num": y_te_num,
        "label_encoder": le,
        "feature_cols": _top_feats,
        "categorical_cols_clean": cat_cols,
        "num_cols": num_present,
        "encoding_map": enc_map,
        "global_mean": g_mean,
        "freq_map": freq_map,
        "n_classes": n_classes,
        "class_names": common,
        "rf_importances": _importances,
    }


def visualize_multiclass(data_prep):
    """Visualise multiclass feature engineering: class balance, RF importance, correlations."""
    X_tr = data_prep["X_train"]
    y_tr = data_prep["y_train"]

    # 1 — Class distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = y_tr.value_counts()
    ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color="#4c72b0")
    ax.set_xlabel("Count")
    ax.set_title("Multiclass target distribution (train)")
    plt.tight_layout(); plt.show()

    # 2 — RF feature importance (top 20)
    if "rf_importances" in data_prep:
        imp = data_prep["rf_importances"].nlargest(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp.index[::-1], imp.values[::-1], color="#dd8452")
        ax.set_xlabel("RF Importance")
        ax.set_title("Top 20 features (RF importance pre-filter)")
        plt.tight_layout(); plt.show()

    # 3 — Feature correlation heatmap (top 20 selected)
    top_cols = list(X_tr.columns[:20])
    corr = X_tr[top_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Feature correlation (top 20 selected)")
    plt.tight_layout(); plt.show()

    print("  ✓ Multiclass FE visualisation complete")

# ══════════════════════════════════════════════════════════════
# SECTION 8 — Feature Engineering: Temporal Regression
# ══════════════════════════════════════════════════════════════

def _create_duration_target(df):
    """Target: retrofit duration in days (umr_av − umr_start)."""
    end = pd.to_datetime(df[config.COL_AV_UR], errors="coerce").dt.tz_localize(None)
    start = pd.to_datetime(df[config.COL_START_UR], errors="coerce").dt.tz_localize(None)
    df[config.COL_RETROFIT_DURATION] = (end - start).dt.days.astype("float32")

    # Remove negatives first, then clip to 1st–99th percentile
    df = df[df[config.COL_RETROFIT_DURATION] >= 0].copy()
    p01 = df[config.COL_RETROFIT_DURATION].quantile(0.01)
    p99 = df[config.COL_RETROFIT_DURATION].quantile(0.99)
    before = len(df)
    df = df[df[config.COL_RETROFIT_DURATION].between(p01, p99)].copy()
    print(f"  Duration: kept P1–P99 [{p01:.0f}–{p99:.0f} d], "
          f"removed {before - len(df):,} outliers")
    return df


def prepare_temporal(df_multi):
    """
    Prepare features for temporal regression (retrofit duration prediction).

    Pipeline:
      1. Create duration target (umr_av − umr_start)
      2. select_features minus _TEMPORAL_LEAK columns
      3. Fill NaN
      4. Add group-level historical stats (type median, derivat median)
      5. Train / test split
      6. Compute group stats from train only (no leakage)
      7. log1p transform target
      8. Smoothed target encoding + frequency encoding
      9. RF importance pre-filter → top 40
     10. Add interaction features (top 10 numeric × pairwise)

    Returns dict consumed by 05_data_modeling_and_evaluation_temporal.py /
    06_data_modeling_and_evaluation_time_series.py.
    """
    target_col = config.COL_RETROFIT_DURATION

    # ── 1. Create duration target ────────────────────────────
    df = _create_duration_target(df_multi.copy())
    if df is None or target_col not in df.columns or len(df) == 0:
        print("ERROR: could not create duration target"); return None

    y_raw = df[target_col]
    print(f"  Raw stats: mean={y_raw.mean():.1f} d, median={y_raw.median():.1f}, "
          f"std={y_raw.std():.1f}  |  range: {y_raw.min():.0f}–{y_raw.max():.0f} d")

    # ── 2. Select features minus temporal leak ───────────────
    cat_cols, num_cols = select_features(df, target_col)
    cat_cols = [c for c in cat_cols if c not in _TEMPORAL_LEAK
                and c.removesuffix("_clean") not in _TEMPORAL_LEAK]
    num_cols = [c for c in num_cols if c not in _TEMPORAL_LEAK]
    feat_cols = cat_cols + num_cols

    if not feat_cols:
        print("ERROR: no features selected"); return None
    print(f"  Features: {len(cat_cols)} cat + {len(num_cols)} num = {len(feat_cols)}")

    X = df[feat_cols].copy()

    # ── 3. Fill NaN ──────────────────────────────────────────
    for c in cat_cols:
        X[c] = X[c].fillna(config.FILLNA_CATEGORICAL)
    for c in num_cols:
        _med = X[c].median()
        X[c] = X[c].fillna(_med if pd.notna(_med) else 0)

    # ── 4. Placeholder columns for group stats ───────────────
    _group_features = []
    _type_col = config.COL_MULTICLASS_TARGET
    _deriv_col = config.COL_DERIVAT_CLEAN if config.COL_DERIVAT_CLEAN in df.columns else config.COL_DERIVAT

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

    # ── 5. Train / test split ────────────────────────────────
    y_log = np.log1p(y_raw)
    X_tr, X_te, y_tr_log, y_te_log = train_test_split(
        X, y_log, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # ── 6. Fill group stats from train only (no leakage) ─────
    _y_tr_raw = np.expm1(y_tr_log)
    _overall_med = float(_y_tr_raw.median())
    _group_maps = {}

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

    y_te_raw = np.expm1(y_te_log)

    # ── 7-8. Smoothed target encoding + frequency encoding ───
    Xtr_enc, Xte_enc, enc_map, g_mean = _smoothed_target_encode(
        X_tr, X_te, y_tr_log, cat_cols)
    X_tr_freq, X_te_freq, freq_map = _frequency_encode(X_tr, X_te, cat_cols)

    freq_cols = [f"{c}_freq" for c in cat_cols]
    num_present = [c for c in num_cols if c in X_tr.columns]
    X_tr_f = pd.concat([Xtr_enc, X_tr_freq, X_tr[num_present].astype("float32")], axis=1)
    X_te_f = pd.concat([Xte_enc, X_te_freq, X_te[num_present].astype("float32")], axis=1)
    X_tr_f.replace([np.inf, -np.inf], 0, inplace=True)
    X_te_f.replace([np.inf, -np.inf], 0, inplace=True)

    # ── 9. RF importance pre-filter (top 40) ─────────────────
    _scout = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        random_state=config.RANDOM_STATE, n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _scout.fit(X_tr_f, y_tr_log)
    importances = pd.Series(_scout.feature_importances_, index=X_tr_f.columns)
    top_n = min(40, len(importances))
    top_feats = importances.nlargest(top_n).index.tolist()
    print(f"  Feature pre-filter: kept top {len(top_feats)} of {X_tr_f.shape[1]} "
          f"(min imp: {importances[top_feats[-1]]:.4f})")
    X_tr_f = X_tr_f[top_feats]
    X_te_f = X_te_f[top_feats]

    # ── 10. Interaction features ─────────────────────────────
    top_num = [c for c in importances.nlargest(10).index if c in num_present]
    X_tr_f, ix_pairs = _add_interactions(X_tr_f, top_num, max_pairs=15)
    X_te_f, _ = _add_interactions(X_te_f, top_num, max_pairs=15)
    all_feats = X_tr_f.columns.tolist()
    print(f"  After interactions: {len(all_feats)} features "
          f"({len(ix_pairs)} interaction pairs)")
    del _scout; gc.collect()

    print(f"  Train: {len(X_tr_f):,} | Test: {len(X_te_f):,} | Features: {X_tr_f.shape[1]}")

    # Train/test feature drift check
    _tr_mean = X_tr_f.mean()
    _tr_std = X_tr_f.std().clip(1e-8)
    _te_mean = X_te_f.mean()
    _drift_z = ((_te_mean - _tr_mean) / _tr_std).abs().sort_values(ascending=False)
    _drifted = _drift_z[_drift_z > 1.0]
    if len(_drifted):
        print(f"  Train/test drift warning ({len(_drifted)} features with |z|>1):")
        for _feat, _z in _drifted.head(10).items():
            print(f"    {_feat:40s}  z={_z:+.2f}  "
                  f"train_μ={_tr_mean[_feat]:.4f}  test_μ={_te_mean[_feat]:.4f}")
    else:
        print("  Train/test drift: no features with |z|>1 — looks stable.")

    return {
        "X_train": X_tr_f,
        "X_test": X_te_f,
        "y_train_log": y_tr_log,
        "y_test_log": y_te_log,
        "y_test_raw": y_te_raw,
        "feature_cols": all_feats,
        "categorical_cols_clean": cat_cols,
        "num_cols": num_present,
        "encoding_map": enc_map,
        "global_mean": g_mean,
        "freq_map": freq_map,
        "group_maps": _group_maps,
        "overall_median": _overall_med,
        "rf_importances": importances,
        "interaction_pairs": ix_pairs,
        "target_col": target_col,
        "df_dur": df,
    }


def visualize_temporal(data_prep):
    """Visualise temporal FE: duration distribution, RF importance, interactions."""
    y_raw = data_prep["y_test_raw"]

    # 1 — Duration distribution (raw days)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_raw, bins=50, color="#4c72b0", edgecolor="white")
    axes[0].set_title("Retrofit duration (days) — test set")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Count")
    axes[1].hist(np.log1p(y_raw), bins=50, color="#dd8452", edgecolor="white")
    axes[1].set_title("log1p(duration) — test set")
    axes[1].set_xlabel("log1p(days)")
    plt.tight_layout(); plt.show()

    # 2 — RF feature importance (top 20)
    if "rf_importances" in data_prep:
        imp = data_prep["rf_importances"].nlargest(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp.index[::-1], imp.values[::-1], color="#55a868")
        ax.set_xlabel("RF Importance")
        ax.set_title("Top 20 features (RF pre-filter)")
        plt.tight_layout(); plt.show()

    # 3 — Correlation heatmap (top 20)
    X_tr = data_prep["X_train"]
    top_cols = list(X_tr.columns[:20])
    corr = X_tr[top_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Feature correlation (top 20)")
    plt.tight_layout(); plt.show()

    print("  ✓ Temporal FE visualisation complete")


# ══════════════════════════════════════════════════════════════
# SECTION 9 — Feature Engineering: Time Series
# ══════════════════════════════════════════════════════════════

def _prepare_monthly_series(df, target_col, date_col=None):
    """Aggregate raw rows to monthly counts / mean target values."""
    if date_col is None:
        date_col = config.COL_START_UR
    df = df.dropna(subset=[date_col, target_col]).copy()
    df["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M")
    monthly = df.groupby("_month")[target_col].agg(["count", "mean"]).reset_index()
    monthly.columns = [config.COL_TS_MONTH, config.COL_TS_COUNT, config.COL_MEAN_TARGET]
    monthly[config.COL_TS_MONTH] = monthly[config.COL_TS_MONTH].dt.to_timestamp()
    monthly = monthly.sort_values(config.COL_TS_MONTH).reset_index(drop=True)
    return monthly


def _classify_demand(series):
    """Classify demand pattern using ADI / CV² (Syntetos-Boylan framework)."""
    non_zero = series[series > 0]
    if len(non_zero) < 2:
        return "Lumpy"
    intervals = []
    last_idx = None
    for i, val in enumerate(series):
        if val > 0:
            if last_idx is not None:
                intervals.append(i - last_idx)
            last_idx = i
    adi = np.mean(intervals) if intervals else float("inf")
    cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() != 0 else 0
    if adi < 1.32 and cv2 < 0.49:
        return "Smooth"
    elif adi < 1.32:
        return "Erratic"
    elif cv2 < 0.49:
        return "Intermittent"
    else:
        return "Lumpy"


def _create_lag_features(series, n_lags=12):
    """Turn a 1-D time-series into supervised learning with lag features."""
    df = pd.DataFrame({"y": series.values})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y


def prepare_time_series(df, target_col, date_col=None, n_lags=12):
    """
    Prepare data for all time-series model families (classical, ML, DL, FM).

    Pipeline:
      1. Aggregate raw rows → monthly series (count + mean_target)
      2. Classify demand pattern (Syntetos-Boylan)
      3. Create lag features for ML/DL regressors
      4. 80/20 chronological split (no shuffle — temporal order matters)

    Returns dict consumed by 07/08/09/10.
    """
    if date_col is None:
        date_col = config.COL_START_UR

    # ── 1. Monthly aggregation ───────────────────────────────
    monthly = _prepare_monthly_series(df, target_col, date_col)
    print(f"  Monthly series: {len(monthly)} points "
          f"({monthly[config.COL_TS_MONTH].min():%Y-%m} → {monthly[config.COL_TS_MONTH].max():%Y-%m})")
    print(f"  Count — mean: {monthly[config.COL_TS_COUNT].mean():.1f}, "
          f"std: {monthly[config.COL_TS_COUNT].std():.1f}")
    print(f"  Target — mean: {monthly[config.COL_MEAN_TARGET].mean():.3f}, "
          f"std: {monthly[config.COL_MEAN_TARGET].std():.3f}")

    # ── 2. Demand classification ─────────────────────────────
    demand_class = _classify_demand(monthly[config.COL_TS_COUNT])
    print(f"  Demand pattern: {demand_class}")

    # ── 3. Chronological train/test split ────────────────────
    split_idx = int(len(monthly) * 0.8)
    train_monthly = monthly.iloc[:split_idx].copy()
    test_monthly = monthly.iloc[split_idx:].copy()
    print(f"  Split: {len(train_monthly)} train | {len(test_monthly)} test months")

    # ── 4. Lag features for ML/DL ────────────────────────────
    series = monthly[config.COL_MEAN_TARGET]
    lag_X = lag_y = None
    if len(series) >= n_lags + 6:
        lag_X, lag_y = _create_lag_features(series, n_lags)
        lag_split = max(int(len(lag_y) * 0.8), n_lags)
        print(f"  Lag features: {lag_X.shape[1]} lags, "
              f"{lag_split} train / {len(lag_y) - lag_split} test samples")
    else:
        print(f"  ⚠ Not enough data for lag features "
              f"(need {n_lags + 6}, have {len(series)})")

    return {
        "monthly": monthly,
        "train_monthly": train_monthly,
        "test_monthly": test_monthly,
        "split_idx": split_idx,
        "demand_class": demand_class,
        "target_col": target_col,
        "date_col": date_col,
        "n_lags": n_lags,
        "lag_X": lag_X,
        "lag_y": lag_y,
    }


def visualize_time_series(data_prep):
    """Visualise time-series FE: monthly trend, demand class, lag ACF."""
    monthly = data_prep["monthly"]
    split_idx = data_prep["split_idx"]
    target_col = data_prep["target_col"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1 — Monthly count over time
    ax = axes[0, 0]
    ax.plot(monthly[config.COL_TS_MONTH], monthly[config.COL_TS_COUNT], marker="o", ms=3)
    ax.axvline(monthly[config.COL_TS_MONTH].iloc[split_idx], color="red",
               ls="--", label="train/test split")
    ax.set_title("Monthly retrofit count")
    ax.set_xlabel("Month"); ax.set_ylabel("Count"); ax.legend()

    # 2 — Mean target value over time
    ax = axes[0, 1]
    ax.plot(monthly[config.COL_TS_MONTH], monthly[config.COL_MEAN_TARGET], marker="o",
            ms=3, color="#dd8452")
    ax.axvline(monthly[config.COL_TS_MONTH].iloc[split_idx], color="red", ls="--")
    ax.set_title(f"Monthly mean {target_col}")
    ax.set_xlabel("Month"); ax.set_ylabel("Mean value")

    # 3 — Distribution of monthly counts
    ax = axes[1, 0]
    ax.hist(monthly[config.COL_TS_COUNT], bins=20, color="#55a868", edgecolor="white")
    ax.set_title(f"Count distribution — {data_prep['demand_class']} demand")
    ax.set_xlabel("Monthly count"); ax.set_ylabel("Frequency")

    # 4 — Lag autocorrelation
    ax = axes[1, 1]
    series = monthly[config.COL_MEAN_TARGET]
    max_lag = min(24, len(series) - 1)
    if max_lag > 1:
        acf_vals = [series.autocorr(lag=k) for k in range(1, max_lag + 1)]
        ax.bar(range(1, max_lag + 1), acf_vals, color="#4c72b0")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("Autocorrelation (mean_target)")
        ax.set_xlabel("Lag (months)"); ax.set_ylabel("ACF")
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Autocorrelation")

    plt.tight_layout(); plt.show()
    print("  ✓ Time-series FE visualisation complete")
