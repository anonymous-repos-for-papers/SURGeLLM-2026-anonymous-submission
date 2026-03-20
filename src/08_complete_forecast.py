# ============================================================
# 08_complete_forecast.py - Forecast, Backtest & CSV Export
#
#   Section 1  evaluate_retrofits()  - predicted-vs-actual backtest
#              -> produces calibration_factors
#   Section 2  run_forecast()        - full forecast pipeline
#              Binary -> Multiclass -> Temporal -> Month x Type pivot
#   Section 3  export_forecast()     - export to CSV
# ============================================================
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .utils import mem


def _build_X_temporal(df_will, model_dict):
    """Build feature matrix for a temporal model (duration or lead-time).

    Parameters
    ----------
    df_will : DataFrame with forecast vehicles
    model_dict : sub-dict from temporal_results (e.g. temporal_results["duration"])

    Returns
    -------
    pd.DataFrame  ready for model.predict()
    """
    d_enc = model_dict["encoding_map"]
    d_gmean = model_dict["global_mean"]
    d_cat = model_dict["cat_cols"]
    d_num = model_dict["num_cols"]
    d_feats = model_dict.get("top_feats", d_cat + d_num)
    d_freq = model_dict.get("freq_map", {})          # frequency encoding maps

    # Pre-compute freq column set for O(1) lookup
    freq_col_set = {f"{c}_freq" for c in d_cat} if d_freq else set()

    # Interaction features
    interaction_cols = {}
    for a, b, col_name in model_dict.get("interaction_pairs", []):
        if a in df_will.columns and b in df_will.columns:
            interaction_cols[col_name] = (
                df_will[a].fillna(0).astype("float32")
                * df_will[b].fillna(0).astype("float32")
            ).to_numpy(dtype=np.float32)

    # Group-stat features (type_hist_median, derivat_hist_median, etc.)
    _gm = model_dict.get("group_maps", {})
    _otm = model_dict.get("overall_target_median", 7.0)
    group_cols = {}
    if "type_median" in _gm:
        group_cols[config.COL_TYPE_HIST_MEDIAN] = (
            df_will[config.COL_PREDICTED_TYPE].map(_gm["type_median"])
            .fillna(_otm).astype("float32").values)
    if "type_count" in _gm:
        group_cols[config.COL_TYPE_HIST_COUNT] = (
            df_will[config.COL_PREDICTED_TYPE].map(_gm["type_count"])
            .fillna(0).astype("float32").values)
    _dcol_f = config.COL_DERIVAT_CLEAN if config.COL_DERIVAT_CLEAN in df_will.columns else config.COL_DERIVAT
    if "derivat_median" in _gm and _dcol_f in df_will.columns:
        group_cols[config.COL_DERIVAT_HIST_MEDIAN] = (
            df_will[_dcol_f].map(_gm["derivat_median"])
            .fillna(_otm).astype("float32").values)

    # Build numpy array
    n = len(df_will)
    X_arr = np.zeros((n, len(d_feats)), dtype=np.float32)
    for i, c in enumerate(d_feats):
        if c in group_cols:
            X_arr[:, i] = group_cols[c]
        elif c in interaction_cols:
            X_arr[:, i] = interaction_cols[c]
        elif c in freq_col_set:
            # Frequency-encoded categorical (same logic as multiclass step)
            orig_col = c.replace("_freq", "")
            fmap = d_freq.get(orig_col, {})
            if orig_col in df_will.columns and fmap:
                X_arr[:, i] = (df_will[orig_col].map(fmap)
                               .fillna(0).to_numpy(dtype=np.float32))
            else:
                X_arr[:, i] = 0
        elif c in d_cat:
            if c in df_will.columns:
                X_arr[:, i] = (df_will[c].map(d_enc.get(c, {}))
                               .fillna(d_gmean).to_numpy(dtype=np.float32))
            else:
                X_arr[:, i] = d_gmean
        else:  # numeric
            if c in df_will.columns:
                X_arr[:, i] = df_will[c].fillna(0).to_numpy(dtype=np.float32)
            else:
                X_arr[:, i] = 0

    return pd.DataFrame(X_arr, columns=d_feats)


def _optimize_blend_weights(dur_model, dur_type, dur_ts=None, y_true=None,
                            step=0.1):
    """Grid-search blend weights (model, type-median, TS) minimizing MAE.

    Tries all weight combos in [0, 1] summing to 1.0,
    with the given step size. If y_true is None or too short,
    falls back to config defaults.

    Returns (w_model, w_type, w_ts) for triple or (w_model, w_type, 0) for dual.
    """
    if y_true is None or len(y_true) < 20:
        if dur_ts is not None:
            return config.BLEND_W_MODEL, config.BLEND_W_TYPE_MEDIAN, config.BLEND_W_TS
        return config.BLEND_W_DUAL, 1.0 - config.BLEND_W_DUAL, 0.0

    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(dur_model, dtype=np.float64)
    t = np.asarray(dur_type, dtype=np.float64)

    best_w, best_mae = None, np.inf
    steps = np.arange(0, 1.0 + step / 2, step)

    if dur_ts is not None:
        ts = np.asarray(dur_ts, dtype=np.float64)
        for w_m in steps:
            for w_t in steps:
                w_s = round(1.0 - w_m - w_t, 4)
                if w_s < -1e-9 or w_s > 1.0 + 1e-9:
                    continue
                pred = w_m * m + w_t * t + max(w_s, 0) * ts
                mae = float(np.mean(np.abs(y - pred)))
                if mae < best_mae:
                    best_mae = mae
                    best_w = (round(w_m, 2), round(w_t, 2), round(max(w_s, 0), 2))
    else:
        for w_m in steps:
            w_t = round(1.0 - w_m, 4)
            pred = w_m * m + w_t * t
            mae = float(np.mean(np.abs(y - pred)))
            if mae < best_mae:
                best_mae = mae
                best_w = (round(w_m, 2), round(w_t, 2), 0.0)

    return best_w


def _predict_best_temporal(df_will, model_dict, label=""):
    """
    Auto-select the best temporal model (individual OR stacked ensemble)
    and produce per-vehicle predictions.

    If `best_overall` is "Stacked Ensemble", it feeds per-vehicle X through
    ALL base models → then through the Ridge meta-learner.  Otherwise it
    uses the single best individual model.

    Returns
    -------
    pred_days : np.ndarray  – predicted values in days (un-logged)
    chosen    : str         – name of the model actually used
    """
    best_overall = model_dict.get("best_overall", model_dict["best_model"])
    best_ind = model_dict["best_model"]

    # Compare MAEs to decide which to use
    res = model_dict["results"]
    mae_ind = res[best_ind]["mae"]
    mae_overall = res.get(best_overall, {}).get("mae", mae_ind)

    if best_overall == "Stacked Ensemble" and mae_overall <= mae_ind:
        # Use stacked ensemble: predict with all base models → meta
        meta = model_dict.get("meta_model")
        final_models = model_dict.get("final_models", {})
        base_names = model_dict.get("base_names", [])

        if meta is not None and final_models and base_names:
            X_pred = _build_X_temporal(df_will, model_dict)
            base_preds = {}
            for name in base_names:
                if name in final_models:
                    base_preds[name] = final_models[name].predict(X_pred)
                else:
                    # Fallback: zero-fill missing base model
                    base_preds[name] = np.zeros(len(X_pred))

            S = np.column_stack([base_preds[n] for n in base_names])
            pred_log = meta.predict(S)
            if model_dict.get("log_transformed"):
                pred_days = np.expm1(pred_log).clip(0)
            else:
                pred_days = pred_log.clip(0)

            print(f"  {label}: using Stacked Ensemble "
                  f"(MAE={mae_overall:.1f}d vs best individual "
                  f"{best_ind} MAE={mae_ind:.1f}d)")
            return pred_days, "Stacked Ensemble"

    # Fallback: use best individual model
    chosen = best_overall if best_overall != "Stacked Ensemble" else best_ind
    m = res[chosen].get("model")
    if m is None:
        # Model not stored in results — check final_models
        m = model_dict.get("final_models", {}).get(chosen)
    if m is None:
        # Last resort: use best_ind
        chosen = best_ind
        m = res[chosen].get("model")
        if m is None:
            m = model_dict.get("final_models", {}).get(chosen)

    if m is None:
        print(f"  ✗ {label}: no trained model found for '{chosen}' — "
              f"falling back to type-median only")
        return None, None

    X_pred = _build_X_temporal(df_will, model_dict)
    pred_log = m.predict(X_pred)
    if model_dict.get("log_transformed"):
        pred_days = np.expm1(pred_log).clip(0)
    else:
        pred_days = pred_log.clip(0)

    print(f"  {label}: using {chosen} (MAE={res.get(chosen, {}).get('mae', 0):.1f}d)")
    return pred_days, chosen


# ── Quality gate thresholds for TS blending ──────────────────
_TS_MIN_R2   = config.TS_MIN_R2
_TS_MAX_MAPE = config.TS_MAX_MAPE


def _ts_is_reasonable(
    ts_comparison: "pd.DataFrame | None",
    overall_median: float,
    label: str = "",
) -> bool:
    """Return True if the best TS model's metrics justify blending.

    Checks two conditions:
      1. R² > ``_TS_MIN_R2``  (explains more variance than the mean)
      2. MAE / overall_median < ``_TS_MAX_MAPE``  (relative error bounded)

    Prints a one-line verdict so the log is transparent.
    """
    if ts_comparison is None or ts_comparison.empty:
        print(f"  TS quality gate ({label}): no comparison table → SKIP")
        return False

    best = ts_comparison.iloc[0]          # sorted by MAE ascending
    r2   = float(best["R2"])
    mae  = float(best["MAE"])
    rel  = mae / max(abs(overall_median), 1e-6)

    ok = (r2 > _TS_MIN_R2) and (rel < _TS_MAX_MAPE)
    tag = "✓ PASS" if ok else "✗ FAIL"
    print(f"  TS quality gate ({label}): {best['Model']}  "
          f"R²={r2:.3f}  MAE={mae:.1f}  relMAE={rel:.1%}  → {tag}")
    return ok


def _map_vehicles_to_ts(
    df: pd.DataFrame,
    ts_future: "pd.Series | None",
    cfg,
) -> "np.ndarray | None":
    """Map each vehicle to its nearest TS monthly forecast value.

    Uses the vehicle's deadline (``rueckgabe_soll``) to determine the
    expected month, then looks up the corresponding classical-TS
    forecast for that month.  Vehicles without a valid date receive the
    TS forecast mean.

    Parameters
    ----------
    df : DataFrame with forecast vehicles (must contain ``rueckgabe_soll``).
    ts_future : pd.Series with ``DatetimeIndex`` (month-start) →
        predicted average value (days) for each month.
        Can be *None* if no classical TS model was trained.
    cfg : config module.

    Returns
    -------
    np.ndarray of shape ``(len(df),)`` or *None* when ``ts_future``
    is unavailable.
    """
    if ts_future is None or len(ts_future) == 0:
        return None

    ts_mean = float(ts_future.mean())
    n = len(df)
    ref_col = cfg.COL_RUECKGABE_SOLL

    # Fallback: no date column → broadcast TS mean
    if ref_col not in df.columns:
        return np.full(n, ts_mean, dtype=np.float32)

    ref_dates = pd.to_datetime(df[ref_col], errors="coerce")
    if ref_dates.isna().all():
        return np.full(n, ts_mean, dtype=np.float32)

    # Normalize to month-start for lookup
    ref_months = ref_dates.dt.to_period("M").dt.to_timestamp()

    # Build lookup: unique_month → nearest TS forecast value
    unique_months = ref_months.dropna().unique()
    lookup: dict = {}
    for m in unique_months:
        idx = ts_future.index.get_indexer([m], method="nearest")[0]
        lookup[m] = float(ts_future.iloc[idx]) if idx >= 0 else ts_mean

    result = ref_months.map(lookup).fillna(ts_mean).to_numpy(dtype=np.float32)
    return result


def _naive_baseline(multi_results, n_forecast_months=18):
    """
    Naive time-series baseline: "same as last N months" average.

    Uses the pre-computed historical monthly type pivot stored in
    multi_results["hist_monthly_pivot"] (computed in the notebook
    visit-stats cell).  Projects the average monthly counts flat
    for *n_forecast_months*.

    Returns
    -------
    dict  with 'pivot_naive' (Period-indexed DataFrame, same shape as
          the ML forecast pivot) and 'summary' string.
    """
    hist_pivot = multi_results.get("hist_monthly_pivot")
    if hist_pivot is None or hist_pivot.empty:
        print("  Naive baseline: skipped (no hist_monthly_pivot in multi_results)")
        return None

    # Average per type per month over the full history
    avg_per_month = hist_pivot.mean(axis=0)
    total_avg = float(avg_per_month.sum())
    n_months_used = len(hist_pivot)

    # Project forward: flat line for n_forecast_months
    now_period = pd.Period(pd.Timestamp.now(), freq="M")
    future_months = pd.period_range(now_period, periods=n_forecast_months, freq="M")

    pivot_naive = pd.DataFrame(
        {col: [round(avg_per_month.get(col, 0))] * len(future_months)
         for col in avg_per_month.index},
        index=future_months,
    ).fillna(0).astype(int)

    first_month = hist_pivot.index.min()
    last_month = hist_pivot.index.max()
    summary = (
        f"Naive baseline: avg of {n_months_used} months "
        f"({first_month}–{last_month}), "
        f"≈{total_avg:.0f} visits/month projected flat"
    )
    print(f"  {summary}")

    return {
        "pivot_naive": pivot_naive,
        "avg_per_month": avg_per_month.to_dict(),
        "total_avg_per_month": total_avg,
        "history_window": f"{first_month}–{last_month}",
        "n_months_used": n_months_used,
        "summary": summary,
    }


def run_forecast(df_binary, data_prep, binary_results, source_results,
                 multi_results, temporal_results, df_source_a_clean=None,
                 threshold=None, calibration_factors=None,
                 automl_results=None):
    """
    Apply all 3 trained models to unseen vehicles and produce
    a Month × Retrofit-Type forecast table.

    Parameters
    ----------
    df_binary       : full Source B DataFrame with came_for_retrofit column
    data_prep       : dict from prepare_binary (encoding maps etc.)
    binary_results  : dict from final_model_comparison (best binary model name)
    source_results  : dict from baseline/tuned/smote run (contains trained models)
    multi_results   : dict from multiclass_retrofit_types
    temporal_results: dict from temporal_prediction
    df_source_a_clean : cleaned Source A DataFrame (historical data). If provided, the
                      final output appends predicted rows to actual Source A data.
    threshold       : probability cutoff for binary prediction.
                      If None, uses the F1-optimal threshold from validation.
    calibration_factors : dict {type_name: float} from backtest. If provided,
                      each predicted visit gets a calibration_weight column
                      and the pivot shows calibrated counts alongside raw.
    automl_results  : dict from run_automl_benchmark() (module 07 automl).
                      If provided and AutoGluon/TabPFN beat the manual pipeline
                      for a given task, the winning model is used instead.

    Returns
    -------
    dict with df_forecast, pivot table, and summary stats
    """
    print("\n" + "=" * 80)
    print("FULL FORECAST PIPELINE")
    print("=" * 80 + "\n")

    # Resolve threshold: config override → user-supplied → F1-optimal
    if threshold is None:
        threshold = getattr(config, "FORECAST_THRESHOLD", None)
    if threshold is not None:
        opt = binary_results.get("optimal_threshold")
        print(f"  Using threshold={threshold:.3f} "
              f"(F1-optimal was {opt:.3f})" if opt else f"  Using threshold={threshold:.3f}")
    else:
        threshold = binary_results.get("optimal_threshold", 0.3)
        print(f"  Using F1-optimal threshold from validation: {threshold:.3f}")

    # ------------------------------------------------------------------
    # 1. Unseen vehicles (never retrofitted)
    #    Filter out old vehicles whose deadline has already passed —
    #    if they haven't come by now, they won't.
    # ------------------------------------------------------------------
    df_unseen = df_binary[df_binary[config.COL_CAME_FOR_RETROFIT] == 0].copy()
    n_all_unseen = len(df_unseen)

    cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
    anchor_col = None
    if config.COL_RUECKGABE_SOLL in df_unseen.columns:
        anchor_col = config.COL_RUECKGABE_SOLL
    elif config.COL_ANLAUF_SOP in df_unseen.columns:
        anchor_col = config.COL_ANLAUF_SOP

    if anchor_col is not None:
        dt = pd.to_datetime(df_unseen[anchor_col], errors="coerce")
        # Keep vehicles with future/recent date OR missing date (NaT)
        mask_keep = dt.isna() | (dt >= cutoff)
        df_unseen = df_unseen[mask_keep].copy()
        n_dropped = n_all_unseen - len(df_unseen)
        print(f"Step 1 — Unseen vehicles: {n_all_unseen:,} total, "
              f"dropped {n_dropped:,} with {anchor_col} before {cutoff.strftime('%Y-%m')}, "
              f"keeping {len(df_unseen):,}")
    else:
        print(f"Step 1 — Unseen vehicles: {n_all_unseen:,} "
              f"(no date column to filter old vehicles)")

    # ------------------------------------------------------------------
    # 2. BINARY: predict which will come
    # ------------------------------------------------------------------
    best_bin = binary_results["best_model"]
    bin_model = source_results["results"][best_bin]["model"]
    bin_enc = data_prep["target_encoding_map"]
    bin_gmean = data_prep["global_mean"]
    bin_cat = data_prep["categorical_cols_clean"]
    bin_feat = data_prep["feature_cols"]

    # LLM-Stacked / LLM-* are meta-learners that cannot run at inference.
    # Fall back to the best usable (non-LLM) model from the comparison.
    if best_bin.startswith("LLM"):
        _cmp = binary_results.get("comparison")
        _fallback = None
        if _cmp is not None:
            for _, _row in _cmp.iterrows():
                _mn = _row["Model"]
                if _mn.startswith("LLM") or _mn.startswith("Dummy"):
                    continue
                if _mn in source_results["results"] and "model" in source_results["results"][_mn]:
                    _fallback = _mn
                    break
        if _fallback is None:
            _ml = {k: v for k, v in source_results["results"].items()
                   if not k.startswith("LLM") and not k.startswith("Dummy") and "model" in v}
            if not _ml:
                raise RuntimeError("No usable (non-LLM, non-Dummy) binary model found")
            _fallback = max(_ml, key=lambda k: _ml[k].get("roc_auc", 0))
        print(f"  ★ {best_bin} cannot run at inference "
              f"→ using next-best model: {_fallback}")
        bin_model = source_results["results"][_fallback]["model"]

    # ── AutoML override: use best pipeline if it beat Manual ──
    if automl_results and automl_results.get("binary_comparison") is not None:
        _bin_winner = automl_results["binary_comparison"].iloc[0]["Pipeline"]
        _bin_auc = automl_results["binary_comparison"].iloc[0]["ROC-AUC"]
        if _bin_winner == "AutoGluon" and automl_results.get("binary", {}).get("model") is not None:
            bin_model = automl_results["binary"]["model"]
            print(f"  ★ AutoGluon won binary (ROC-AUC={_bin_auc:.4f}) → using AG model")
        elif _bin_winner == "TabPFN" and automl_results.get("tabpfn_binary", {}).get("model") is not None:
            bin_model = automl_results["tabpfn_binary"]["model"]
            print(f"  ★ TabPFN won binary (ROC-AUC={_bin_auc:.4f}) → using TabPFN model")

    # If source comes from Feature Selection, restrict to selected features
    sel_feats = source_results.get("selected_features")
    if sel_feats:
        bin_feat = sel_feats
        bin_cat = [c for c in bin_cat if c in sel_feats]
        print(f"  Using {len(sel_feats)} selected features (feature selection)")

    bin_num = [c for c in bin_feat if c not in bin_cat]

    X_bin = _encode(df_unseen, bin_cat, bin_num, bin_enc, bin_gmean)
    prob = bin_model.predict_proba(X_bin[bin_feat])[:, 1]
    df_unseen[config.COL_PROB_COMING] = prob

    df_will = df_unseen[df_unseen[config.COL_PROB_COMING] >= threshold].copy()
    print(f"Step 2 — Binary (prob >= {threshold}): {len(df_will):,} predicted to come")

    if df_will.empty:
        print("WARNING: no vehicles predicted to come. Try lowering the threshold.")
        return None

    n_vehicles_predicted = len(df_will)   # save before multi-visit expansion

    # ------------------------------------------------------------------
    # 3. MULTICLASS: predict top-K visit PACKAGES per vehicle
    # ------------------------------------------------------------------
    # Each class = a retrofit *package* (e.g. "Mech., Mech. SV").
    # A vehicle can come multiple times, each with a DIFFERENT package.
    # We predict the K most probable packages → K rows per vehicle,
    # each row = 1 future visit with its own package and staggered date.
    if multi_results and multi_results.get("best_model") is not None:
        best_multi = multi_results["best_model"]
        m_model = multi_results["results"][best_multi]["model"]
        le = multi_results["label_encoder"]

        # LLM-Stacked / LLM-* meta-learners use a different feature space
        # (stacked probabilities) and cannot run on the regular feature matrix.
        # Fall back to the best usable (non-LLM) model.
        if best_multi.startswith("LLM"):
            _cmp_m = multi_results.get("comparison")
            _fallback_m = None
            if _cmp_m is not None:
                for _, _row in _cmp_m.iterrows():
                    _mn = _row["Model"]
                    if _mn.startswith("LLM"):
                        continue
                    if _mn in multi_results["results"] and "model" in multi_results["results"][_mn]:
                        _fallback_m = _mn
                        break
            if _fallback_m is None:
                _ml_m = {k: v for k, v in multi_results["results"].items()
                         if not k.startswith("LLM") and "model" in v}
                if _ml_m:
                    _fallback_m = max(_ml_m, key=lambda k: _ml_m[k].get("f1_weighted", 0))
            if _fallback_m:
                print(f"  ★ {best_multi} cannot run at inference "
                      f"→ using next-best multiclass model: {_fallback_m}")
                m_model = multi_results["results"][_fallback_m]["model"]
                best_multi = _fallback_m

        # ── AutoML override: use best pipeline if it beat Manual ──
        if automl_results and automl_results.get("multiclass_comparison") is not None:
            _mc_winner = automl_results["multiclass_comparison"].iloc[0]["Pipeline"]
            _mc_f1w = automl_results["multiclass_comparison"].iloc[0]["F1_weighted"]
            if _mc_winner == "AutoGluon" and automl_results.get("multiclass", {}).get("model") is not None:
                m_model = automl_results["multiclass"]["model"]
                print(f"  ★ AutoGluon won multiclass (F1w={_mc_f1w:.4f}) → using AG model")
            elif _mc_winner == "TabPFN" and automl_results.get("tabpfn_multiclass", {}).get("model") is not None:
                m_model = automl_results["tabpfn_multiclass"]["model"]
                print(f"  ★ TabPFN won multiclass (F1w={_mc_f1w:.4f}) → using TabPFN model")

        m_enc = multi_results.get("encoding_map", {})
        m_gmean = float(multi_results.get("global_mean", 0))
        m_cat = multi_results.get("cat_cols", [])
        m_num = multi_results.get("num_cols", [])
        m_freq = multi_results.get("freq_map", {})   # v2 freq encoding
        freq_col_names = [f"{c}_freq" for c in m_cat] if m_freq else []
        all_m_feats = m_cat + freq_col_names + m_num

        if not all_m_feats:
            if hasattr(m_model, "feature_names_in_"):
                all_m_feats = list(m_model.feature_names_in_)
            elif hasattr(m_model, "n_features_in_"):
                all_m_feats = [f"f{i}" for i in range(m_model.n_features_in_)]
            m_cat = []
            m_num = all_m_feats
            print(f"  [WARN] cat_cols/num_cols not in multi_results — "
                  f"using model.feature_names_in_ ({len(all_m_feats)} feats)")

        # Build numpy array directly (avoids DataFrame dtype issues)
        n = len(df_will)
        X_arr = np.zeros((n, len(all_m_feats)), dtype=np.float32)
        n_avail = 0
        freq_col_set = set(freq_col_names)
        for i, c in enumerate(all_m_feats):
            if c in m_cat:
                # Target-encoded categorical
                if c in df_will.columns:
                    X_arr[:, i] = (df_will[c].map(m_enc.get(c, {}))
                                   .fillna(m_gmean).to_numpy(dtype=np.float32))
                    n_avail += 1
                else:
                    X_arr[:, i] = m_gmean
            elif c in freq_col_set:
                # Frequency-encoded categorical (v2)
                orig_col = c.replace("_freq", "")  # e.g. derivat_clean_freq → derivat_clean
                fmap = m_freq.get(orig_col, {})
                if orig_col in df_will.columns and fmap:
                    X_arr[:, i] = (df_will[orig_col].map(fmap)
                                   .fillna(0).to_numpy(dtype=np.float32))
                    n_avail += 1
                else:
                    X_arr[:, i] = 0
            else:
                # Numeric
                if c in df_will.columns:
                    vals = df_will[c]
                    if vals.dtype == object:
                        X_arr[:, i] = m_gmean
                    else:
                        X_arr[:, i] = vals.fillna(0).to_numpy(dtype=np.float32)
                    n_avail += 1
                else:
                    X_arr[:, i] = 0

        X_pred = pd.DataFrame(X_arr, columns=all_m_feats)

        # ── Filter to selected features (top-N from pre-filter) ─
        sel_feats = multi_results.get("selected_features")
        if sel_feats:
            X_pred = X_pred[[c for c in sel_feats if c in X_pred.columns]]
        elif hasattr(m_model, "feature_names_in_"):
            X_pred = X_pred[[c for c in m_model.feature_names_in_ if c in X_pred.columns]]

        # ── Per-derivat K: top-K packages based on historical visits ──
        visit_stats = multi_results.get("visit_stats", {})
        avg_visits_global = visit_stats.get("mean", 1.0)
        if avg_visits_global is None or (isinstance(avg_visits_global, float) and np.isnan(avg_visits_global)):
            avg_visits_global = 1.0
        per_derivat = visit_stats.get("per_derivat", {})
        dcol = visit_stats.get("derivat_col", config.COL_DERIVAT_CLEAN)
        print(f"  Global avg visits/vehicle: {avg_visits_global:.2f} "
              f"(used for unknown derivats)")

        # Compute K per vehicle: use derivat-specific avg, fallback to global
        if per_derivat and dcol in df_will.columns:
            K_per_veh = (
                df_will[dcol]
                .map(per_derivat)
                .fillna(avg_visits_global)       # derivat nuevo → media global
                .apply(np.ceil)
                .astype(int)
                .values
            )
            n_known = int((df_will[dcol].isin(per_derivat)).sum())
            n_unknown = len(df_will) - n_known
            print(f"Step 3 — Multiclass: per-derivat K, no cap (known: {n_known:,}, "
                  f"unknown→global: {n_unknown:,})")
        else:
            K_per_veh = np.full(len(df_will), int(np.ceil(avg_visits_global)))
            print(f"Step 3 — Multiclass: global K={K_per_veh[0]}")

        proba_matrix = m_model.predict_proba(X_pred)  # (n_vehicles, n_classes)

        _class_labels = le.classes_

        # Ensure proba columns align with le.classes_ ordering
        # (AG/TabPFN wrappers may use a different class order)
        if hasattr(m_model, "classes_") and not np.array_equal(m_model.classes_, _class_labels):
            reorder = [list(m_model.classes_).index(c) for c in _class_labels
                       if c in m_model.classes_]
            if len(reorder) == len(_class_labels):
                proba_matrix = proba_matrix[:, reorder]
                print(f"  Reordered proba columns to match LabelEncoder "
                      f"({len(reorder)} classes)")

        # Variable-K expansion: each vehicle gets its own K
        n_veh = len(df_will)
        rows_idx = []
        types_idx = []
        types_prob = []
        visit_nums = []

        # ── BO-type affinity: penalise atypical types per BO ────
        # Adjusts ranking (sort order) but stored probability stays original.
        _adj_proba = proba_matrix.copy()
        _bo_col = config.COL_BO_CLEAN
        _n_bo_penalised = 0
        _cls_parts = []
        if config.BO_TYPICAL_TYPES and _bo_col in df_will.columns:
            _bo_vals = df_will[_bo_col].values
            # Pre-compute per-class component sets
            _cls_parts = [
                {p.strip() for p in str(_class_labels[c]).split(",")}
                for c in range(len(_class_labels))
            ]
            for i, bo in enumerate(_bo_vals):
                typical = config.BO_TYPICAL_TYPES.get(bo)
                if typical is None:
                    continue
                for c, parts in enumerate(_cls_parts):
                    if not (parts & typical):   # no overlap → atypical
                        _adj_proba[i, c] *= config.BO_ATYPICAL_TYPE_PENALTY
                        _n_bo_penalised += 1
            if _n_bo_penalised:
                print(f"  BO-type affinity: penalised {_n_bo_penalised:,} "
                      f"vehicle×class cells (factor={config.BO_ATYPICAL_TYPE_PENALTY})")

        # ── Prio ↔ type mismatch: penalise contradictory combinations ──
        _prio_col = config.COL_PRIO
        _n_prio_pen = 0
        if config.PRIO_TYPE_MISMATCHES and _prio_col in df_will.columns:
            _prio_vals = df_will[_prio_col].astype(str).values
            if not _cls_parts:          # build if BO block was skipped
                _cls_parts = [
                    {p.strip() for p in str(_class_labels[c]).split(",")}
                    for c in range(len(_class_labels))
                ]
            for i, pv in enumerate(_prio_vals):
                bad_parts = config.PRIO_TYPE_MISMATCHES.get(pv)
                if bad_parts is None:
                    continue
                for c, parts in enumerate(_cls_parts):
                    if parts & bad_parts:   # overlap → mismatch
                        _adj_proba[i, c] *= config.PRIO_TYPE_MISMATCH_PENALTY
                        _n_prio_pen += 1
            if _n_prio_pen:
                print(f"  Prio-type mismatch: penalised {_n_prio_pen:,} "
                      f"vehicle×class cells (factor={config.PRIO_TYPE_MISMATCH_PENALTY})")

        sorted_idx = np.argsort(_adj_proba, axis=1)[:, ::-1]   # ranking from adjusted
        sorted_prob = np.take_along_axis(proba_matrix, sorted_idx, axis=1)  # original probs
        n_classes = sorted_idx.shape[1]

        for v in range(n_veh):
            k = K_per_veh[v]
            visit = 0
            j = 0  # pointer into sorted classes
            while visit < k and j < n_classes:
                cidx = sorted_idx[v, j]
                cprob = sorted_prob[v, j]
                rows_idx.append(v)
                types_idx.append(cidx)
                types_prob.append(cprob)
                visit += 1
                visit_nums.append(visit)
                j += 1

        df_will = df_will.iloc[rows_idx].copy().reset_index(drop=True)
        df_will[config.COL_PREDICTED_TYPE] = le.inverse_transform(np.array(types_idx))
        df_will[config.COL_TYPE_PROBABILITY] = np.array(types_prob, dtype=np.float32)
        df_will[config.COL_VISIT_NUMBER] = np.array(visit_nums, dtype=np.int32)

        # Drop very unlikely visits (prob < 5%)
        before_filter = len(df_will)
        df_will = df_will[df_will[config.COL_TYPE_PROBABILITY] >= config.MIN_TYPE_PROBABILITY].copy()

        print(f"  ({n_avail}/{len(all_m_feats)} features from Source B)")
        print(f"  {n_veh:,} vehicles → {len(df_will):,} predicted visits "
              f"(avg {len(df_will)/n_veh:.1f}/vehicle)")
        if before_filter > len(df_will):
            print(f"  Dropped {before_filter - len(df_will):,} visits with prob < 5%")
        print(f"  Package distribution:\n{df_will[config.COL_PREDICTED_TYPE].value_counts().to_string()}")
    else:
        df_will[config.COL_PREDICTED_TYPE] = config.FILLNA_CATEGORICAL
        df_will[config.COL_TYPE_PROBABILITY] = 1.0
        df_will[config.COL_VISIT_NUMBER] = 1
        avg_visits = 1.0
        dcol = config.COL_DERIVAT_CLEAN

    # Apply calibration weights from backtest
    if calibration_factors:
        df_will[config.COL_CALIBRATION_WEIGHT] = (
            df_will[config.COL_PREDICTED_TYPE].map(calibration_factors).fillna(1.0)
        )
        print(f"\n  Calibration factors applied:")
        for t, f in sorted(calibration_factors.items()):
            print(f"    {t:20s} × {f:.3f}")
    else:
        df_will[config.COL_CALIBRATION_WEIGHT] = 1.0

    # Make predicted type available as multiclass target for temporal model
    df_will[config.COL_MULTICLASS_TARGET] = df_will[config.COL_PREDICTED_TYPE]

    # ------------------------------------------------------------------
    # 4. TEMPORAL: predict when
    # ------------------------------------------------------------------
    dur_chosen = "N/A"

    # ── 4a. Duration prediction ──────────────────────────────
    if temporal_results and temporal_results.get("duration"):
        dur = temporal_results["duration"]

        # ── AutoML override: swap in best model_dict if it beat Manual ──
        if automl_results and automl_results.get("temporal_comparison") is not None:
            _tmp_winner = automl_results["temporal_comparison"].iloc[0]["Pipeline"]
            _tmp_mae = automl_results["temporal_comparison"].iloc[0]["MAE_days"]
            if _tmp_winner == "AutoGluon" and automl_results.get("temporal", {}).get("temporal_model_dict") is not None:
                # Merge: keep blending metadata from manual, swap in AG model
                dur = {**dur, **automl_results["temporal"]["temporal_model_dict"]}
                print(f"  ★ AutoGluon won temporal (MAE={_tmp_mae:.1f}d) → using AG model")
            elif _tmp_winner == "TabPFN" and automl_results.get("tabpfn_temporal", {}).get("temporal_model_dict") is not None:
                dur = {**dur, **automl_results["tabpfn_temporal"]["temporal_model_dict"]}
                print(f"  ★ TabPFN won temporal (MAE={_tmp_mae:.1f}d) → using TabPFN model")

        dur_pred_model, dur_chosen = _predict_best_temporal(
            df_will, dur, label="Step 4a — Duration")

        # Hybrid: blend model prediction with type-group median (+ TS)
        type_group_median = dur.get("type_group_median", {})
        # Fallback chain: overall_median (from training data) -> weighted
        # mean of type-group medians -> model-prediction median -> last resort.
        if "overall_median" in dur:
            overall_median_dur = dur["overall_median"]
        elif type_group_median:
            _ps = dur.get("type_group_stats")
            if _ps is not None:
                overall_median_dur = float(
                    (_ps["median"] * _ps["count"]).sum() / _ps["count"].sum())
            else:
                overall_median_dur = float(np.mean(list(type_group_median.values())))
        elif dur_pred_model is not None:
            overall_median_dur = float(np.median(dur_pred_model))
        else:
            overall_median_dur = float(np.nan)
            print("  ⚠ No duration reference available — using NaN")

        # Only map TS forecast when metrics pass the quality gate
        dur_ts_ok = _ts_is_reasonable(
            temporal_results.get("duration_ts_comparison"),
            overall_median_dur, label="Duration")
        dur_ts_vals = (_map_vehicles_to_ts(
            df_will, temporal_results.get("duration_ts_future"), config)
            if dur_ts_ok else None)

        # Map predicted_type -> type-group -> type-group median duration
        if type_group_median:
            dur_pred_type = (
                df_will[config.COL_PREDICTED_TYPE]
                .apply(config.map_type_to_group)
                .map(type_group_median)
                .fillna(overall_median_dur)
                .to_numpy(dtype=np.float32)
            )
            print(f"  Type-group medians: {type_group_median}")

            # ── Optimize blend weights on temporal test set ──
            _y_test_dur = dur.get("y_test")
            _y_test_types = dur.get("y_test_types")
            _best_key = dur.get("best_overall", dur.get("best_model"))
            _model_test_pred = (
                dur["results"].get(_best_key, {}).get("y_pred")
                if dur.get("results") else None)
            if (_y_test_dur is not None and _model_test_pred is not None
                    and _y_test_types is not None):
                _type_test_pred = np.array([
                    type_group_median.get(
                        config.map_type_to_group(t), overall_median_dur)
                    for t in _y_test_types
                ], dtype=np.float64)
                opt_m, opt_t, _ = _optimize_blend_weights(
                    _model_test_pred, _type_test_pred,
                    dur_ts=None, y_true=_y_test_dur)
                print(f"  Optimised dual weights on test set: "
                      f"model={opt_m:.0%}, type-median={opt_t:.0%}")
            else:
                opt_m = config.BLEND_W_DUAL
                opt_t = 1.0 - config.BLEND_W_DUAL
                print(f"  Using default dual weights (no test data for optimisation)")

            if dur_pred_model is None:
                # No trained temporal model - use type-group median only
                dur_pred = dur_pred_type
                print(f"  No temporal model - using type-group-median only")
            elif dur_ts_vals is not None:
                # Triple blend: keep optimised model/type ratio, add TS
                w_ts = config.BLEND_W_TS
                scale = 1.0 - w_ts
                w_m = round(opt_m * scale, 3)
                w_t = round(opt_t * scale, 3)
                dur_pred = (w_m * dur_pred_model
                            + w_t * dur_pred_type
                            + w_ts * dur_ts_vals)
                print(f"  Triple blend: {dur_chosen}×{w_m:.0%} + "
                      f"type-group-median×{w_t:.0%} + TS-monthly×{w_ts:.0%}")
            else:
                # Dual blend with optimised weights
                dur_pred = opt_m * dur_pred_model + opt_t * dur_pred_type
                print(f"  Dual blend: {dur_chosen}×{opt_m:.0%} + "
                      f"type-group-median×{opt_t:.0%}")
        else:
            if dur_pred_model is not None:
                dur_pred = dur_pred_model
                print(f"  No type-group medians - using {dur_chosen} only")
            else:
                dur_pred = np.full(len(df_will), overall_median_dur, dtype=np.float32)
                print(f"  No temporal model or type-group medians - "
                      f"using overall median ({overall_median_dur:.0f} d)")

        df_will[config.COL_PREDICTED_DURATION] = dur_pred
        df_will[config.COL_DUR_MODEL] = dur_pred_model if dur_pred_model is not None else dur_pred
        df_will[config.COL_DUR_TYPE_MEDIAN] = (
            df_will[config.COL_PREDICTED_TYPE]
            .apply(config.map_type_to_group)
            .map(type_group_median)
            .fillna(overall_median_dur)
            if type_group_median else dur_pred_model)
        print(f"  Duration: mean={dur_pred.mean():.0f} d, "
              f"median={np.median(dur_pred):.0f} d")

        # Per-type-group duration distribution
        _pa_col = (df_will[config.COL_PREDICTED_TYPE]
                   .apply(config.map_type_to_group))
        _dur_by_pa = (pd.DataFrame({"type_group": _pa_col.values,
                                     "duration": dur_pred})
                      .groupby("type_group")["duration"]
                      .agg(["count", "mean", "median", "std", "min", "max"]))
        print(f"\n  Predicted duration by type-group:")
        print(_dur_by_pa.round(1).to_string())
    else:
        dur_pred = None
        print("Step 4a — Duration: skipped (no duration model)")

    # ── 4b. Estimated dates from deadline (rueckgabe_soll) ─────
    #
    # rueckgabe_soll is the date by which **all** retrofits for a
    # vehicle (same AT-Nummer) must be finished.  For multi-visit
    # vehicles (K > 1) we stagger visits BACKWARDS from the deadline
    # so that visit K ends at rueckgabe_soll and earlier visits are
    # spaced before it.
    #
    # NOTE: Lead-time (av_erstaufbau → umr_start) was removed entirely
    # because av_erstaufbau is a Source A-only column — unseen forecast
    # vehicles never have it, so lead-time is useless at prediction time.
    if dur_pred is not None:
        # Build per-row anchor date: rueckgabe_soll → anlauf_sop → now
        anchor = pd.Series(pd.NaT, index=df_will.index)
        anchor_source = pd.Series("none", index=df_will.index)

        if config.COL_RUECKGABE_SOLL in df_will.columns:
            rs = pd.to_datetime(df_will[config.COL_RUECKGABE_SOLL], errors="coerce")
            mask_rs = rs.notna() & anchor.isna()
            anchor = anchor.where(~mask_rs, rs)
            anchor_source = anchor_source.where(~mask_rs, "rueckgabe_soll")

        if config.COL_ANLAUF_SOP in df_will.columns:
            sop = pd.to_datetime(df_will[config.COL_ANLAUF_SOP], errors="coerce")
            mask_sop = sop.notna() & anchor.isna()
            anchor = anchor.where(~mask_sop, sop)
            anchor_source = anchor_source.where(~mask_sop, "anlauf_sop")

        mask_now = anchor.isna()
        anchor = anchor.fillna(pd.Timestamp.now())
        anchor_source = anchor_source.where(~mask_now, "now")

        n_rs  = (anchor_source == "rueckgabe_soll").sum()
        n_sop = (anchor_source == "anlauf_sop").sum()
        n_now = (anchor_source == "now").sum()
        print(f"Step 4b — Anchor dates: rueckgabe_soll={n_rs:,}, "
              f"anlauf_sop={n_sop:,}, now={n_now:,}")

        # Inter-visit spacing (per-derivat median, fallback to global)
        _vs = (multi_results or {}).get("visit_stats", {})
        inter_days_global = _vs.get("inter_visit_median_days", 120)
        inter_per_derivat = _vs.get("inter_visit_per_derivat", {})

        has_multi = (config.COL_VISIT_NUMBER in df_will.columns
                     and df_will[config.COL_VISIT_NUMBER].max() > 1)

        dur_offset = pd.to_timedelta(dur_pred, unit="D")

        if has_multi:
            join_key = config.COL_JOIN_KEY
            if join_key not in df_will.columns:
                join_key = config.COL_JOIN_KEY_SOURCE_A
            if join_key not in df_will.columns:
                join_key = df_will.columns[0]
            max_visit = df_will.groupby(join_key)[config.COL_VISIT_NUMBER].transform("max")

            # Per-derivat spacing: map each row to its derivat's median gap
            if inter_per_derivat and dcol in df_will.columns:
                inter_days_arr = (
                    df_will[dcol]
                    .map(inter_per_derivat)
                    .fillna(inter_days_global)
                    .values
                )
                n_deriv_known = int(df_will[dcol].isin(inter_per_derivat).sum())
            else:
                inter_days_arr = inter_days_global
                n_deriv_known = 0

            backward_offset = pd.to_timedelta(
                (max_visit - df_will[config.COL_VISIT_NUMBER]) * inter_days_arr, unit="D")
            df_will[config.COL_ESTIMATED_START] = anchor - dur_offset - backward_offset
            print(f"Step 4c — anchor − duration − backward stagger "
                  f"(global inter={inter_days_global:.0f} d, "
                  f"per-derivat: {n_deriv_known}/{len(df_will)}, "
                  f"max K={int(max_visit.max())})")
        else:
            df_will[config.COL_ESTIMATED_START] = anchor - dur_offset
            print(f"Step 4c — anchor − duration")

        # Fallback for remaining NaTs (missing deadline etc.)
        fallback = pd.Timestamp.now() + pd.Timedelta(days=config.FALLBACK_OFFSET_DAYS)
        df_will[config.COL_ESTIMATED_START] = df_will[config.COL_ESTIMATED_START].fillna(fallback)

        df_will[config.COL_ESTIMATED_MONTH] = df_will[config.COL_ESTIMATED_START].dt.to_period("M")

        # Filter to reasonable future window (today → +12 months)
        now_period = pd.Period(pd.Timestamp.now(), freq="M")
        max_period = now_period + config.FORECAST_HORIZON_MONTHS
        before_filter = len(df_will)
        df_will = df_will[
            (df_will[config.COL_ESTIMATED_MONTH] >= now_period) &
            (df_will[config.COL_ESTIMATED_MONTH] <= max_period)
        ].copy()
        print(f"  Filtered to {now_period}–{max_period}: "
              f"{len(df_will):,} visits (dropped {before_filter - len(df_will):,} outside window)")

        if df_will.empty:
            print("  ⚠ No vehicles in forecast window after filtering.")
        else:
            print(f"  Range: {df_will['estimated_month'].min()} → "
                  f"{df_will['estimated_month'].max()}")
    else:
        print("Step 4 — Temporal: skipped (no temporal model)")

    # ------------------------------------------------------------------
    # 4d. Row-level calibration — adjust actual row count per type
    #     so that row counts in the output match calibrated totals.
    #     Only acts when factor is outside [0.50, 1.50] (±50 % tolerance).
    #     Factor < 0.50 → drop lowest-probability visits for that type.
    #     Factor > 1.50 → duplicate highest-probability visits.
    # ------------------------------------------------------------------
    CAL_LO, CAL_HI = config.CAL_LO, config.CAL_HI
    if calibration_factors and not df_will.empty:
        before_cal = len(df_will)
        parts_cal = []
        skipped, adjusted = [], []
        for rtype, factor in calibration_factors.items():
            mask = df_will[config.COL_PREDICTED_TYPE] == rtype
            sub = df_will[mask]
            if sub.empty:
                continue
            if CAL_LO <= factor <= CAL_HI:
                parts_cal.append(sub)
                skipped.append(rtype)
                continue
            target_n = max(1, round(len(sub) * factor))
            if target_n < len(sub):
                parts_cal.append(
                    sub.nlargest(target_n, config.COL_TYPE_PROBABILITY))
            elif target_n > len(sub):
                extra_n = target_n - len(sub)
                extras = sub.nlargest(extra_n, config.COL_TYPE_PROBABILITY).copy()
                parts_cal.append(sub)
                parts_cal.append(extras)
            else:
                parts_cal.append(sub)
            adjusted.append((rtype, factor, len(sub), target_n))
        # Types not in calibration_factors: keep as-is
        known_types = set(calibration_factors.keys())
        other = df_will[~df_will[config.COL_PREDICTED_TYPE].isin(known_types)]
        if not other.empty:
            parts_cal.append(other)
        df_will = pd.concat(parts_cal, ignore_index=True)
        if adjusted:
            print(f"\nStep 4d — Row-level calibration ({before_cal:,} → {len(df_will):,} visits)")
            for rtype, factor, n_was, n_now in adjusted:
                print(f"    {rtype:20s}  ×{factor:.3f}  {n_was:,} → {n_now:,}")
        if skipped:
            print(f"  Within tolerance ({CAL_LO}–{CAL_HI}), unchanged: "
                  f"{', '.join(skipped)}")

    # ------------------------------------------------------------------
    # 5. Forecast summary + pivot
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("FORECAST RESULT")
    print(f"{'=' * 80}\n")

    # Pivot table for visualization (1 row = 1 vehicle's first predicted visit)
    pivot = None
    pivot_future = None
    pivot_calibrated = None
    if config.COL_ESTIMATED_MONTH in df_will.columns and not df_will.empty:
        pivot = (df_will
                 .groupby([config.COL_ESTIMATED_MONTH, config.COL_PREDICTED_TYPE])
                 .size()
                 .unstack(fill_value=0)
                 .sort_index())

        # Calibrated pivot: sum of calibration_weight instead of count
        if calibration_factors:
            pivot_calibrated = (df_will
                .groupby([config.COL_ESTIMATED_MONTH, config.COL_PREDICTED_TYPE])[config.COL_CALIBRATION_WEIGHT]
                .sum()
                .unstack(fill_value=0)
                .sort_index()
                .round(0)
                .astype(int))

        # Future months only
        current = pd.Period(pd.Timestamp.now(), freq="M")
        pivot_future = pivot[pivot.index >= current]

        if not pivot_future.empty:
            n_visits_total = int(pivot_future.values.sum())

            # ── Compact monthly summary (total per month) ──
            monthly_totals = pivot_future.sum(axis=1).astype(int)
            monthly_totals.name = "visits"
            print("Predicted visits per month (raw):")
            print(monthly_totals.to_string())
            print(f"\nVehicles predicted to come: {n_vehicles_predicted:,}")
            print(f"Total predicted visits: {n_visits_total:,} "
                  f"(avg {n_visits_total/n_vehicles_predicted:.1f} visits/vehicle)")

            # ── Type distribution (total across all months) ──
            type_totals = pivot_future.sum(axis=0).astype(int).sort_values(ascending=False)
            type_totals = type_totals[type_totals > 0]
            print(f"\nBy retrofit type (raw):")
            for t, n in type_totals.items():
                print(f"  {t:<45s} {n:>5,}")

            # ── Calibrated summary ──
            if pivot_calibrated is not None:
                pf_cal = pivot_calibrated[pivot_calibrated.index >= current]
                if not pf_cal.empty:
                    n_cal = int(pf_cal.values.sum())
                    cal_monthly = pf_cal.sum(axis=1).astype(int)
                    cal_monthly.name = "visits_cal"
                    print(f"\nCalibrated visits per month:")
                    print(cal_monthly.to_string())
                    print(f"Total calibrated visits: {n_cal:,}")
                    cal_types = pf_cal.sum(axis=0).astype(int).sort_values(ascending=False)
                    cal_types = cal_types[cal_types > 0]
                    print(f"\nBy retrofit type (calibrated):")
                    for t, n in cal_types.items():
                        print(f"  {t:<45s} {n:>5,}")

            # Plot
            pf_cal_plot = None
            if pivot_calibrated is not None:
                pf_cal_plot = pivot_calibrated[pivot_calibrated.index >= current]
                if pf_cal_plot.empty:
                    pf_cal_plot = None
            _plot_forecast(pivot_future, pf_cal_plot)
        else:
            print("No future months in forecast — all predicted dates are in the past.")
            pivot_future = pivot
    else:
        pivot_future = None
        print(f"Total predicted visits: {len(df_will):,}")
        print(f"Package distribution:\n{df_will['predicted_type'].value_counts().to_string()}")

    gc.collect()
    mem()

    # ------------------------------------------------------------------
    # 6. Naive time-series baseline (flat projection of historical avg)
    # ------------------------------------------------------------------
    naive = _naive_baseline(multi_results)

    # ------------------------------------------------------------------
    # 7. Model metadata (traceability for CSV export)
    # ------------------------------------------------------------------
    best_bin = binary_results.get("best_model", "unknown")
    best_approach = binary_results.get("best_approach", "unknown")
    best_multi = (multi_results or {}).get("best_model", "N/A")
    # dur_chosen is set by _predict_best_temporal above
    best_dur = dur_chosen if dur_pred is not None else "N/A"

    model_metadata = {
        "binary_approach": best_approach,
        "binary_model": best_bin,
        "multiclass_model": best_multi,
        "temporal_duration_model": best_dur,
        "optimal_threshold": threshold,
    }
    print(f"\n  Model metadata for export:")
    for k, v in model_metadata.items():
        print(f"    {k:30s} = {v}")

    return {
        "df_forecast": df_will,
        "pivot": pivot,
        "pivot_future": pivot_future,
        "pivot_calibrated": pivot_calibrated,
        "naive_baseline": naive,
        "n_unseen": len(df_unseen),
        "n_vehicles": n_vehicles_predicted,
        "n_visits": len(df_will),
        "threshold": threshold,
        "model_metadata": model_metadata,
    }


def _encode(df, cat_cols, num_cols, enc_map, global_mean):
    """Target-encode categoricals and fill numerics."""
    X = pd.DataFrame(index=df.index)
    for col in cat_cols:
        if col in df.columns:
            X[col] = df[col].map(enc_map.get(col, {})).fillna(global_mean).astype("float32")
        else:
            X[col] = global_mean
    for col in num_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0).astype("float32")
        else:
            X[col] = 0.0
    return X


def _plot_forecast(pivot_future, pivot_calibrated=None):
    """Stacked bar chart: month × retrofit type, plus per-type detail plots."""
    # ── 1. Main stacked bar (same as before) ──
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot_future.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Predicted Retrofits by Month and Type")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Vehicles")
    ax.legend(title="Retrofit Type", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ── 2. Per-type subplots: raw bar + calibrated line ──
    # Only the 6 base types (skip multi-packages)
    base_types = config.BASE_RETROFIT_TYPES
    types_to_plot = [t for t in base_types if t in pivot_future.columns]
    if not types_to_plot:
        return

    n = len(types_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    x_labels = [str(idx) for idx in pivot_future.index]
    x = np.arange(len(x_labels))

    for i, rtype in enumerate(types_to_plot):
        ax = axes[i // ncols][i % ncols]
        raw_vals = pivot_future[rtype].values.astype(float)
        ax.bar(x, raw_vals, color="steelblue", alpha=0.6, label="Raw")

        if pivot_calibrated is not None and rtype in pivot_calibrated.columns:
            cal_vals = pivot_calibrated[rtype].reindex(pivot_future.index, fill_value=0).values.astype(float)
            ax.plot(x, cal_vals, color="red", linewidth=2, marker="o",
                    markersize=3, label="Calibrated")

        ax.set_title(rtype, fontsize=11, fontweight="bold")
        ax.set_xticks(x[::3])
        ax.set_xticklabels([x_labels[j] for j in range(0, len(x_labels), 3)],
                           rotation=45, fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylabel("Visits")

    # Hide empty subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Raw vs Calibrated Forecast by Retrofit Type",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — BACKTEST (predicted vs actual → calibration factors)
# ══════════════════════════════════════════════════════════════════


def evaluate_retrofits(df_binary, df_multi, data_prep, binary_results,
                       source_results, multi_results, temporal_results,
                       automl_results=None):
    """
    Full end-to-end backtest using Source B-only features:

        1. Binary  → which of ALL vehicles will come?
        2. Multiclass → what retrofit type for those predicted to come?
        3. Compare predicted vs actual counts per individual type + month.

    This mirrors exactly what run_forecast() does for future vehicles,
    so the accuracy shown here is what you can expect from real forecasts.

    Parameters
    ----------
    df_binary        : full Source B DataFrame (merge_left_join), all vehicles.
    df_multi         : INNER JOIN DataFrame (Source B+Source A) — used only to get
                       actual retrofit types and umr_start for comparison.
    data_prep        : dict from prepare_binary (encoding maps).
    binary_results   : dict from final_model_comparison (best binary model).
    source_results   : dict from the winning approach (trained binary model).
    multi_results    : dict from multiclass_retrofit_types.
    temporal_results : dict from temporal_prediction (unused, kept for API).
    automl_results   : dict from run_automl_benchmark() (module 07).
                       If provided and AutoGluon/TabPFN beat the manual
                       pipeline, the winning model is used instead.
    """
    print("\n" + "=" * 80)
    print("PREDICTED vs ACTUAL — Full End-to-End Backtest (Source B-only)")
    print("  Mirrors forecast.run_forecast() on historical data")
    print("=" * 80 + "\n")

    # ── 1. Binary: predict who will come ─────────────────────
    threshold = binary_results.get("optimal_threshold", 0.3)
    best_bin  = binary_results["best_model"]
    bin_model = source_results["results"][best_bin]["model"]
    bin_enc   = data_prep["target_encoding_map"]
    bin_gmean = data_prep["global_mean"]
    bin_cat   = data_prep["categorical_cols_clean"]
    bin_feat  = data_prep["feature_cols"]

    # LLM-Stacked / LLM-* are meta-learners that cannot run on raw features
    # at inference time.  Fall back to the best usable (non-LLM) model.
    if best_bin.startswith("LLM"):
        _cmp = binary_results.get("comparison")
        _fallback = None
        if _cmp is not None:
            for _, _row in _cmp.iterrows():
                _mn = _row["Model"]
                if _mn.startswith("LLM") or _mn.startswith("Dummy"):
                    continue
                if _mn in source_results["results"] and "model" in source_results["results"][_mn]:
                    _fallback = _mn
                    break
        if _fallback is None:
            _ml = {k: v for k, v in source_results["results"].items()
                   if not k.startswith("LLM") and not k.startswith("Dummy") and "model" in v}
            if not _ml:
                raise RuntimeError("No usable (non-LLM, non-Dummy) binary model found")
            _fallback = max(_ml, key=lambda k: _ml[k].get("roc_auc", 0))
        print(f"  [backtest] {best_bin} cannot run at inference "
              f"→ using next-best model: {_fallback}")
        bin_model = source_results["results"][_fallback]["model"]

    # ── AutoML override: use best pipeline if it beat Manual ──
    if automl_results and automl_results.get("binary_comparison") is not None:
        _bin_winner = automl_results["binary_comparison"].iloc[0]["Pipeline"]
        _bin_auc = automl_results["binary_comparison"].iloc[0]["ROC-AUC"]
        if _bin_winner == "AutoGluon" and automl_results.get("binary", {}).get("model") is not None:
            bin_model = automl_results["binary"]["model"]
            print(f"  ★ AutoGluon won binary (ROC-AUC={_bin_auc:.4f}) → using AG model")
        elif _bin_winner == "TabPFN" and automl_results.get("tabpfn_binary", {}).get("model") is not None:
            bin_model = automl_results["tabpfn_binary"]["model"]
            print(f"  ★ TabPFN won binary (ROC-AUC={_bin_auc:.4f}) → using TabPFN model")

    sel_feats = source_results.get("selected_features")
    if sel_feats:
        bin_feat = sel_feats
        bin_cat  = [c for c in bin_cat if c in sel_feats]

    bin_num = [c for c in bin_feat if c not in bin_cat]

    # Predict on ALL vehicles (not just unseen)
    df = df_binary.copy()
    X_bin = _encode(df, bin_cat, bin_num, bin_enc, bin_gmean)
    prob = bin_model.predict_proba(X_bin[bin_feat])[:, 1]
    df[config.COL_PROB_COMING] = prob

    df["pred_coming"] = (df[config.COL_PROB_COMING] >= threshold).astype(int)

    n_total       = len(df)
    actual_came   = int(df[config.COL_CAME_FOR_RETROFIT].sum())
    predicted_come = int(df["pred_coming"].sum())

    tp = int(((df[config.COL_CAME_FOR_RETROFIT] == 1) & (df["pred_coming"] == 1)).sum())
    fn = int(((df[config.COL_CAME_FOR_RETROFIT] == 1) & (df["pred_coming"] == 0)).sum())
    fp = int(((df[config.COL_CAME_FOR_RETROFIT] == 0) & (df["pred_coming"] == 1)).sum())

    recall    = tp / max(actual_came, 1) * 100
    precision = tp / max(predicted_come, 1) * 100

    print(f"Step 1 — Binary (threshold={threshold:.3f}, model={best_bin})")
    print(f"  Total vehicles:       {n_total:,}")
    print(f"  Actually came:        {actual_came:,}")
    print(f"  Predicted to come:    {predicted_come:,}")
    print(f"  True positives:       {tp:,}  (recall={recall:.1f}%)")
    print(f"  False negatives:      {fn:,}  (missed)")
    print(f"  False positives:      {fp:,}  (precision={precision:.1f}%)")

    # ── 2. Multiclass: predict type for predicted-to-come ────
    #    Includes per-derivat K expansion (same as forecast.py Step 3)
    df_will = df[df["pred_coming"] == 1].copy()

    best_name = multi_results.get("best_model")
    if best_name is None:
        print("  ✗ No multiclass best_model — skipping type prediction in evaluate_retrofits")
        return {"error": "no multiclass best_model"}
    m_model   = multi_results["results"][best_name]["model"]
    le        = multi_results["label_encoder"]

    # LLM-Stacked / LLM-* meta-learners use a different feature space
    # (stacked probabilities) and cannot run on the regular feature matrix.
    # Fall back to the best usable (non-LLM) model.
    if best_name.startswith("LLM"):
        _cmp_m = multi_results.get("comparison")
        _fallback_m = None
        if _cmp_m is not None:
            for _, _row in _cmp_m.iterrows():
                _mn = _row["Model"]
                if _mn.startswith("LLM"):
                    continue
                if _mn in multi_results["results"] and "model" in multi_results["results"][_mn]:
                    _fallback_m = _mn
                    break
        if _fallback_m is None:
            _ml_m = {k: v for k, v in multi_results["results"].items()
                     if not k.startswith("LLM") and "model" in v}
            if not _ml_m:
                print("  ✗ No usable (non-LLM) multiclass model — skipping")
                return {"error": "no usable multiclass model"}
            _fallback_m = max(_ml_m, key=lambda k: _ml_m[k].get("f1_weighted", 0))
        print(f"  [backtest] {best_name} cannot run at inference "
              f"→ using next-best multiclass model: {_fallback_m}")
        m_model = multi_results["results"][_fallback_m]["model"]
        best_name = _fallback_m

    # ── AutoML override: use best pipeline if it beat Manual ──
    if automl_results and automl_results.get("multiclass_comparison") is not None:
        _mc_winner = automl_results["multiclass_comparison"].iloc[0]["Pipeline"]
        _mc_f1w = automl_results["multiclass_comparison"].iloc[0]["F1_weighted"]
        if _mc_winner == "AutoGluon" and automl_results.get("multiclass", {}).get("model") is not None:
            m_model = automl_results["multiclass"]["model"]
            print(f"  ★ AutoGluon won multiclass (F1w={_mc_f1w:.4f}) → using AG model")
        elif _mc_winner == "TabPFN" and automl_results.get("tabpfn_multiclass", {}).get("model") is not None:
            m_model = automl_results["tabpfn_multiclass"]["model"]
            print(f"  ★ TabPFN won multiclass (F1w={_mc_f1w:.4f}) → using TabPFN model")

    m_enc     = multi_results.get("encoding_map", {})
    m_gmean   = float(multi_results.get("global_mean", 0))
    m_cat     = multi_results.get("cat_cols", [])
    m_num     = multi_results.get("num_cols", [])
    m_freq    = multi_results.get("freq_map", {})   # v2 freq encoding
    freq_col_names = [f"{c}_freq" for c in m_cat] if m_freq else []
    all_feats = m_cat + freq_col_names + m_num

    # Build feature matrix (Source B columns only — same as forecast.py)
    n = len(df_will)
    X_arr = np.zeros((n, len(all_feats)), dtype=np.float32)
    freq_col_set = set(freq_col_names)
    for i, c in enumerate(all_feats):
        if c in m_cat:
            if c in df_will.columns:
                X_arr[:, i] = (df_will[c].map(m_enc.get(c, {}))
                               .fillna(m_gmean).to_numpy(dtype=np.float32))
            else:
                X_arr[:, i] = m_gmean
        elif c in freq_col_set:
            # Frequency-encoded categorical (v2)
            orig_col = c.replace("_freq", "")
            fmap = m_freq.get(orig_col, {})
            if orig_col in df_will.columns and fmap:
                X_arr[:, i] = (df_will[orig_col].map(fmap)
                               .fillna(0).to_numpy(dtype=np.float32))
            else:
                X_arr[:, i] = 0
        else:
            if c in df_will.columns and df_will[c].dtype != object:
                X_arr[:, i] = df_will[c].fillna(0).to_numpy(dtype=np.float32)
            else:
                X_arr[:, i] = 0

    X_multi = pd.DataFrame(X_arr, columns=all_feats)

    # ── Filter to selected features (top-N from pre-filter) ─
    sel_feats = multi_results.get("selected_features")
    if sel_feats:
        X_multi = X_multi[[c for c in sel_feats if c in X_multi.columns]]
    elif hasattr(m_model, "feature_names_in_"):
        X_multi = X_multi[[c for c in m_model.feature_names_in_ if c in X_multi.columns]]

    # ── Per-derivat K expansion (identical to run_forecast) ───
    visit_stats = multi_results.get("visit_stats", {})
    avg_visits_global = visit_stats.get("mean", 1.0)
    if avg_visits_global is None or (isinstance(avg_visits_global, float) and np.isnan(avg_visits_global)):
        avg_visits_global = 1.0
    per_derivat = visit_stats.get("per_derivat", {})
    dcol = visit_stats.get("derivat_col", config.COL_DERIVAT_CLEAN)

    if per_derivat and dcol in df_will.columns:
        K_per_veh = (
            df_will[dcol]
            .map(per_derivat)
            .fillna(avg_visits_global)
            .apply(np.ceil)
            .astype(int)
            .values
        )
    else:
        K_per_veh = np.full(n, int(np.ceil(avg_visits_global)))

    proba_matrix = m_model.predict_proba(X_multi)

    _class_names = le.classes_

    # Ensure proba columns align with le.classes_ (AG/TabPFN may differ)
    if hasattr(m_model, "classes_") and not np.array_equal(m_model.classes_, _class_names):
        reorder = [list(m_model.classes_).index(c) for c in _class_names
                   if c in m_model.classes_]
        if len(reorder) == len(_class_names):
            proba_matrix = proba_matrix[:, reorder]

    # ── BO-type affinity: penalise atypical types per BO ────
    _adj_proba = proba_matrix.copy()
    _bo_col = config.COL_BO_CLEAN
    _cls_parts = []
    if config.BO_TYPICAL_TYPES and _bo_col in df_will.columns:
        _bo_vals = df_will[_bo_col].values
        _cls_parts = [
            {p.strip() for p in str(_class_names[c]).split(",")}
            for c in range(len(_class_names))
        ]
        _n_pen = 0
        for i, bo in enumerate(_bo_vals):
            typical = config.BO_TYPICAL_TYPES.get(bo)
            if typical is None:
                continue
            for c, parts in enumerate(_cls_parts):
                if not (parts & typical):
                    _adj_proba[i, c] *= config.BO_ATYPICAL_TYPE_PENALTY
                    _n_pen += 1
        if _n_pen:
            print(f"  BO-type affinity: penalised {_n_pen:,} "
                  f"vehicle×class cells (factor={config.BO_ATYPICAL_TYPE_PENALTY})")

    # ── Prio ↔ type mismatch: penalise contradictory combinations ──
    _prio_col = config.COL_PRIO
    _n_prio_pen = 0
    if config.PRIO_TYPE_MISMATCHES and _prio_col in df_will.columns:
        _prio_vals = df_will[_prio_col].astype(str).values
        if not _cls_parts:
            _cls_parts = [
                {p.strip() for p in str(_class_names[c]).split(",")}
                for c in range(len(_class_names))
            ]
        for i, pv in enumerate(_prio_vals):
            bad_parts = config.PRIO_TYPE_MISMATCHES.get(pv)
            if bad_parts is None:
                continue
            for c, parts in enumerate(_cls_parts):
                if parts & bad_parts:
                    _adj_proba[i, c] *= config.PRIO_TYPE_MISMATCH_PENALTY
                    _n_prio_pen += 1
        if _n_prio_pen:
            print(f"  Prio-type mismatch: penalised {_n_prio_pen:,} "
                  f"vehicle×class cells (factor={config.PRIO_TYPE_MISMATCH_PENALTY})")

    sorted_idx  = np.argsort(_adj_proba, axis=1)[:, ::-1]
    sorted_prob = np.take_along_axis(proba_matrix, sorted_idx, axis=1)

    # ── Expected counts per type (K-weighted, probability-weighted) ────
    #    Each vehicle contributes K_v × P(type|vehicle) to the expected
    #    count, where K_v = how many visits we expect for that vehicle.
    #    This avoids the 5%-threshold loss while respecting multi-visit.
    _exp_per_class = (proba_matrix * K_per_veh[:, None]).sum(axis=0)
    _expected_per_type = {}
    for _ci, _cn in enumerate(_class_names):
        for _part in (p.strip() for p in str(_cn).split(",")
                      if p.strip() and p.strip() != "<NA>"):
            _expected_per_type[_part] = (
                _expected_per_type.get(_part, 0.0) + _exp_per_class[_ci])

    n_classes = sorted_idx.shape[1]
    rows_idx, types_idx, types_prob, visit_nums = [], [], [], []
    for v in range(n):
        k = K_per_veh[v]
        visit = 0
        j = 0
        while visit < k and j < n_classes:
            cidx = sorted_idx[v, j]
            cprob = sorted_prob[v, j]
            rows_idx.append(v)
            types_idx.append(cidx)
            types_prob.append(cprob)
            visit += 1
            visit_nums.append(visit)
            j += 1

    df_will = df_will.iloc[rows_idx].copy().reset_index(drop=True)
    df_will[config.COL_PACKAGE_PRED]     = le.inverse_transform(np.array(types_idx))
    df_will[config.COL_TYPE_PROBABILITY] = np.array(types_prob, dtype=np.float32)
    df_will[config.COL_VISIT_NUMBER]     = np.array(visit_nums, dtype=np.int32)

    # Drop very unlikely visits (prob < 5%)
    before_filter = len(df_will)
    df_will = df_will[df_will[config.COL_TYPE_PROBABILITY] >= config.MIN_TYPE_PROBABILITY].copy()
    n_dropped = before_filter - len(df_will)

    print(f"\nStep 2 — Multiclass + K expansion (model={best_name})")
    print(f"  {n:,} vehicles → {len(df_will):,} predicted visits "
          f"(avg {len(df_will)/max(n,1):.1f}/vehicle)")
    print(f"  Global avg visits: {avg_visits_global:.2f}")
    if n_dropped:
        print(f"  Dropped {n_dropped:,} visits with prob < 5%")
    _exp_total = sum(_expected_per_type.values())
    print(f"  Expected type-visits (prob-weighted): {_exp_total:,.0f}")

    # ── 3. Explode packages → individual types ───────────────
    _pkg_col = config.COL_MULTICLASS_TARGET
    _s = config.COL_START_UR
    _k = config.COL_JOIN_KEY
    if _k not in df_multi.columns:
        _k = config.COL_JOIN_KEY_SOURCE_A

    def _explode_types(series):
        return (series.fillna("").astype(str)
                .str.split(",")
                .apply(lambda x: [p.strip() for p in x
                                  if p.strip() and p.strip() != "<NA>"]))

    # Actual types: from df_multi (INNER JOIN — has Source A columns)
    df_actual = df_multi.copy()
    df_actual["_month"] = pd.to_datetime(df_actual[_s], errors="coerce").dt.to_period("M")
    df_actual["actual_types"] = _explode_types(df_actual[_pkg_col])

    df_act = (df_actual[["actual_types", "_month"]]
              .explode("actual_types")
              .rename(columns={"actual_types": "retrofit_type"})
              .query("retrofit_type != ''"))

    # Predicted types: from vehicles predicted to come
    # Look up umr_start from df_multi for monthly grouping (only for TPs)
    _month_map = (
        df_multi.drop_duplicates(_k)
        .set_index(_k)[_s]
        .pipe(lambda s: pd.to_datetime(s, errors="coerce").dt.to_period("M"))
    )
    df_will["_month"] = df_will[_k].map(_month_map)  # NaT for FPs (no Source A visit)
    df_will["predicted_types"] = _explode_types(df_will[config.COL_PACKAGE_PRED])

    df_prd = (df_will[["predicted_types", "_month"]]
              .explode("predicted_types")
              .rename(columns={"predicted_types": "retrofit_type"})
              .query("retrofit_type != ''"))

    n_act_types = df_act["retrofit_type"].nunique()
    n_prd_types = df_prd["retrofit_type"].nunique()
    print(f"\nStep 3 — Exploded to individual types: "
          f"{n_act_types} actual unique, {n_prd_types} predicted unique")

    # ── 4. Count comparison per type ─────────────────────────
    cnt_act = df_act.groupby("retrofit_type").size().rename("count_actual")
    cnt_prd = df_prd.groupby("retrofit_type").size().rename("count_predicted")
    count_compare = (pd.concat([cnt_act, cnt_prd], axis=1)
                     .fillna(0).astype(int)
                     .sort_index())
    count_compare["diff"] = count_compare["count_predicted"] - count_compare["count_actual"]
    count_compare["pct_diff"] = (
        count_compare["diff"] / count_compare["count_actual"].replace(0, np.nan) * 100
    ).round(1)

    # Expected counts (probability-weighted — no K / no threshold)
    cnt_exp = pd.Series(_expected_per_type, name="count_expected")
    count_compare = count_compare.join(cnt_exp.round(0).astype(int)).fillna(0)
    count_compare["exp_diff"] = count_compare["count_expected"] - count_compare["count_actual"]
    count_compare["exp_pct_diff"] = (
        count_compare["exp_diff"] / count_compare["count_actual"].replace(0, np.nan) * 100
    ).round(1)

    # Totals row
    totals = count_compare.sum(numeric_only=True)
    totals.name = "TOTAL"
    totals["pct_diff"] = round(totals["diff"] / max(totals["count_actual"], 1) * 100, 1)
    totals["exp_pct_diff"] = round(totals["exp_diff"] / max(totals["count_actual"], 1) * 100, 1)
    count_compare = pd.concat([count_compare, totals.to_frame().T])

    print(f"\n── Count comparison per retrofit type ──")
    print(count_compare.to_string())

    # ── 5. Per-type calibration factors ───────────────────────
    #    factor = actual / predicted  →  multiply forecast counts to calibrate.
    calibration_factors = {}
    for rtype in count_compare.index:
        if rtype == "TOTAL":
            continue
        act = float(count_compare.loc[rtype, "count_actual"])
        prd = float(count_compare.loc[rtype, "count_predicted"])
        calibration_factors[rtype] = round(act / prd, 3) if prd > 0 else 1.0

    print(f"\n── Calibration factors (actual ÷ predicted) ──")
    print(f"  Multiply forecast counts by these to get calibrated estimates:")
    for rtype, factor in sorted(calibration_factors.items()):
        act = int(count_compare.loc[rtype, "count_actual"])
        prd = int(count_compare.loc[rtype, "count_predicted"])
        print(f"  {rtype:20s}  {prd:>6,} × {factor:.3f} = {prd * factor:>8,.0f}  "
              f"(actual={act:,})")

    # ── 6. Plots ─────────────────────────────────────────────
    _plot_count_comparison(count_compare, calibration_factors)
    _plot_monthly_by_type(df_act, df_prd, calibration_factors)
    _plot_binary_summary(actual_came, predicted_come, tp, fn, fp)

    print(f"\n{'=' * 80}")
    return {
        "count_compare": count_compare,
        "binary_stats": {
            "threshold": threshold, "actual_came": actual_came,
            "predicted_come": predicted_come, "tp": tp, "fn": fn, "fp": fp,
            "recall": recall, "precision": precision,
        },
        "df_predicted": df_will,
        "calibration_factors": calibration_factors,
    }


# ── Plot helpers ─────────────────────────────────────────────

def _plot_count_comparison(count_compare, calibration_factors=None):
    """Side-by-side bar: actual vs predicted vs expected vs calibrated counts per type."""
    # Exclude TOTAL row from bar chart
    plot_df = count_compare.drop("TOTAL", errors="ignore").copy()

    # Add calibrated column: predicted × calibration factor
    if calibration_factors:
        plot_df["count_calibrated"] = [
            int(round(plot_df.loc[t, "count_predicted"] * calibration_factors.get(t, 1.0)))
            for t in plot_df.index
        ]

    cols   = ["count_actual", "count_predicted"]
    colors = ["#2196F3", "#FF9800"]
    legend = ["Actual", "Predicted (K-expansion, ≥5%)"]
    if "count_calibrated" in plot_df.columns:
        cols.append("count_calibrated")
        colors.append("#E91E63")
        legend.append("Calibrated")
    if "count_expected" in plot_df.columns:
        cols.append("count_expected")
        colors.append("#4CAF50")
        legend.append("Expected (prob-weighted)")

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_df[cols].plot(kind="bar", ax=ax, color=colors, width=0.80)
    ax.set_title("End-to-End Backtest: Actual vs Predicted vs Calibrated vs Expected",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Retrofit Type")
    ax.legend(legend, fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def _plot_monthly_by_type(df_act, df_prd, calibration_factors=None):
    """One subplot per retrofit type: monthly actual vs predicted vs calibrated."""
    act_monthly = (df_act.groupby(["retrofit_type", "_month"])
                   .size().rename("actual").reset_index())
    prd_monthly = (df_prd.groupby(["retrofit_type", "_month"])
                   .size().rename("predicted").reset_index())
    merged = pd.merge(act_monthly, prd_monthly,
                      on=["retrofit_type", "_month"], how="outer").fillna(0)

    # Add calibrated column
    if calibration_factors:
        merged["calibrated"] = merged.apply(
            lambda r: r["predicted"] * calibration_factors.get(r["retrofit_type"], 1.0),
            axis=1
        )

    type_counts = merged.groupby("retrofit_type")[["actual", "predicted"]].sum()
    types = sorted(type_counts[type_counts["actual"] >= 50].index)
    if not types:
        return

    n_types = len(types)
    fig, axes = plt.subplots(n_types, 1, figsize=(16, 4 * n_types), sharex=True)
    if n_types == 1:
        axes = [axes]

    for ax, rtype in zip(axes, types):
        sub = merged[merged["retrofit_type"] == rtype].sort_values("_month")
        x = sub["_month"].dt.to_timestamp()
        ax.plot(x, sub["actual"], label="Actual", color="#2196F3",
                linewidth=2, marker="o", markersize=3)
        ax.plot(x, sub["predicted"], label="Predicted", color="#FF9800",
                linewidth=2, marker="s", markersize=3, linestyle="--")
        if "calibrated" in sub.columns:
            ax.plot(x, sub["calibrated"], label="Calibrated", color="#E91E63",
                    linewidth=2, marker="D", markersize=3, linestyle="-.")
        ax.set_title(rtype, fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

    axes[-1].set_xlabel("Month")
    fig.suptitle("End-to-End Backtest: Monthly Actual vs Predicted vs Calibrated",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def _plot_binary_summary(actual_came, predicted_come, tp, fn, fp):
    """Bar chart summarising binary model performance."""
    tn = 0  # not computed, just show the key numbers
    labels = ["Actually\ncame", "Predicted\nto come", "True\nPositives",
              "Missed\n(FN)", "False\nAlarm (FP)"]
    values = [actual_came, predicted_come, tp, fn, fp]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                f"{val:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Binary Model: Who Comes for Retrofit?",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Vehicles")
    plt.tight_layout()
    plt.show()

# -- Part C: Export forecast to CSV ------------------------------------


def export_forecast(forecast_results, df_source_a_raw=None):
    """
    Build a combined historical + forecast table and write it to CSV.

    The output table contains:
      - ALL raw source-A rows (historical, including cancelled) with the
        base OUTPUT_BASE_COLUMNS
      - Forecast rows (ML predictions) mapped from source-B data to the same
        base columns, plus OUTPUT_FORECAST_COLUMNS

    Downstream analysis can load the CSV and filter on
    umr_status = 'Forecast' to separate predictions from history.

    Parameters
    ----------
    forecast_results : dict returned by forecast.run_forecast()
    df_source_a_raw  : pd.DataFrame – raw source-A data (all rows including
                       cancelled). If None, only forecast rows are exported.
    """
    if not isinstance(forecast_results, dict):
        forecast_results = {}
    df = forecast_results.get("df_forecast")
    has_forecast = df is not None and not df.empty
    has_source_a = df_source_a_raw is not None and not df_source_a_raw.empty

    if not has_forecast and not has_source_a:
        print("No forecast data and no Source A data to export.")
        return None

    print("\n" + "=" * 80)
    print("EXPORT FORECAST TO CSV")
    print("=" * 80 + "\n")

    all_columns = config.OUTPUT_BASE_COLUMNS + config.OUTPUT_FORECAST_COLUMNS

    # ------------------------------------------------------------------
    # 1. Source A historical rows  (27 base cols + 10 null forecast cols)
    # ------------------------------------------------------------------
    if has_source_a:
        source_a_out = pd.DataFrame()
        for col in config.OUTPUT_BASE_COLUMNS:
            if col in df_source_a_raw.columns:
                source_a_out[col] = df_source_a_raw[col].values
            else:
                source_a_out[col] = None
        # Forecast-metadata columns are null for historical rows.
        # Use np.nan (not None) so numeric cols stay float64 after concat
        # with forecast rows — avoids accidental string conversion.
        for col in config.OUTPUT_FORECAST_COLUMNS:
            source_a_out[col] = np.nan
        print(f"  Source A historical rows: {len(source_a_out):,}")
    else:
        source_a_out = pd.DataFrame(columns=all_columns)
        print("  ⚠ No Source A data provided — exporting forecast rows only")

    # ------------------------------------------------------------------
    # 2. Forecast rows  (Source B → 27 base cols + 10 filled forecast cols)
    # ------------------------------------------------------------------
    if has_forecast:
        meta = forecast_results.get("model_metadata", {})
        bin_approach = meta.get("binary_approach", "unknown")
        bin_model    = meta.get("binary_model", "unknown")
        mc_model     = meta.get("multiclass_model", "N/A")
        dur_model    = meta.get("temporal_duration_model", "N/A")
        opt_thr      = meta.get("optimal_threshold", 0.5)

        print(f"\n  Best procedure used:")
        print(f"    Binary:     {bin_approach} → {bin_model}")
        print(f"    Multiclass: {mc_model}")
        print(f"    Duration:   {dur_model}")
        print(f"    Threshold:  {opt_thr:.3f}")

        prog = pd.DataFrame()

        # ── Source B columns that map 1:1 by name (or _clean suffix) ──
        for col in config.FORECAST_DIRECT_COLUMNS:
            if col in df.columns:
                prog[col] = df[col].values
            elif f"{col}_clean" in df.columns:
                prog[col] = df[f"{col}_clean"].values
            else:
                prog[col] = None

        # ── Renamed mappings (Source A name ← Source B name) ──
        for src_a_col, src_b_col in config.FORECAST_RENAME_COLUMNS.items():
            if src_b_col in df.columns:
                prog[src_a_col] = df[src_b_col].values
            elif f"{src_b_col}_clean" in df.columns:
                prog[src_a_col] = df[f"{src_b_col}_clean"].values
            else:
                prog[src_a_col] = None

        # umr_status -> always Forecast
        prog[config.COL_UMRUSTSTATUS] = config.FORECAST_STATUS_LABEL

        # umr_art <- predicted_type
        prog[config.COL_UMRUESTART] = (
            df[config.COL_PREDICTED_TYPE].values if config.COL_PREDICTED_TYPE in df.columns
            else config.FILLNA_CATEGORICAL
        )
        # umr_start <- estimated_start  (predicted)
        prog[config.COL_START_UR] = (
            pd.to_datetime(df[config.COL_ESTIMATED_START], errors="coerce").values
            if config.COL_ESTIMATED_START in df.columns else pd.NaT
        )
        # umr_av ← estimated_start + predicted_duration_days  (predicted)
        if config.COL_PREDICTED_DURATION in df.columns and config.COL_ESTIMATED_START in df.columns:
            _start = pd.to_datetime(df[config.COL_ESTIMATED_START], errors="coerce")
            prog[config.COL_AV_UR] = (_start + pd.to_timedelta(
                df[config.COL_PREDICTED_DURATION].fillna(0), unit="D")).values
        else:
            prog[config.COL_AV_UR] = pd.NaT

        # ── Columns that are null for forecast rows ──
        for col in config.FORECAST_NULL_COLUMNS:
            prog[col] = None

        # ── Forecast-metadata columns (filled) ──
        prog[config.COL_PROB_COMING] = (
            df[config.COL_PROB_COMING].round(4).values
            if config.COL_PROB_COMING in df.columns else None
        )
        prog[config.COL_TYPE_PROBABILITY] = (
            df[config.COL_TYPE_PROBABILITY].round(4).values
            if config.COL_TYPE_PROBABILITY in df.columns else None
        )
        prog[config.COL_VISIT_NUMBER] = (
            df[config.COL_VISIT_NUMBER].values if config.COL_VISIT_NUMBER in df.columns else 1
        )
        prog[config.COL_CALIBRATION_WEIGHT] = (
            df[config.COL_CALIBRATION_WEIGHT].round(4).values
            if config.COL_CALIBRATION_WEIGHT in df.columns else 1.0
        )
        prog["forecast_binary_approach"]  = bin_approach
        prog["forecast_binary_model"]     = bin_model
        prog["forecast_multiclass_model"] = mc_model
        prog["forecast_duration_model"]   = dur_model
        prog["forecast_threshold"]        = round(opt_thr, 4)
        prog["forecast_timestamp"]        = pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S")

        # Enforce canonical column order
        prog = prog.reindex(columns=all_columns)

        print(f"\n  Forecast rows: {len(prog):,}")
        if config.COL_UMRUESTART in prog.columns:
            print(f"  Predicted types:\n{prog[config.COL_UMRUESTART].value_counts().to_string()}")
    else:
        prog = pd.DataFrame(columns=all_columns)
        print("  ⚠ No forecast data — exporting Source A rows only")

    # ------------------------------------------------------------------
    # 3. Concatenate Source A + Forecast
    # ------------------------------------------------------------------
    out = pd.concat([source_a_out, prog], ignore_index=True)
    out = out.reindex(columns=all_columns)

    print(f"\n  Combined table: {len(out):,} rows "
          f"({len(source_a_out):,} historical + {len(prog):,} forecast)")
    print(f"  Columns ({len(all_columns)}): {all_columns}")

    # ------------------------------------------------------------------
    # 4. Write to CSV
    # ------------------------------------------------------------------
    # Convert date columns to timestamp
    for col in config.DATE_COLUMNS + getattr(config, "SOURCE_B_DATE_COLUMNS", []):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Convert object columns to proper string (replace None/nan → pd.NA)
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].where(out[col].notna(), None).astype("string")

    # Convert float32 → float64 for Parquet compatibility
    for col in out.select_dtypes(include=["float32"]).columns:
        out[col] = out[col].astype("float64")

    # Diagnostics
    print(f"\n  DataFrame dtypes:")
    for col in out.columns:
        null_count = out[col].isna().sum()
        print(f"    {col:40s}  {str(out[col].dtype):15s}  nulls={null_count}")

    # Write CSV
    import os
    os.makedirs(os.path.dirname(config.CSV_OUTPUT_FORECAST), exist_ok=True)
    print(f"\n  Writing to {config.CSV_OUTPUT_FORECAST} ...")
    try:
        out.to_csv(config.CSV_OUTPUT_FORECAST, sep=config.CSV_SEP, index=False)
        print(f"  ✓ {len(out):,} rows written "
              f"({len(source_a_out):,} historical + {len(prog):,} forecast)")
        print(f"  ✓ CSV path: {config.CSV_OUTPUT_FORECAST}")
    except Exception as e:
        print(f"\n  ✗ EXPORT FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return out
