#!/usr/bin/env python3
"""
run_pipeline.py – Full ML pipeline
====================================
Runs the complete 8-step pipeline as a plain Python script.
Checkpoints are saved automatically — if the process is interrupted,
re-running it picks up where it left off.

Usage:
    python run_pipeline.py
"""
import sys, os, gc, time, warnings, traceback, importlib, subprocess
import matplotlib; matplotlib.use("Agg")    # headless — no display needed
import numpy as np
import pandas as pd

# ── Auto-install optional FM/LLM packages if missing ─────────
def _ensure_packages():
    """Install FM/LLM dependencies once (idempotent, pip no-ops if present)."""
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    only_bin = pip + ["--only-binary", ":all:", "numpy<2.4"]
    subprocess.check_call(only_bin + ["torch", "transformers"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(pip + ["chronos-forecasting"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(pip + ["tabpfn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(pip + ["autogluon.tabular[all]"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    import boto3, torch, transformers, chronos  # noqa: F401
    import tabpfn, autogluon.tabular  # noqa: F401
except ImportError:
    print("[run_pipeline] Installing optional FM/LLM packages (one-time) …")
    _ensure_packages()

# ── Suppress noisy warnings BEFORE any library imports ───────

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["XGB_VERBOSITY"] = "0"           # XGBoost C-level verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # TensorFlow (if present)
os.environ["XGBOOST_BUILD_DOC"] = "1"       # silences glibc FutureWarning
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"  # Jupyter debugger noise

warnings.filterwarnings("ignore")
# XGBoost glibc < 2.28 FutureWarning — must also match by message because
# xgboost.core fires it from many call-sites, evading module-only filters.
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*glibc.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*manylinux2014.*", category=FutureWarning)
# LightGBM / sklearn / pandas chatter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── sklearn 1.6 compat: force_all_finite → ensure_all_finite ─
# Older LightGBM / XGBoost sklearn wrappers call check_X_y() and
# check_array() with the deprecated kwarg `force_all_finite`, which was
# renamed to `ensure_all_finite` in sklearn ≥ 1.6.  Monkey-patch BEFORE
# any library imports so every subsequent `from sklearn… import check_X_y`
# picks up the wrapped version.
try:
    import sklearn.utils.validation as _skval

    # Guard: only patch once (survives importlib.reload)
    if not getattr(_skval.check_array, "_sklearn_compat_patched", False):
        _orig_check_array = _skval.check_array
        def _compat_check_array(*args, **kwargs):
            if "force_all_finite" in kwargs:
                kwargs.setdefault("ensure_all_finite",
                                  kwargs.pop("force_all_finite"))
            return _orig_check_array(*args, **kwargs)
        _compat_check_array._sklearn_compat_patched = True
        _skval.check_array = _compat_check_array

    if hasattr(_skval, "check_X_y") and not getattr(_skval.check_X_y, "_sklearn_compat_patched", False):
        _orig_check_X_y = _skval.check_X_y
        def _compat_check_X_y(*args, **kwargs):
            if "force_all_finite" in kwargs:
                kwargs.setdefault("ensure_all_finite",
                                  kwargs.pop("force_all_finite"))
            return _orig_check_X_y(*args, **kwargs)
        _compat_check_X_y._sklearn_compat_patched = True
        _skval.check_X_y = _compat_check_X_y
except Exception:
    pass

# Ensure project root is on sys.path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

# ── Import all modules ───────────────────────────────────────
# Use importlib.import_module() for each submodule explicitly so the
# pipeline is not sensitive to __init__.py bytecode-cache mismatches
# or partial-init issues across environments.
importlib.invalidate_caches()
import src                                                   # package shell
config        = importlib.import_module("src.config")
utils         = importlib.import_module("src.utils")
prepare       = importlib.import_module("src.01_data_cleansing_and_transformation")
fe            = importlib.import_module("src.02_feature_engineering")
models        = importlib.import_module("src.03_data_modeling_and_evaluation_binary")
multiclass    = importlib.import_module("src.04_data_modeling_and_evaluation_multiclass")
temporal      = importlib.import_module("src.06_data_modeling_and_evaluation_time_series")
automl_mod    = importlib.import_module("src.07_data_modeling_and_evaluation_automl")
forecast      = importlib.import_module("src.08_complete_forecast")


def _step(name):
    """Print a timestamped step header."""
    print(f"\n{'═'*70}")
    print(f"  {name}  [{time.strftime('%H:%M:%S')}]")
    print(f"{'═'*70}")
    utils.mem()


def main():
    t0 = time.time()
    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ══════════════════════════════════════════════════════════
    # 01 — Data Upload + Cleansing + Transformation
    # ══════════════════════════════════════════════════════════
    _step("01 — Data Upload + Cleansing + Transformation")

    df_source_a_raw = prepare.load_source_a()
    df_source_b_raw = prepare.load_source_b()

    df_source_a_clean = prepare.cleanse_source_a(df_source_a_raw)
    prepare.visualize_cleansing(df_source_a_clean)
    # Keep df_source_a_raw alive for CSV export (raw rows incl. cancelled)
    gc.collect()

    df_binary = prepare.merge_left_join(df_source_a_clean, df_source_b_raw)
    df_multi = prepare.merge_inner_join(df_source_a_clean, df_source_b_raw)
    del df_source_b_raw; gc.collect()

    prepare.visualize_transformation(df_binary, df_multi, df_source_a_clean)
    utils.validate_cache(df_binary, df_multi)

    # ══════════════════════════════════════════════════════════
    # 02 — Feature Engineering
    # ══════════════════════════════════════════════════════════
    _step("02 — Feature Engineering")

    data_prep = fe.prepare_binary(df_binary)
    fe.visualize_binary(data_prep)

    # ══════════════════════════════════════════════════════════
    # 03 — Binary Classification (Will it come?)
    # ══════════════════════════════════════════════════════════
    _step("03 — Binary Classification (Will it come?)")

    baseline_results = utils.load_checkpoint("baseline_results")
    if baseline_results is None:
        baseline_results = models.binary_classification(data_prep)
        utils.save_checkpoint(baseline_results, "baseline_results")

    # 3.2 Hyperparameter Tuning & Final Selection
    tuned_results = utils.load_checkpoint("tuned_results")
    if tuned_results is None:
        tuned_results = models.hyperparameter_tuning(data_prep)
        utils.save_checkpoint(tuned_results, "tuned_results")

    smote_results = utils.load_checkpoint("smote_results")
    if smote_results is None:
        smote_results = models.smote_classification(data_prep)
        utils.save_checkpoint(smote_results, "smote_results")

    _eval_bundle = utils.load_checkpoint("eval_bundle")
    if _eval_bundle is None:
        models.compare_results(baseline_results, tuned_results, "Baseline", "Tuned")
        models.compare_results(baseline_results, smote_results, "Baseline", "SMOTE")
        ensemble_results = models.ensemble_voting(smote_results)
        featsel_results = models.feature_selection(data_prep, baseline_results)
        _eval_bundle = {"ensemble": ensemble_results, "featsel": featsel_results}
        utils.save_checkpoint(_eval_bundle, "eval_bundle")
    else:
        ensemble_results = _eval_bundle["ensemble"]
        featsel_results = _eval_bundle["featsel"]
    del _eval_bundle; gc.collect()

    baseline_results["approach_name"] = "Baseline"
    tuned_results["approach_name"] = "Hyperparameter Tuned"
    smote_results["approach_name"] = "SMOTE Balanced"
    featsel_results["approach_name"] = "Feature Selection"
    ensemble_results["approach_name"] = "Ensemble"

    binary_results = models.final_model_comparison(
        baseline_results, tuned_results, smote_results,
        featsel_results, ensemble_results
    )

    _approach_sources = {
        "Baseline": baseline_results,
        "Hyperparameter Tuned": tuned_results,
        "SMOTE Balanced": smote_results,
        "Feature Selection": featsel_results,
    }
    _source = _approach_sources.get(binary_results["best_approach"])
    if _source is None:
        _source = smote_results
        binary_results["best_model"] = smote_results["best_model"]
        binary_results["best_approach"] = "SMOTE Balanced"

    for _name, _res in _approach_sources.items():
        if _res is not _source and _name != "Baseline":
            for _mn in list(_res.get("results", {}).keys()):
                _res["results"][_mn].pop("model", None)
                _res["results"][_mn].pop("y_pred_proba", None)
                _res["results"][_mn].pop("y_pred", None)
                _res["results"][_mn].pop("y_pred_opt", None)
    del _approach_sources; gc.collect()
    print(f"Winner: {binary_results['best_approach']} - {binary_results['best_model']}")

    # ══════════════════════════════════════════════════════════
    # 04 — Multiclass Classification (What type?)
    # ══════════════════════════════════════════════════════════
    _step("04 — Multiclass Classification (What type?)")

    multi_results = utils.load_checkpoint("multi_results")
    if multi_results is None:
        source_b_cols = set(df_binary.columns)
        keep_m = [c for c in df_multi.columns
                  if c in source_b_cols or c == config.COL_MULTICLASS_TARGET]
        multi_results = multiclass.multiclass_retrofit_types(df_multi[keep_m])
        utils.save_checkpoint(multi_results, "multi_results")
    gc.collect()

    # 4.3 Visit Frequency & Historical Analysis
    _vk = config.COL_JOIN_KEY if config.COL_JOIN_KEY in df_multi.columns else config.COL_JOIN_KEY_SOURCE_A
    visit_counts = df_multi.groupby(_vk).size()
    multi_results["visit_stats"] = {
        "mean": float(visit_counts.mean()),
        "median": float(visit_counts.median()),
        "distribution": visit_counts.value_counts().sort_index().to_dict(),
    }

    _dcol = config.COL_DERIVAT_CLEAN if config.COL_DERIVAT_CLEAN in df_multi.columns else config.COL_DERIVAT
    if _dcol in df_multi.columns:
        _vd = df_multi.groupby([_vk, _dcol]).size().reset_index(name="_n")
        _vd_total = _vd.groupby(_vk)["_n"].sum().reset_index(name="_total")
        _vd = _vd.merge(_vd_total, on=_vk)
        _deriv_avg = _vd.drop_duplicates(_vk).groupby(_dcol)["_total"].mean()
        multi_results["visit_stats"]["per_derivat"] = _deriv_avg.to_dict()
        multi_results["visit_stats"]["derivat_col"] = _dcol

    _start = config.COL_START_UR
    if _start in df_multi.columns:
        _sorted = df_multi.sort_values([_vk, _start])
        _sorted["_start_dt"] = pd.to_datetime(_sorted[_start], errors="coerce")
        _sorted["_prev_dt"] = _sorted.groupby(_vk)["_start_dt"].shift(1)
        _sorted["_gap"] = (_sorted["_start_dt"] - _sorted["_prev_dt"]).dt.days
        _intervals = _sorted["_gap"].dropna()
        _intervals = _intervals[(_intervals > 0) & (_intervals < 1000)]
        if len(_intervals) > 0:
            multi_results["visit_stats"]["inter_visit_median_days"] = float(_intervals.median())
            multi_results["visit_stats"]["inter_visit_mean_days"] = float(_intervals.mean())

        if _dcol in _sorted.columns and len(_intervals) > 0:
            _sorted_valid = _sorted[_sorted["_gap"].between(1, 999, inclusive="both")]
            _deriv_gap = _sorted_valid.groupby(_dcol)["_gap"].median()
            multi_results["visit_stats"]["inter_visit_per_derivat"] = _deriv_gap.to_dict()

    _type_col = config.COL_MULTICLASS_TARGET
    if _start in df_multi.columns and _type_col in df_multi.columns:
        _hd = df_multi[[_start, _type_col]].copy()
        _hd[_start] = pd.to_datetime(_hd[_start], errors="coerce")
        _hd = _hd.dropna(subset=[_start])
        _hd["_month"] = _hd[_start].dt.to_period("M")
        _hist_pivot = (_hd.groupby(["_month", _type_col]).size()
                       .unstack(fill_value=0).sort_index())
        multi_results["hist_monthly_pivot"] = _hist_pivot

    # ══════════════════════════════════════════════════════════
    # 05 — Temporal Regression (How long?)
    # 06 — Time Series (Monthly aggregates)
    # ══════════════════════════════════════════════════════════
    _step("05/06 — Temporal Regression + Time Series")

    temporal_results = utils.load_checkpoint("temporal_results")
    if temporal_results is None:
        source_b_cols = set(df_binary.columns)
        target_src = {config.COL_AV_UR, config.COL_START_UR,
                      config.COL_MULTICLASS_TARGET,
                      config.COL_UMRUSTSTATUS}
        keep_t = [c for c in df_multi.columns
                  if c in source_b_cols or c in target_src]
        temporal_results = temporal.temporal_prediction(df_multi[keep_t])
        utils.save_checkpoint(temporal_results, "temporal_results")
    gc.collect()

    # ══════════════════════════════════════════════════════════
    # 07 — AutoML Benchmark (AutoGluon + TabPFN vs Manual)
    # ══════════════════════════════════════════════════════════
    _step("07 — AutoML Benchmark (AutoGluon + TabPFN vs Manual)")

    automl_results = utils.load_checkpoint("automl_results")
    if automl_results is None:
        # prepare_multiclass / prepare_temporal produce the same splits
        # that modules 04/06 used internally
        source_b_cols_m = set(df_binary.columns)
        keep_m = [c for c in df_multi.columns
                  if c in source_b_cols_m or c == config.COL_MULTICLASS_TARGET]
        data_prep_multi = fe.prepare_multiclass(df_multi[keep_m])

        target_src_t = {config.COL_AV_UR, config.COL_START_UR,
                        config.COL_MULTICLASS_TARGET,
                        config.COL_UMRUSTSTATUS}
        keep_t = [c for c in df_multi.columns
                  if c in source_b_cols_m or c in target_src_t]
        data_prep_temporal = fe.prepare_temporal(df_multi[keep_t])

        automl_results = automl_mod.run_automl_benchmark(
            data_prep_binary=data_prep,
            data_prep_multiclass=data_prep_multi,
            data_prep_temporal=data_prep_temporal,
            binary_results=binary_results,
            multi_results=multi_results,
            temporal_results=temporal_results,
        )
        utils.save_checkpoint(automl_results, "automl_results")
        del data_prep_multi, data_prep_temporal
    gc.collect()

    # ══════════════════════════════════════════════════════════
    # 08 — Forecast + Backtest + Export CSV
    # ══════════════════════════════════════════════════════════
    _step("08 — Forecast + Backtest + Export CSV")

    # 8a. Backtest
    backtest = utils.load_checkpoint("backtest")
    if backtest is None:
        backtest = forecast.evaluate_retrofits(
            df_binary, df_multi, data_prep, binary_results, _source,
            multi_results, temporal_results,
            automl_results=automl_results,
        )
        utils.save_checkpoint(backtest, "backtest")
    del df_multi; gc.collect()

    # 8b. Forecast (passes automl_results so best model is used)
    forecast_results = utils.load_checkpoint("forecast_results")
    if forecast_results is None:
        forecast_results = forecast.run_forecast(
            df_binary, data_prep, binary_results, _source,
            multi_results, temporal_results,
            calibration_factors=backtest.get("calibration_factors"),
            automl_results=automl_results,
        )
        utils.save_checkpoint(forecast_results, "forecast_results")
    gc.collect()

    # 8c. Export to CSV
    if forecast_results is not None:
        forecast.export_forecast(forecast_results, df_source_a_raw=df_source_a_raw)
    else:
        print("  ⚠ No forecast produced (0 vehicles above threshold) "
              "— exporting Source A-only table")
        forecast.export_forecast({}, df_source_a_raw=df_source_a_raw)

    # Clean up AutoGluon model artifacts from /tmp
    automl_mod.cleanup_all()

    # ── Done ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*70}")
    print(f"  Pipeline completed in {elapsed/60:.1f} min")
    print(f"{'═'*70}")
    utils.mem()


if __name__ == "__main__":
    main()
