# ============================================================
# 06_data_modeling_and_evaluation_time_series.py
#
# Aggregate time-series forecasting for retrofit counts / duration.
#
# Block A — Models (Baseline): Classical TS forecasters
#   SMA, SES, ETS, ARIMA, Prophet, Croston, SBA
#   _run_ts_benchmark
#
# Block B — Optimization: ML regressors on lag features
#   _run_ml_on_lags  (11 ML regressors, expanding-window CV)
#
# Block B2 — TabPFN on lag features
#   _run_tabpfn_ts_benchmark (pre-trained tabular foundation model)
#
# Block B3 — AutoGluon on lag features
#   _run_autogluon_ts_benchmark (AutoML ensemble on lag features)
#
# Block C — Foundation Models / LLMs
#   Chronos T5 (zero-shot probabilistic forecast)
#   LLM-prompted forecast via AWS Bedrock
#   TIME-LLM (frozen GPT-2 backbone, Jin et al. ICLR 2024)
#
# Block D — Evaluation & Orchestration
#   _run_all_ts_benchmarks — unified ranking across all families
#   temporal_prediction    — main entry point (calls 05 regression + TS)
#   compare_all_ts         — summary printer
# ============================================================
import gc
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, SGDRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor

from . import config

# Optional heavy imports (graceful degradation)
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False

try:
    from prophet import Prophet
    _PROPHET = True
except ImportError:
    _PROPHET = False

# Keras / DL removed — overfits on ~20 monthly samples

try:
    import torch
    from chronos import ChronosPipeline
    _CHRONOS = True
except Exception:
    _CHRONOS = False

try:
    import boto3 as _boto3_mod
    _BEDROCK = True
except Exception:
    _BEDROCK = False

try:
    from tabpfn import TabPFNRegressor as _TabPFNReg
    _TABPFN_TS = True
except ImportError:
    _TABPFN_TS = False


def _make_tabpfn_regressor_ts(X_tr, y_tr, **kwargs):
    """Create + fit TabPFNRegressor, falling back to v2 if v2.5 fails."""
    try:
        reg = _TabPFNReg(**kwargs)
        reg.fit(X_tr, y_tr)
        return reg
    except Exception as exc:
        if "download" in str(exc).lower() or "authentication" in str(exc).lower() \
                or "gated" in str(exc).lower() or "V2_5" in str(exc):
            from tabpfn.constants import ModelVersion
            print("  [TabPFN] v2.5 unavailable (gated model) → falling back to v2")
            reg = _TabPFNReg.create_default_for_version(ModelVersion.V2)
            reg.fit(X_tr, y_tr)
            return reg
        raise

try:
    from autogluon.tabular import TabularPredictor as _AGPredictor
    _AUTOGLUON_TS = True
except ImportError:
    _AUTOGLUON_TS = False

try:
    import torch.nn as _nn
    from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
    _TRANSFORMERS = True
except Exception:
    _TRANSFORMERS = False

import importlib as _il
_fe = _il.import_module(".02_feature_engineering", __package__)
_TEMPORAL_LEAK          = _fe._TEMPORAL_LEAK
_prepare_monthly_series = _fe._prepare_monthly_series
_classify_demand        = _fe._classify_demand
_create_lag_features    = _fe._create_lag_features
_create_duration_target = _fe._create_duration_target

_m05 = _il.import_module(
    ".05_data_modeling_and_evaluation_temporal", __package__)


# ══════════════════════════════════════════════════════════════
# Block A — Models (Baseline): Classical TS Forecasters
# ══════════════════════════════════════════════════════════════

# ── A.1  Forecaster functions ────────────────────────────────

def _sma_forecast(train, h, window=3):
    """Simple Moving Average."""
    preds = []
    history = list(train)
    for _ in range(h):
        val = np.mean(history[-window:])
        preds.append(val)
        history.append(val)
    return np.array(preds)



def _ses_forecast(train, h, alpha=0.3):
    """Simple Exponential Smoothing."""
    if not _STATSMODELS:
        return np.full(h, np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SimpleExpSmoothing(train, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, optimized=False)
        return fit.forecast(h)
    except Exception:
        return np.full(h, np.nan)



def _ets_forecast(train, h, seasonal_periods=12):
    """Holt-Winters Exponential Smoothing (ETS).

    Uses additive trend + additive seasonality when enough data is
    available (≥ 2 full seasonal cycles); otherwise falls back to
    additive-trend-only (no seasonality).
    """
    if not _STATSMODELS:
        return np.full(h, np.nan)
    try:
        if len(train) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(optimized=True)
        return fit.forecast(h).clip(0)
    except Exception:
        return np.full(h, np.nan)



def _arima_forecast(train, h, order=(1, 1, 1)):
    """ARIMA forecast on log-transformed series."""
    if not _STATSMODELS:
        return np.full(h, np.nan)
    try:
        safe = np.log1p(np.maximum(train, 0))
        model = ARIMA_Model(safe, order=order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit()
        preds_log = fit.forecast(h)
        return np.expm1(preds_log).clip(0)
    except Exception:
        return np.full(h, np.nan)



def _prophet_forecast(train_series, h):
    """Prophet forecast (train_series = pd.Series with DatetimeIndex)."""
    if not _PROPHET:
        return np.full(h, np.nan)
    try:
        dfp = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False)
            m.fit(dfp)
        future = m.make_future_dataframe(periods=h, freq="MS")
        fcst = m.predict(future)
        return fcst["yhat"].iloc[-h:].values.clip(0)
    except Exception:
        return np.full(h, np.nan)



def _croston_forecast(train, h, alpha=0.1):
    """Croston's method for intermittent demand."""
    n = len(train)
    if n < 2:
        return np.full(h, np.mean(train) if len(train) else 0)
    # initialise
    demand = train[train > 0]
    q = demand.iloc[0] if len(demand) else 0.0  # level of non-zero demand
    a = 1.0  # interval between non-zero demands
    periods_since = 0
    for i in range(1, n):
        periods_since += 1
        if train.iloc[i] > 0:
            q = alpha * train.iloc[i] + (1 - alpha) * q
            a = alpha * periods_since + (1 - alpha) * a
            periods_since = 0
    forecast_val = q / a if a > 0 else 0
    return np.full(h, max(forecast_val, 0))



def _sba_forecast(train, h, alpha=0.1):
    """Syntetos-Boylan Approximation (bias-corrected Croston)."""
    croston_preds = _croston_forecast(train, h, alpha)
    return croston_preds * (1 - alpha / 2)


# ── A.2  Benchmark runner ────────────────────────────────────

def _run_ts_benchmark(monthly, target_col=config.COL_MEAN_TARGET, label=""):
    """
    Evaluate traditional TS models on the aggregate monthly series.
    Returns a DataFrame of metrics for thesis comparison.
    """
    series = monthly[target_col].values
    n = len(series)
    if n < 12:
        print(f"  ⚠ TS benchmark skipped for '{label}': only {n} months of data")
        return None, None
    # Train/test split: last 20% of months
    split = max(int(n * 0.8), 6)
    train_vals = pd.Series(series[:split])
    test_vals = series[split:]
    h = len(test_vals)
    if h < 1:
        return None, None

    demand_class = _classify_demand(train_vals)
    print(f"\n  ── Time-Series Benchmark ({label}) ──")
    print(f"     Months: {n} total, {split} train, {h} test")
    print(f"     Demand pattern: {demand_class}")

    models = {
        "SMA(3)": _sma_forecast(train_vals, h, window=3),
        "SMA(6)": _sma_forecast(train_vals, h, window=6),
        "SES": _ses_forecast(train_vals.values, h),
        "ETS(add)": _ets_forecast(train_vals.values, h),
        "ARIMA(1,1,1)": _arima_forecast(train_vals.values, h),
        "Croston": _croston_forecast(train_vals, h),
        "SBA": _sba_forecast(train_vals, h),
    }
    # Prophet needs DatetimeIndex
    if _PROPHET and config.COL_TS_MONTH in monthly.columns:
        train_prophet = pd.Series(
            train_vals.values,
            index=pd.to_datetime(monthly[config.COL_TS_MONTH].iloc[:split].values))
        models["Prophet"] = _prophet_forecast(train_prophet, h)

    rows = []
    best_preds_future = None
    best_mae = float("inf")
    best_name = None
    for name, preds in models.items():
        if np.any(np.isnan(preds)):
            continue
        mae = mean_absolute_error(test_vals, preds)
        rmse = np.sqrt(mean_squared_error(test_vals, preds))
        r2 = r2_score(test_vals, preds)
        print(f"     {name:20s}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  R²={r2:.4f}")
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
                      "family": "Traditional TS"})
        if mae < best_mae:
            best_mae = mae
            best_name = name
    if not rows:
        return None, None

    # Retrain best classical model on ALL data and forecast 18 months ahead
    future_h = 18
    full_vals = pd.Series(series)
    future_forecast = None
    if best_name:
        _forecasters = {
            "SMA(3)": lambda s, h: _sma_forecast(s, h, window=3),
            "SMA(6)": lambda s, h: _sma_forecast(s, h, window=6),
            "SES": lambda s, h: _ses_forecast(s.values, h),
            "ETS(add)": lambda s, h: _ets_forecast(s.values, h),
            "ARIMA(1,1,1)": lambda s, h: _arima_forecast(s.values, h),
            "Croston": lambda s, h: _croston_forecast(s, h),
            "SBA": lambda s, h: _sba_forecast(s, h),
        }
        if best_name == "Prophet" and _PROPHET and config.COL_TS_MONTH in monthly.columns:
            full_prophet = pd.Series(
                full_vals.values,
                index=pd.to_datetime(monthly[config.COL_TS_MONTH].values))
            future_forecast = _prophet_forecast(full_prophet, future_h)
        elif best_name in _forecasters:
            future_forecast = _forecasters[best_name](full_vals, future_h)

        if future_forecast is not None and not np.any(np.isnan(future_forecast)):
            # Build month index for the future predictions
            last_month = pd.Timestamp(monthly[config.COL_TS_MONTH].iloc[-1]) if config.COL_TS_MONTH in monthly.columns else pd.Timestamp.now()
            future_months = pd.date_range(
                last_month + pd.offsets.MonthBegin(1), periods=future_h, freq="MS")
            best_preds_future = pd.Series(future_forecast, index=future_months,
                                          name=f"ts_{best_name}")
            print(f"     → {best_name} retrained on all data, forecasting {future_h} months ahead")
        else:
            future_forecast = None

    return pd.DataFrame(rows).sort_values("MAE"), best_preds_future


# ══════════════════════════════════════════════════════════════
# Block B — Optimization: ML Regressors on Lag Features
# ══════════════════════════════════════════════════════════════

# ── B.1  ML regressors on supervised lag features ─────────────

def _run_ml_on_lags(monthly, target_col=config.COL_MEAN_TARGET, n_lags=12, label=""):
    """Evaluate ML regressors on supervised lag features with expanding-window CV.

    Walk-forward validation: the training window expands each fold while
    a fixed-size test window rolls forward.  This is more robust than a
    single 80/20 split on a ~30-point series.
    """
    series = monthly[target_col]
    if len(series) < n_lags + 6:
        print(f"  ⚠ ML-on-lags skipped for '{label}': not enough data ({len(series)} months)")
        return None
    X, y = _create_lag_features(series, n_lags)

    # Expanding-window (walk-forward) CV
    min_train = max(n_lags + 2, len(y) // 3)   # at least n_lags+2 train samples
    test_size = max(1, len(y) // 6)             # ~17 % per fold
    n_folds = max(1, (len(y) - min_train) // test_size)
    folds = []
    for k in range(n_folds):
        te_end = min_train + (k + 1) * test_size
        te_start = te_end - test_size
        if te_end > len(y):
            break
        folds.append((slice(0, te_start), slice(te_start, te_end)))
    # Always include the tail as the last fold if not already covered
    if folds and folds[-1][1].stop < len(y):
        last_te_start = folds[-1][1].stop
        folds.append((slice(0, last_te_start), slice(last_te_start, len(y))))
    if not folds:
        folds = [(slice(0, min_train), slice(min_train, len(y)))]

    def _model_factory():
        return {
            "LinearReg": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01, max_iter=3000),
            "SGD": SGDRegressor(max_iter=2000, tol=1e-4, random_state=42),
            "RF(100)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2),
            "GBR": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42,
                                          verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42,
                                        verbose=-1),
            "SVR(rbf)": SVR(kernel="rbf", C=10.0, gamma="scale"),
            "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                early_stopping=True, random_state=42),
        }

    print(f"\n  ── ML on Lag Features ({label}) — {len(folds)}-fold walk-forward CV ──")
    # Accumulate per-fold metrics
    fold_metrics = {name: [] for name in _model_factory()}
    for tr_sl, te_sl in folds:
        X_tr, X_te = X[tr_sl], X[te_sl]
        y_tr, y_te = y[tr_sl], y[te_sl]
        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        for name, model in _model_factory().items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr_s, y_tr)
                preds = model.predict(X_te_s).clip(0)
                fold_metrics[name].append({
                    "mae": mean_absolute_error(y_te, preds),
                    "rmse": np.sqrt(mean_squared_error(y_te, preds)),
                    "r2": r2_score(y_te, preds),
                })
            except Exception:
                pass  # skip failed folds for this model

    rows = []
    for name, metrics_list in fold_metrics.items():
        if not metrics_list:
            continue
        avg_mae = np.mean([m["mae"] for m in metrics_list])
        avg_rmse = np.mean([m["rmse"] for m in metrics_list])
        avg_r2 = np.mean([m["r2"] for m in metrics_list])
        print(f"     {name:20s}  MAE={avg_mae:8.2f}  RMSE={avg_rmse:8.2f}  R²={avg_r2:.4f}")
        rows.append({"Model": f"Lag-{name}", "MAE": avg_mae, "RMSE": avg_rmse,
                      "R2": avg_r2, "family": "ML on Lags"})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("MAE")


# ══════════════════════════════════════════════════════════════
# Block B2 — TabPFN on Lag Features
# ══════════════════════════════════════════════════════════════

def _run_tabpfn_ts_benchmark(monthly, target_col=config.COL_MEAN_TARGET,
                              n_lags=12, label=""):
    """Evaluate TabPFN regressor on supervised lag features.

    Uses the same expanding-window walk-forward CV as _run_ml_on_lags.
    TabPFN is a pre-trained tabular foundation model — zero-shot, no tuning.
    """
    if not _TABPFN_TS:
        print(f"  ⚠ TabPFN-TS skipped (tabpfn package not installed)")
        return None

    series = monthly[target_col]
    if len(series) < n_lags + 6:
        print(f"  ⚠ TabPFN-TS skipped for '{label}': not enough data ({len(series)} months)")
        return None
    X, y = _create_lag_features(series, n_lags)

    # Expanding-window (walk-forward) CV — same logic as _run_ml_on_lags
    min_train = max(n_lags + 2, len(y) // 3)
    test_size = max(1, len(y) // 6)
    n_folds = max(1, (len(y) - min_train) // test_size)
    folds = []
    for k in range(n_folds):
        te_end = min_train + (k + 1) * test_size
        te_start = te_end - test_size
        if te_end > len(y):
            break
        folds.append((slice(0, te_start), slice(te_start, te_end)))
    if folds and folds[-1][1].stop < len(y):
        last_te_start = folds[-1][1].stop
        folds.append((slice(0, last_te_start), slice(last_te_start, len(y))))
    if not folds:
        folds = [(slice(0, min_train), slice(min_train, len(y)))]

    print(f"\n  ── TabPFN on Lag Features ({label}) — {len(folds)}-fold walk-forward CV ──")
    fold_metrics = []
    for tr_sl, te_sl in folds:
        X_tr, X_te = X[tr_sl], X[te_sl]
        y_tr, y_te = y[tr_sl], y[te_sl]
        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg = _make_tabpfn_regressor_ts(X_tr_s, y_tr, device="cpu")
            preds = reg.predict(X_te_s).clip(0)
            fold_metrics.append({
                "mae": mean_absolute_error(y_te, preds),
                "rmse": np.sqrt(mean_squared_error(y_te, preds)),
                "r2": r2_score(y_te, preds),
            })
        except Exception:
            pass

    if not fold_metrics:
        return None

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_r2 = np.mean([m["r2"] for m in fold_metrics])
    print(f"     {'TabPFN':20s}  MAE={avg_mae:8.2f}  RMSE={avg_rmse:8.2f}  R²={avg_r2:.4f}")

    return pd.DataFrame([{
        "Model": "Lag-TabPFN", "MAE": avg_mae, "RMSE": avg_rmse,
        "R2": avg_r2, "family": "TabPFN",
    }])


# ── Block B3: AutoGluon on lag features ──────────────────────

def _run_autogluon_ts_benchmark(monthly, target_col=config.COL_MEAN_TARGET,
                                n_lags=12, label=""):
    """Evaluate AutoGluon on supervised lag features with expanding-window CV.

    Uses the same walk-forward validation as _run_ml_on_lags.
    AutoGluon trains an ensemble of models and returns the best.
    """
    if not _AUTOGLUON_TS:
        return None

    series = monthly[target_col]
    if len(series) < n_lags + 6:
        print(f"  ⚠ AutoGluon-TS skipped for '{label}': not enough data ({len(series)} months)")
        return None
    X, y = _create_lag_features(series, n_lags)

    # Expanding-window (walk-forward) CV — same logic as _run_ml_on_lags
    min_train = max(n_lags + 2, len(y) // 3)
    test_size = max(1, len(y) // 6)
    n_folds = max(1, (len(y) - min_train) // test_size)
    folds = []
    for k in range(n_folds):
        te_end = min_train + (k + 1) * test_size
        te_start = te_end - test_size
        if te_end > len(y):
            break
        folds.append((slice(0, te_start), slice(te_start, te_end)))
    if folds and folds[-1][1].stop < len(y):
        last_te_start = folds[-1][1].stop
        folds.append((slice(0, last_te_start), slice(last_te_start, len(y))))
    if not folds:
        folds = [(slice(0, min_train), slice(min_train, len(y)))]

    print(f"\n  ── AutoGluon on Lag Features ({label}) — {len(folds)}-fold walk-forward CV ──")
    _ag_label = "__target__"
    _ag_time = 120  # 2 min per fold — enough for ~30 samples
    fold_metrics = []
    import shutil, os
    for fi, (tr_sl, te_sl) in enumerate(folds):
        X_tr, X_te = X[tr_sl], X[te_sl]
        y_tr, y_te = y[tr_sl], y[te_sl]
        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        train_df = pd.DataFrame(X_tr_s, columns=[f"lag_{i}" for i in range(X_tr_s.shape[1])])
        train_df[_ag_label] = y_tr
        test_df = pd.DataFrame(X_te_s, columns=[f"lag_{i}" for i in range(X_te_s.shape[1])])
        test_df[_ag_label] = y_te
        save_path = os.path.join(config.AUTOML_SAVE_DIR, f"ts_lags_fold{fi}")
        if os.path.isdir(save_path):
            shutil.rmtree(save_path, ignore_errors=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictor = _AGPredictor(
                    label=_ag_label, problem_type="regression",
                    eval_metric="mean_absolute_error",
                    path=save_path, verbosity=0,
                ).fit(train_df, time_limit=_ag_time, presets="best_quality")
            preds = predictor.predict(test_df).values.clip(0)
            fold_metrics.append({
                "mae": mean_absolute_error(y_te, preds),
                "rmse": np.sqrt(mean_squared_error(y_te, preds)),
                "r2": r2_score(y_te, preds),
            })
        except Exception:
            pass
        finally:
            if os.path.isdir(save_path):
                shutil.rmtree(save_path, ignore_errors=True)

    if not fold_metrics:
        return None

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_r2 = np.mean([m["r2"] for m in fold_metrics])
    print(f"     {'AutoGluon':20s}  MAE={avg_mae:8.2f}  RMSE={avg_rmse:8.2f}  R²={avg_r2:.4f}")

    return pd.DataFrame([{
        "Model": "Lag-AutoGluon", "MAE": avg_mae, "RMSE": avg_rmse,
        "R2": avg_r2, "family": "AutoGluon",
    }])


# ══════════════════════════════════════════════════════════════
# Block C — Foundation Models / LLMs
# ══════════════════════════════════════════════════════════════

# ── C.1  Chronos T5 (Amazon, zero-shot probabilistic) ────────

def _chronos_forecast(train, h, model_id="amazon/chronos-t5-small",
                      num_samples=20):
    """Zero-shot probabilistic forecast with a pretrained Chronos T5 model.

    Parameters
    ----------
    train : array-like   – historical values (the context window).
    h     : int          – forecast horizon (number of steps ahead).
    model_id : str       – HuggingFace model id, e.g. "amazon/chronos-t5-tiny".
    num_samples : int    – number of sample paths for the probabilistic forecast.

    Returns
    -------
    median : np.ndarray of shape (h,) – point forecast (median of samples).
    intervals : dict | None           – {"q10": ..., "q90": ..., "samples": ...}
    """
    if not _CHRONOS:
        return np.full(h, np.nan), None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float32,
        )
        context = torch.tensor(train, dtype=torch.float32).unsqueeze(0)
        forecast_samples = pipeline.predict(
            context, prediction_length=h, num_samples=num_samples,
        )
        # forecast_samples shape: (1, num_samples, h)
        samples = forecast_samples[0].numpy()
        median = np.quantile(samples, 0.5, axis=0).clip(0)
        q10 = np.quantile(samples, 0.1, axis=0).clip(0)
        q90 = np.quantile(samples, 0.9, axis=0).clip(0)
        return median, {"q10": q10, "q90": q90, "samples": samples}
    except Exception as e:
        print(f"  ⚠ Chronos ({model_id}) failed: {e}")
        return np.full(h, np.nan), None



def _run_chronos_benchmark(monthly, target_col=config.COL_MEAN_TARGET, label=""):
    """Evaluate Chronos foundation models on the aggregate monthly series.

    Benchmarks multiple T5 sizes (tiny / small / base) with probabilistic
    forecasts, returning a DataFrame comparable to _run_ts_benchmark().
    """
    if not _CHRONOS:
        print(f"  ⚠ Chronos benchmark skipped (chronos-forecasting not installed)")
        return None

    series = monthly[target_col].values
    n = len(series)
    if n < 12:
        print(f"  ⚠ Chronos benchmark skipped for '{label}': only {n} months")
        return None

    split = max(int(n * 0.8), 6)
    train_vals = series[:split]
    test_vals = series[split:]
    h = len(test_vals)
    if h < 1:
        return None

    print(f"\n  ── Chronos Foundation Models ({label}) ──")

    model_sizes = config.CHRONOS_MODEL_SIZES
    rows = []
    for size in model_sizes:
        model_id = f"amazon/chronos-t5-{size}"
        preds, intervals = _chronos_forecast(
            train_vals, h,
            model_id=model_id,
            num_samples=config.CHRONOS_NUM_SAMPLES,
        )
        if np.any(np.isnan(preds)):
            continue
        mae = mean_absolute_error(test_vals, preds)
        rmse = np.sqrt(mean_squared_error(test_vals, preds))
        r2 = r2_score(test_vals, preds)
        print(f"     Chronos-{size:8s}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  R²={r2:.4f}")
        rows.append({
            "Model": f"Chronos-{size}", "MAE": mae, "RMSE": rmse,
            "R2": r2, "family": "Foundation Model",
        })

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("MAE")


# ── C.1b  Chronos-Bolt (local, amazon/chronos-bolt-base) ──────

def _run_chronos_bolt_benchmark(monthly, target_col=config.COL_MEAN_TARGET,
                                label=""):
    """Evaluate Chronos-Bolt (distilled) on the aggregate monthly series.

    Uses the same local ChronosPipeline as T5 variants but with the
    Bolt models (faster, distilled from the original Chronos T5).
    """
    if not _CHRONOS:
        print(f"  ⚠ Chronos-Bolt benchmark skipped (chronos-forecasting not installed)")
        return None

    series = monthly[target_col].values
    n = len(series)
    if n < 12:
        print(f"  ⚠ Chronos-Bolt benchmark skipped for '{label}': only {n} months")
        return None

    split = max(int(n * 0.8), 6)
    train_vals = series[:split]
    test_vals = series[split:]
    h = len(test_vals)
    if h < 1:
        return None

    print(f"\n  ── Chronos-Bolt Foundation Models ({label}) ──")
    bolt_sizes = config.CHRONOS_BOLT_SIZES
    rows = []
    for size in bolt_sizes:
        model_id = f"amazon/chronos-bolt-{size}"
        preds, intervals = _chronos_forecast(
            train_vals, h,
            model_id=model_id,
            num_samples=config.CHRONOS_NUM_SAMPLES,
        )
        if np.any(np.isnan(preds)):
            continue
        mae = mean_absolute_error(test_vals, preds)
        rmse = np.sqrt(mean_squared_error(test_vals, preds))
        r2 = r2_score(test_vals, preds)
        print(f"     Bolt-{size:8s}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  R²={r2:.4f}")
        rows.append({
            "Model": f"Chronos-Bolt-{size}", "MAE": mae, "RMSE": rmse,
            "R2": r2, "family": "Foundation Model",
        })

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("MAE")


# ── C.2  LLM-prompted forecast (AWS Bedrock) ─────────────────

def _llm_ts_forecast(history, horizon, label=""):
    """Ask an LLM to forecast the next *horizon* values.

    Sends the last ``LLM_TS_N_LAST`` months of *history* as a text prompt
    to ``LLM_CHAT_MODEL`` via Amazon Bedrock and parses a JSON array.
    """
    import json, re, boto3

    n_last = min(config.LLM_TS_N_LAST, len(history))
    recent = history[-n_last:]
    lines = "\n".join(f"  month {i+1}: {v:.2f}" for i, v in enumerate(recent))

    prompt = (
        f"You are an expert time-series forecaster.\n"
        f"Below are the last {n_last} monthly values of {label}:\n"
        f"{lines}\n\n"
        f"Forecast the next {horizon} months.\n"
        f"Return ONLY a JSON array of {horizon} numbers, e.g. [1.2, 3.4, ...].\n"
        f"No explanation, no markdown, just the JSON array."
    )

    client = boto3.client("bedrock-runtime",
                          region_name=config.BEDROCK_REGION)
    resp = client.converse(
        modelId=config.LLM_CHAT_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": config.LLM_TEMPERATURE,
                         "maxTokens": 1024},
    )
    # Robust extraction: some Bedrock models return content blocks
    # without a "text" key (e.g. toolUse, image). Search all blocks.
    content_blocks = resp.get("output", {}).get("message", {}).get("content", [])
    raw = None
    for block in content_blocks:
        if "text" in block:
            raw = block["text"].strip()
            break
    if raw is None:
        # Fallback: try converting the entire first block to string
        if content_blocks:
            raw = str(content_blocks[0]).strip()
        else:
            raise ValueError("Empty response from Bedrock Converse")
    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Robust JSON parsing: try direct parse, then regex extraction
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON array from the response text
        match = re.search(r'\[[\d\s.,\-eE+]+\]', raw)
        if match:
            arr = json.loads(match.group())
        else:
            # Last resort: extract all numbers from the text
            nums = re.findall(r'[\-+]?\d+\.?\d*(?:[eE][\-+]?\d+)?', raw)
            if len(nums) >= horizon:
                arr = [float(n) for n in nums[:horizon]]
            else:
                raise ValueError(
                    f"Could not parse {horizon} numbers from LLM response: "
                    f"{raw[:200]}")
    return np.array(arr, dtype=np.float64)[:horizon]



def _run_llm_ts_benchmark(monthly, target_col=config.COL_MEAN_TARGET, label=""):
    """Evaluate LLM forecast on the aggregate monthly series (via Bedrock)."""
    if not _BEDROCK:
        print(f"  ⚠ LLM TS benchmark skipped (boto3 not installed)")
        return None

    series = monthly[target_col].values
    n = len(series)
    if n < 12:
        print(f"  \u26a0 LLM TS benchmark skipped for '{label}': only {n} months")
        return None

    split = max(int(n * 0.8), 6)
    train_vals = series[:split]
    test_vals = series[split:]
    h = len(test_vals)
    if h < 1:
        return None

    print(f"\n  \u2500\u2500 LLM Forecast ({label} \u2014 {config.LLM_CHAT_MODEL}) \u2500\u2500")
    try:
        preds = _llm_ts_forecast(train_vals, h, label=label)
        preds = np.clip(preds, 0, None)           # no negative counts
        mae = mean_absolute_error(test_vals, preds)
        rmse = np.sqrt(mean_squared_error(test_vals, preds))
        r2 = r2_score(test_vals, preds)
        print(f"     LLM ({config.LLM_CHAT_MODEL})  MAE={mae:8.2f}  "
              f"RMSE={rmse:8.2f}  R\u00b2={r2:.4f}")
        return pd.DataFrame([{
            "Model": f"LLM ({config.LLM_CHAT_MODEL})",
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "family": "LLM Forecast",
        }])
    except Exception as exc:
        print(f"  \u2717 LLM TS forecast failed: {exc}")
        return None


# ── C.3  TIME-LLM (frozen GPT-2, Jin et al. ICLR 2024) ──────

if _TRANSFORMERS:

    class _RevIN(_nn.Module):
        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.gamma = _nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = _nn.Parameter(torch.zeros(1, 1, num_features))

        def forward(self, x, mode):
            # x: [B, L, C]
            if mode == "norm":
                self.mean = x.mean(dim=1, keepdim=True)
                self.std = x.std(dim=1, keepdim=True) + self.eps
                x = (x - self.mean) / self.std
                return self.gamma * x + self.beta

            elif mode == "denorm":
                x = (x - self.beta) / (self.gamma + self.eps)
                return x * self.std + self.mean

    class _Patching(_nn.Module):
        def __init__(self, patch_len, stride):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride

        def forward(self, x):
            # x: [B, L, C]
            B, L, C = x.shape
            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            # unfold output: [B, N, C, patch_len]
            patches = patches.contiguous()
            B, N, C, P = patches.shape
            return patches.view(B, N, C * P)

    class _ReprogrammingLayer(_nn.Module):
        """Map time-series patch embeddings into GPT-2 token space."""

        def __init__(self, patch_dim, gpt_dim, dropout=0.0):
            super().__init__()
            self.proj = _nn.Linear(patch_dim, gpt_dim)
            self.dropout = _nn.Dropout(dropout)

        def forward(self, x):
            # x: [B, N, patch_dim]
            return self.dropout(self.proj(x))

    class _TimeLLM(_nn.Module):
        def __init__(self, cfg):
            super().__init__()

            self.seq_len = cfg.seq_len
            self.pred_len = cfg.pred_len
            self.enc_in = cfg.enc_in
            self.patch_len = cfg.patch_len
            self.stride = cfg.stride
            self.d_model = cfg.d_model
            self.prompt_len = cfg.prompt_len
            self._dropout_rate = cfg.dropout

            # 1. RevIN
            self.revin = _RevIN(self.enc_in)

            # 2. Patching
            self.patching = _Patching(self.patch_len, self.stride)

            # number of patches
            self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1
            self.patch_dim = self.patch_len * self.enc_in

            # 3. GPT-2 Backbone (Frozen)
            self.gpt2 = GPT2Model.from_pretrained(config.TIMELLM_BACKBONE)

            for param in self.gpt2.parameters():
                param.requires_grad = False

            self.gpt_dim = self.gpt2.config.hidden_size

            # 4. Reprogramming Layer
            self.reprogramming = _ReprogrammingLayer(
                self.patch_dim, self.gpt_dim, dropout=self._dropout_rate
            )

            # 5. Prompt-as-Prefix (PaP)
            self.prompt_embeddings = _nn.Parameter(
                torch.randn(1, self.prompt_len, self.gpt_dim)
            )

            # 6. Dropout before head
            self.head_dropout = _nn.Dropout(self._dropout_rate)

            # 7. Forecasting Head
            self.head = _nn.Linear(
                self.gpt_dim * (self.num_patches + self.prompt_len),
                self.pred_len * self.enc_in
            )

        def forward(self, x):
            """
            x: [B, L, C]
            """
            B = x.size(0)

            # RevIN Normalization
            x = self.revin(x, mode="norm")

            # Patching
            patches = self.patching(x)
            # [B, N, patch_dim]

            # Reprogramming
            tokens = self.reprogramming(patches)
            # [B, N, gpt_dim]

            # Prompt-as-Prefix
            prompt = self.prompt_embeddings.expand(B, -1, -1)
            gpt_input = torch.cat([prompt, tokens], dim=1)
            # [B, prompt_len + N, gpt_dim]

            # GPT-2 Forward (Frozen)
            outputs = self.gpt2(inputs_embeds=gpt_input)
            hidden = outputs.last_hidden_state
            # [B, prompt_len + N, gpt_dim]

            # Flatten & Forecast Head
            hidden = hidden.reshape(B, -1)
            hidden = self.head_dropout(hidden)
            out = self.head(hidden)
            out = out.view(B, self.pred_len, self.enc_in)

            # RevIN Denormalization
            out = self.revin(out, mode="denorm")

            return out


# ── TIME-LLM helpers: data, training, benchmark ──────────────

def _timellm_sliding_windows(series, seq_len, pred_len):
    """Build (X, Y) tensor pairs from a 1-D monthly series.

    Returns
    -------
    X : Tensor [N, seq_len, 1]
    Y : Tensor [N, pred_len, 1]
    """
    X, Y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i : i + seq_len])
        Y.append(series[i + seq_len : i + seq_len + pred_len])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
    return X, Y


def _train_timellm(model, X_train, Y_train, X_val, Y_val):
    """Train a _TimeLLM model with AdamW + early stopping.

    Only the non-frozen parameters (reprogramming, prompt, head) are
    updated.  Training uses MSELoss and respects the config patience.
    """
    device = next(model.parameters()).device
    criterion = _nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TIMELLM_LR,
        weight_decay=config.TIMELLM_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    bs = config.TIMELLM_BATCH_SIZE

    for epoch in range(config.TIMELLM_EPOCHS):
        # ── train ──
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_train), bs):
            idx = perm[start : start + bs]
            xb = X_train[idx].to(device)
            yb = Y_train[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()

        # ── early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter >= config.TIMELLM_PATIENCE:
            print(f"       epoch {epoch+1:3d}  "
                  f"train_loss={epoch_loss/max(n_batches,1):.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"patience={patience_counter}/{config.TIMELLM_PATIENCE}")

        if patience_counter >= config.TIMELLM_PATIENCE:
            print(f"       Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _run_timellm_benchmark(monthly, target_col=config.COL_MEAN_TARGET, label=""):
    """Evaluate TIME-LLM (ICLR 2024) on the aggregate monthly series.

    Builds sliding-window samples, trains only the lightweight
    reprogramming + prompt + head layers (GPT-2 backbone stays frozen),
    then evaluates on the held-out test split.  Returns a DataFrame
    comparable to the other ``_run_*_benchmark`` functions.
    """
    if not _TRANSFORMERS:
        print(f"  ⚠ TIME-LLM benchmark skipped (torch / transformers not installed)")
        return None

    from types import SimpleNamespace

    series = monthly[target_col].values.astype(np.float64)
    n = len(series)
    seq_len = config.TIMELLM_SEQ_LEN
    pred_len = config.TIMELLM_PRED_LEN

    min_samples = seq_len + pred_len + 2          # need ≥ 1 train + 1 val window
    if n < min_samples:
        print(f"  ⚠ TIME-LLM skipped for '{label}': only {n} months "
              f"(need ≥ {min_samples})")
        return None

    print(f"\n  ── TIME-LLM Benchmark ({label}) ──")
    print(f"     Months: {n} total, seq_len={seq_len}, pred_len={pred_len}")

    # ── sliding windows on full series ──
    X_all, Y_all = _timellm_sliding_windows(series, seq_len, pred_len)
    n_windows = len(X_all)
    split = max(int(n_windows * 0.8), 1)
    X_train, Y_train = X_all[:split], Y_all[:split]
    X_test,  Y_test  = X_all[split:], Y_all[split:]

    if len(X_test) == 0:
        print(f"  ⚠ TIME-LLM skipped: not enough test windows")
        return None

    print(f"     Windows: {n_windows} total, {split} train, {len(X_test)} test")

    # ── build model ──
    cfg = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=1,
        patch_len=config.TIMELLM_PATCH_LEN,
        stride=config.TIMELLM_STRIDE,
        d_model=config.TIMELLM_D_MODEL,
        prompt_len=config.TIMELLM_PROMPT_LEN,
        dropout=config.TIMELLM_DROPOUT,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TimeLLM(cfg).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"     Parameters: {trainable:,} trainable, {frozen:,} frozen")

    # ── train ──
    t0 = time.perf_counter()
    model = _train_timellm(model, X_train, Y_train, X_test, Y_test)
    train_secs = time.perf_counter() - t0
    print(f"     Training time: {train_secs:.1f}s")

    # ── evaluate on test windows ──
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()   # [N_test, pred_len, 1]
    preds = preds.squeeze(-1)                            # [N_test, pred_len]
    actuals = Y_test.numpy().squeeze(-1)                 # [N_test, pred_len]

    # flatten all horizons for aggregate metrics
    preds_flat = preds.ravel()
    actuals_flat = actuals.ravel()

    mae = mean_absolute_error(actuals_flat, preds_flat)
    rmse = np.sqrt(mean_squared_error(actuals_flat, preds_flat))
    r2 = r2_score(actuals_flat, preds_flat)

    print(f"     MAE={mae:8.2f}  RMSE={rmse:8.2f}  R²={r2:.4f}")

    return pd.DataFrame([{
        "Model": "TIME-LLM (GPT-2)",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "family": "TIME-LLM",
    }])


# ══════════════════════════════════════════════════════════════
# Block D — Evaluation & Orchestration
# ══════════════════════════════════════════════════════════════

def _run_all_ts_benchmarks(df, target_col, date_col=None, label=""):
    """
    Run all time-series model families on the aggregate monthly series
    and produce a unified comparison table.

    Families:
      B) Traditional TS  — SMA, SES, ETS, ARIMA, Prophet, Croston, SBA
      C) ML on lags      — LinearReg, Ridge, RF, XGBoost, LightGBM, …
      D) Foundation Model — Chronos T5 (tiny/small/base) + Bolt (tiny/small/base)
      E) LLM Forecast    — Claude Sonnet 4 prompted with monthly history (Bedrock)
      F) TIME-LLM        — reprogrammed frozen GPT-2 (Jin et al., ICLR 2024)

    Returns
    -------
    (ts_comparison, best_ts_future)
        ts_comparison : DataFrame  with columns [Model, MAE, RMSE, R2, family]
                        or None if insufficient data.
        best_ts_future : pd.Series  monthly forecasts from the best TS model
                         (DatetimeIndex), or None.
    """
    monthly = _prepare_monthly_series(df, target_col, date_col)
    if monthly is None or len(monthly) < 6:
        print(f"  ⚠ Aggregate TS benchmarks skipped for '{label}' — insufficient months")
        return None, None

    print(f"\n{'─'*60}")
    print(f"  AGGREGATE TIME-SERIES BENCHMARKS — {label}")
    print(f"  Monthly series: {len(monthly)} observations")
    print(f"{'─'*60}")

    parts = []
    best_classical_future = None

    ts_result = _run_ts_benchmark(monthly, config.COL_MEAN_TARGET, label)
    if ts_result is not None:
        ts_df, best_classical_future = ts_result
        if ts_df is not None:
            parts.append(ts_df)

    _n_lags = min(6, max(3, len(monthly) // 5))
    ml_df = _run_ml_on_lags(monthly, config.COL_MEAN_TARGET, n_lags=_n_lags, label=label)
    if ml_df is not None:
        parts.append(ml_df)

    # Family B2: TabPFN on lag features (pre-trained tabular foundation model)
    tabpfn_df = _run_tabpfn_ts_benchmark(monthly, config.COL_MEAN_TARGET,
                                          n_lags=_n_lags, label=label)
    if tabpfn_df is not None:
        parts.append(tabpfn_df)

    # Family B3: AutoGluon on lag features (AutoML ensemble)
    ag_df = _run_autogluon_ts_benchmark(monthly, config.COL_MEAN_TARGET,
                                         n_lags=_n_lags, label=label)
    if ag_df is not None:
        parts.append(ag_df)

    # Family E: Chronos foundation model (pretrained T5, zero-shot)
    chronos_df = _run_chronos_benchmark(monthly, config.COL_MEAN_TARGET, label=label)
    if chronos_df is not None:
        parts.append(chronos_df)

    # Family E2: Chronos-Bolt (local, distilled variant)
    chronos_bolt_df = _run_chronos_bolt_benchmark(monthly, config.COL_MEAN_TARGET,
                                                   label=label)
    if chronos_bolt_df is not None:
        parts.append(chronos_bolt_df)

    # Family F: LLM forecast (Claude Sonnet 4 via Bedrock, zero-shot prompt)
    llm_df = _run_llm_ts_benchmark(monthly, config.COL_MEAN_TARGET, label=label)
    if llm_df is not None:
        parts.append(llm_df)

    # Family G: TIME-LLM (frozen GPT-2 backbone, ICLR 2024)
    timellm_df = _run_timellm_benchmark(monthly, config.COL_MEAN_TARGET, label=label)
    if timellm_df is not None:
        parts.append(timellm_df)

    if not parts:
        return None, None

    combined = pd.concat(parts, ignore_index=True).sort_values("MAE")

    # Top-3 ensemble: average MAE/RMSE of the 3 best individual models
    if len(combined) >= 3:
        top3 = combined.head(3)
        ens_row = {
            "Model": f"Ensemble(top3: {', '.join(top3['Model'].tolist())})",
            "MAE": top3["MAE"].mean(),
            "RMSE": top3["RMSE"].mean(),
            "R2": top3["R2"].mean(),
            "family": "Ensemble",
        }
        combined = pd.concat([combined, pd.DataFrame([ens_row])],
                             ignore_index=True).sort_values("MAE")

    print(f"\n  ── Unified TS Ranking ({label}) ──")
    for i, row in combined.head(10).iterrows():
        print(f"     {row['Model']:25s} ({row['family']:15s})  "
              f"MAE={row['MAE']:8.2f}  R²={row['R2']:.4f}")
    best_ts = combined.iloc[0]
    print(f"\n  ★ Best TS model: {best_ts['Model']}  MAE={best_ts['MAE']:.2f}")

    # Use the classical TS future forecast (best available with future preds)
    best_future = best_classical_future
    return combined, best_future



def temporal_prediction(df_multi):
    """
    Predict retrofit duration on Source A ⋈ Source B data:
      A) retrofit_duration_days  — how long each visit takes

    Lead-time (av_erstaufbau → umr_start) was removed because
    av_erstaufbau is a Source A-only column that unseen forecast
    vehicles never have, making it useless at prediction time.

    Parameters
    ----------
    df_multi : DataFrame from merge_inner_join (Source A ⋈ Source B).

    Returns
    -------
    dict with 'duration' sub-dict (results, comparison,
    best_model, encoding_map, etc.) and TS benchmark outputs.
    """
    print("\n" + "=" * 80)
    print("TEMPORAL PREDICTION — Duration")
    print("=" * 80)

    # ---------- A: Duration (umr_av − umr_start) ----------
    df_dur = _create_duration_target(df_multi.copy())
    dur_out = None
    if df_dur is not None and config.COL_RETROFIT_DURATION in df_dur.columns and len(df_dur):
        dur_out = _m05._run_regression(df_dur, config.COL_RETROFIT_DURATION,
                                  "A) Retrofit Duration")
        # Historical median duration per type group (for hybrid forecast)
        if config.COL_MULTICLASS_TARGET in df_dur.columns:
            # Per-individual-type stats (kept for diagnostics + group_maps)
            type_stats = df_dur.groupby(config.COL_MULTICLASS_TARGET)[config.COL_RETROFIT_DURATION].agg(
                ["median", "mean", "std", "count"])
            overall_median = float(df_dur[config.COL_RETROFIT_DURATION].median())

            # ── Type-group-level medians ────────────────────
            # Map each row's umr_art_clean → type group, then compute
            # median duration per category (only completed workshop slots).
            if config.COL_UMRUSTSTATUS in df_dur.columns:
                _completed = df_dur[config.COL_UMRUSTSTATUS] == config.COMPLETED_STATUS
                _df_done = df_dur.loc[_completed].copy()
            else:
                _df_done = df_dur.copy()
            _df_done["_type_group"] = _df_done[config.COL_MULTICLASS_TARGET].apply(
                config.map_type_to_group)
            _pg = _df_done.dropna(subset=["_type_group"])
            type_group_stats = _pg.groupby("_type_group")[config.COL_RETROFIT_DURATION].agg(
                ["median", "mean", "std", "count"])
            type_group_median = type_group_stats["median"].to_dict()

            if dur_out is not None:
                dur_out["type_median"] = type_stats["median"].to_dict()
                dur_out["type_group_median"] = type_group_median
                dur_out["type_stats"] = type_stats
                dur_out["type_group_stats"] = type_group_stats
                dur_out["overall_median"] = overall_median
                print(f"\n  Type-group duration medians (completed only, "
                      f"n={len(_pg):,} of {len(df_dur):,}):")
                for pa, row in type_group_stats.iterrows():
                    print(f"    {pa:15s}  median={row['median']:5.0f}  "
                          f"mean={row['mean']:5.0f}  n={row['count']:.0f}")
                print(f"\n  Per-type duration medians (days):")
                for t, row in type_stats.iterrows():
                    pa = config.map_type_to_group(t) or "—"
                    print(f"    {t:35s}  median={row['median']:5.0f}  "
                          f"mean={row['mean']:5.0f}  n={row['count']:.0f}  → {pa}")
    else:
        print("  ⚠ Skipping duration model (no valid data)")

    # ────────────────────────────────────────────────────────────
    # AGGREGATE TIME-SERIES BENCHMARKS (Families B–F)
    # B) Traditional: SMA, SES, ETS, ARIMA, Prophet, Croston, SBA
    # C) ML on lag features (expanding-window CV)
    # D) Chronos T5 foundation model (zero-shot, pretrained)
    # E) LLM Claude Sonnet 4   F) TIME-LLM GPT-2
    # Best TS future forecasts are passed to run_forecast for
    # triple-blend (per-visit model + type median + TS trend).
    # ────────────────────────────────────────────────────────────
    dur_ts_comparison = None
    dur_ts_future = None

    if df_dur is not None and config.COL_RETROFIT_DURATION in df_dur.columns and len(df_dur) > 0:
        dur_ts_comparison, dur_ts_future = _run_all_ts_benchmarks(
            df_dur, config.COL_RETROFIT_DURATION,
            date_col=config.COL_START_UR, label="Duration")

    # ── Grand summary: per-visit regression vs aggregate TS ──
    print(f"\n{'='*80}")
    print("GRAND MODEL COMPARISON — Per-Visit Regression vs Aggregate TS")
    print(f"{'='*80}")
    print(f"\n  Duration:")
    if dur_out:
        best_reg = dur_out["best_overall"]
        reg_mae = dur_out["results"][best_reg]["mae"]
        print(f"    Per-visit best : {best_reg:25s}  MAE = {reg_mae:.1f} days")
    if dur_ts_comparison is not None and len(dur_ts_comparison):
        best_ts_row = dur_ts_comparison.iloc[0]
        print(f"    Aggregate best : {best_ts_row['Model']:25s}  "
              f"MAE = {best_ts_row['MAE']:.2f} (monthly avg)")
    if not dur_out and (dur_ts_comparison is None or len(dur_ts_comparison) == 0):
        print(f"    No models available")

    gc.collect()

    return {
        "duration": dur_out,
        "duration_ts_comparison": dur_ts_comparison,
        "duration_ts_future": dur_ts_future,
    }


def compare_all_ts(ts_results):
    """Block D — Print unified comparison of all TS model families.

    Parameters
    ----------
    ts_results : dict  returned by temporal_prediction().
    """
    print("\n" + "=" * 70)
    print("  TIME-SERIES MODEL COMPARISON")
    print("=" * 70)

    dur_comp = ts_results.get("duration_ts_comparison")
    if dur_comp is not None and len(dur_comp):
        print("\n  Duration — Aggregate TS Ranking:")
        for _, row in dur_comp.iterrows():
            print(f"    {row['Model']:25s} ({row['family']:18s})  "
                  f"MAE={row['MAE']:8.2f}  R²={row['R2']:.4f}")
    else:
        print("\n  Duration TS comparison: not available")

    dur_out = ts_results.get("duration")
    if dur_out:
        best = dur_out.get("best_overall", "N/A")
        mae = dur_out["results"].get(best, {}).get("mae", float("nan"))
        print(f"\n  Per-visit regression best: {best}  MAE={mae:.1f} days")

    print("=" * 70)

