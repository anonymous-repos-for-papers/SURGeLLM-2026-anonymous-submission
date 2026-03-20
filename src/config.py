# ============================================================
# config.py - All constants, paths, and hyperparameters
# ============================================================
import os as _os

# ── sklearn 1.6 compatibility shim ──────────────────────────
# sklearn ≥ 1.6 renamed `force_all_finite` → `ensure_all_finite` in
# check_array / check_X_y.  Older versions of LightGBM / XGBoost /
# CatBoost still pass the old name, causing TypeError.
# This shim translates the old kwarg so both names work.
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
    pass  # sklearn not installed or structure changed

# ── Suppress noisy XGBoost glibc FutureWarning ──
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*glibc.*", category=FutureWarning)

# Project root (one level above src/)
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Data source / storage
# CSVs live in the parent folder (paper/) next to code_paper/
CSV_DIR = _os.path.dirname(_PROJECT_ROOT)  # …/paper/
CSV_SOURCE_A = _os.path.join(CSV_DIR, "anonymized_data_retrofits.csv")    # retrofit orders
CSV_SOURCE_B = _os.path.join(CSV_DIR, "anonymized_data_prototypes.csv")   # vehicle master data
CSV_OUTPUT_FORECAST = _os.path.join(CSV_DIR, "forecast_output.csv")
CSV_SEP = ";"  # delimiter used in the anonymised CSVs

# Column renames applied right after loading so that the rest of the
# pipeline can use a single canonical name for each column.
CSV_RENAME_SOURCE_A = {
    "start_ur": "umr_start",
    "av_ur": "umr_av",
    "av_first_istufe": "av_erstaufbau",
    "umruststatus": "umr_status",
    "umruestart": "umr_art",
}
CSV_RENAME_SOURCE_B = {
    "vnumber": "v_nummer",
    "btnumber": "bt_nummer",
    "at_id": "at_nummer",
    "anlauf_sop_at": "sop",
    "prio_int": "prio",
}

# ── Column names (single source of truth) ────────────────────
# If a column is renamed in Source A/B, change it ONLY here.
COL_V_NUMMER = "v_nummer"                     # Vehicle number (Source A+B)
COL_JOIN_KEY_SOURCE_A = "at_nummer"           # JOIN key (Source A)
COL_JOIN_KEY_SOURCE_B = "at_nummer"           # JOIN key (Source B)
COL_VNUMBER_SOURCE_B = "v_nummer"             # Vehicle number (Source B)

# Composite join key: fallback chain for matching Source B <-> Source A.
# Uses at_nummer; if null/empty -> bt_nummer; if null/empty -> v_nummer.
COL_JOIN_KEY = "_join_key"                    # computed column name
JOIN_KEY_FALLBACK = ["at_nummer", "bt_nummer", "v_nummer"]

COL_START_UR = "umr_start"                    # Retrofit start date (Source A)
COL_AV_UR = "umr_av"                          # Retrofit end date (Source A)
COL_AV_FIRST_ISTUFE = "av_erstaufbau"         # First change identification date (Source A)
COL_ZIEL_SAB = "ziel_sab"                     # Target date (Source A)

COL_UMRUSTSTATUS = "umr_status"               # Retrofit status (Source A)
COL_UMRUESTART = "umr_art"                    # Retrofit type (Source A)

COL_RUECKGABE_SOLL = "rueckgabe_soll"         # Deadline (Source B)
COL_ANLAUF_SOP = "sop"                        # Vehicle SOP date (Source B)

COL_CAME_FOR_RETROFIT = "came_for_retrofit"   # Binary target (derived)

# Derived column names (auto-computed from above — do NOT edit manually)
COL_MULTICLASS_TARGET = COL_UMRUESTART + "_clean"   # e.g. "umr_art_clean"

# ── Additional column names used across modules ──────────────
COL_DERIVAT = "derivat"                       # Vehicle derivat (raw)
COL_DERIVAT_CLEAN = "derivat_clean"           # Vehicle derivat (cleaned)
COL_BO_CLEAN = "bo_clean"                     # Vehicle owner group (cleaned)
COL_PRIO = "prio"                             # Priority column

# Forecast / backtest derived columns
COL_PROB_COMING = "prob_coming"               # Binary predicted probability
COL_PREDICTED_TYPE = "predicted_type"         # Multiclass predicted type (forecast)
COL_TYPE_PROBABILITY = "type_probability"     # Multiclass type confidence
COL_VISIT_NUMBER = "visit_number"             # Visit index per vehicle
COL_CALIBRATION_WEIGHT = "calibration_weight" # Calibration factor per row
COL_PREDICTED_DURATION = "predicted_duration_days"  # Predicted duration
COL_DUR_MODEL = "dur_model"                   # Raw model duration prediction
COL_DUR_TYPE_MEDIAN = "dur_type_median"       # Type-median duration fallback
COL_ESTIMATED_START = "estimated_start"       # Predicted start date
COL_ESTIMATED_MONTH = "estimated_month"       # Derived period column
COL_PACKAGE_PRED = "package_pred"             # Backtest predicted type

# Temporal regression target and group-stat features
COL_RETROFIT_DURATION = "retrofit_duration_days"  # Duration target (umr_av - umr_start)
COL_TYPE_HIST_MEDIAN = "type_hist_median"         # Group stat: median duration per type
COL_TYPE_HIST_COUNT = "type_hist_count"           # Group stat: count per type
COL_DERIVAT_HIST_MEDIAN = "derivat_hist_median"   # Group stat: median duration per derivat

# Time-series columns
COL_MEAN_TARGET = "mean_target"               # Monthly aggregated target
COL_TS_MONTH = "month"                        # Month column for TS DataFrames
COL_TS_COUNT = "count"                        # Count column for TS DataFrames

# NaN fill value for categorical columns
FILLNA_CATEGORICAL = "UNKNOWN"

# Data cleansing
DATE_COLUMNS = [COL_START_UR, COL_AV_UR, COL_AV_FIRST_ISTUFE, COL_ZIEL_SAB]
SOURCE_B_DATE_COLUMNS = [COL_ANLAUF_SOP, COL_RUECKGABE_SOLL]
CANCELLED_STATUS = "Cancelled"           # adapt to your dataset's status labels
COMPLETED_STATUS = "Completed"            # adapt to your dataset's status labels

# Invalid event-type combinations to remove from training data.
# Populate with domain-specific invalid combinations from your dataset.
INVALID_UMR_ART = set()  # e.g. {"TypeA, TypeB", "TypeC, TypeD"}

# Feature engineering
# Features are now auto-detected: _clean columns (categorical) + numeric columns.
# IDs, targets, dates, and raw versions of _clean columns are excluded automatically.

# Train / test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Models
N_ESTIMATORS = 300
SMOTE_STRATEGY = 0.6          # minority → 60% of majority (was 0.5; higher helps multiclass)
SMOTE_K_NEIGHBORS = 5
HP_CV_FOLDS = 5               # RandomizedSearchCV fallback CV
BASELINE_CV_FOLDS = 3         # baseline & SMOTE evaluation CV
OPTUNA_CV_FOLDS = 3           # Optuna internal cross-validation
STACKING_CV_FOLDS = 3         # stacking / voting ensemble internal CV
FEATURE_SELECTION_TOP_N = 15
SHAP_PREFILTER_TOP_N = 20       # SHAP pre-filter: keep top N features for Tuning/SMOTE
ENSEMBLE_TOP_N = 3

# Hyperparameter tuning (Optuna Bayesian optimisation)
OPTUNA_N_TRIALS = 12          # TPE trials per model (binary — TPE converges by ~10)
OPTUNA_TIMEOUT = 300          # seconds per model (5 min safety cap)
OPTUNA_MC_N_TRIALS = 25       # trials per model (multiclass — more classes need more search)
OPTUNA_TEMP_N_TRIALS = 20     # trials per model (temporal regression)
HP_N_ITER = 30                # fallback RandomizedSearchCV iterations

# Models to skip during Optuna tuning (slow or low-value)
OPTUNA_SKIP_BINARY = {"SVC", "MLP", "AdaBoost"}       # SVC O(n²), MLP slow, Ada weak
OPTUNA_SKIP_MULTICLASS = {"AdaBoost", "SVC", "MLP"}   # AdaBoost F1w≈0.28
OPTUNA_SKIP_TEMPORAL = {"SVR", "MLP", "AdaBoost"}     # SVR O(n²), MLP slow, Ada weak

# Caching
CACHE_DIR = _os.path.join(_PROJECT_ROOT, ".cache")
CACHE_FINGERPRINT_FILE = "_data_fingerprint.txt"  # filename inside CACHE_DIR

# ── CatBoost training directory ──────────────────────────────
CATBOOST_TRAIN_DIR = "/tmp/catboost_info"

# ── Multiclass focal loss ────────────────────────────────────
FOCAL_GAMMA = 2.0                             # focal-loss γ (down-weight easy examples)

# ── Temporal duration binning ────────────────────────────────
DURATION_BIN_EDGES = [0, 7, 30, float("inf")]  # day boundaries
DURATION_BIN_LABELS = ["Short (<7d)", "Medium (7-30d)", "Long (>30d)"]

# ── Forecast parameters ──────────────────────────────────────
FORECAST_THRESHOLD = 0.05                     # Recall-oriented prob cutoff (overrides F1-optimal)
MIN_TYPE_PROBABILITY = 0.02                   # Drop visits with prob below this
FORECAST_HORIZON_MONTHS = 24                  # Filter to now → +N months
FALLBACK_OFFSET_DAYS = 90                     # Default anchor when no deadline
TS_MIN_R2 = 0.0                               # TS quality gate: must beat flat-mean
TS_MAX_MAPE = 0.50                            # TS quality gate: relative MAE < 50%

# Triple/dual blend weights (model + type-median + time-series)
BLEND_W_MODEL = 0.30                          # Weight for ML model prediction
BLEND_W_TYPE_MEDIAN = 0.40                    # Weight for type-median
BLEND_W_TS = 0.30                             # Weight for TS forecast
BLEND_W_DUAL = 0.30                           # Fallback: model weight when no TS

# Row-level calibration tolerance
CAL_LO = 0.50                                 # Calibration lower bound
CAL_HI = 1.50                                 # Calibration upper bound

# ── Base event types (for per-type subplots) ────────────────
# Populate with the distinct class labels from your multiclass target.
BASE_RETROFIT_TYPES = []  # e.g. ["TypeA", "TypeB", "TypeC"]

# ── Higher-level event grouping ──────────────────────────────
# Optional: map multiclass labels to coarser groups for duration estimation.
# Populate TPKHW_COMPONENTS and the mapping function with your domain logic.
TPKHW_COMPONENTS = set()  # e.g. {"TypeB", "TypeD", "TypeE"}


def map_type_to_group(event_type_clean: str) -> str | None:
    """Map an event_type_clean value to a higher-level group.

    Implement your own domain-specific grouping rules here.
    Returns None if the type doesn't match any category.
    """
    # Example skeleton — replace with your rules:
    # parts = {p.strip() for p in event_type_clean.split(",")}
    # if "SomeType" in parts: return "GroupA"
    # if parts <= TPKHW_COMPONENTS: return "GroupB"
    return None


# ── Data validation ──────────────────────────────────────────
VALIDATION_NAN_THRESHOLD = 0.20       # max NaN ratio per column
VALIDATION_DRIFT_THRESHOLD = 0.10     # max mean-shift ratio vs previous batch
VALIDATION_MIN_ROWS = 50              # minimum rows in a valid dataset
VALIDATION_DUPLICATE_WARN = True      # warn on duplicate rows

# Required Source A columns (minimum schema after cleansing)
REQUIRED_COLUMNS_SOURCE_A = [
    COL_V_NUMMER, COL_JOIN_KEY_SOURCE_A,
    COL_START_UR, COL_AV_UR,
    COL_UMRUSTSTATUS, COL_UMRUESTART,
]
# Required Source B columns
REQUIRED_COLUMNS_SOURCE_B = [
    COL_VNUMBER_SOURCE_B, COL_JOIN_KEY_SOURCE_B,
    COL_ANLAUF_SOP, COL_RUECKGABE_SOLL,
]

# ── Chronos foundation model (Amazon, TMLR 2024) ────────────
CHRONOS_MODEL_SIZES = ["tiny", "small", "base"]   # T5 variants to benchmark
CHRONOS_BOLT_SIZES  = ["tiny", "small", "base"]   # Bolt (distilled) variants
CHRONOS_NUM_SAMPLES = 20                           # probabilistic forecast samples

# ── LLM / Amazon Bedrock settings ─────────────────────────────
# All LLM calls go through Amazon Bedrock (no OpenAI API key needed).
# Models available via Bedrock in the configured region.
BEDROCK_REGION          = "eu-west-1"
LLM_EMBEDDING_MODEL     = "amazon.titan-embed-text-v2:0" # 1024 dims (Bedrock Titan V2)
LLM_EMBEDDING_DIMS      = 1024                           # Titan V2 output vector size
LLM_CHAT_MODEL          = "eu.anthropic.claude-sonnet-4-6"  # Claude Sonnet 4 via Bedrock (cross-region inference profile)
LLM_TEMPERATURE         = 0.0                            # deterministic output
LLM_TS_N_LAST           = 36                             # months of history to send
LLM_EMBEDDING_BATCH_SIZE = 50                            # Titan: 1 row/call; keep moderate
LLM_EMBED_MAX_SAMPLES   = 5_000                          # subsample train/test for LLM-Embed
LLM_EMBED_WORKERS       = 10                             # concurrent Bedrock API calls
LLM_PROMPTED_MAX_SAMPLES = 200                           # subsample test for LLM-Prompted (1 call/row)
LLM_PROMPTED_FEW_SHOT_K  = 3                             # few-shot examples per class in prompt

# ── LLM enhancement settings ─────────────────────────────────
LLM_SELF_CONSISTENCY_N   = 3       # majority-vote runs per row (improvement #5)
LLM_CONSISTENCY_TEMP     = 0.7     # temperature for self-consistency samples
LLM_TOP_FEATURES         = 15      # max features to include in LLM prompt (#3)
LLM_CALIBRATION_SAMPLES  = 50      # held-out calibration rows for Platt scaling (#7)
LLM_STACKED_MAX_SAMPLES  = 500     # max train rows for stacking model (#8)
MIN_TEST_SAMPLES_FOR_BEST = 500    # models tested on fewer rows cannot be crowned "best"

# ── TIME-LLM (Jin et al., ICLR 2024) ────────────────────────
# Reprogramming framework: frozen LLM backbone + learned input/output layers
TIMELLM_BACKBONE       = "openai-community/gpt2"  # frozen backbone (124 M)
TIMELLM_PATCH_LEN      = 4        # patch length in time steps (months)
TIMELLM_STRIDE         = 2        # sliding stride for overlapping patches
TIMELLM_D_MODEL        = 32       # patch embedding hidden dimension (d_m)
TIMELLM_N_HEADS        = 8        # cross-attention heads for reprogramming (K)
TIMELLM_N_PROTOTYPES   = 100      # text prototypes V' (≪ vocab V)
TIMELLM_DROPOUT        = 0.1      # dropout in reprogramming layer
TIMELLM_EPOCHS         = 30       # training epochs (early-stopping patience=10)
TIMELLM_LR             = 1e-3     # AdamW learning rate
TIMELLM_WEIGHT_DECAY   = 1e-4     # AdamW weight decay
TIMELLM_PATIENCE       = 10       # early-stopping patience (epochs)
TIMELLM_BATCH_SIZE     = 16       # mini-batch size
TIMELLM_PROMPT_LEN     = 5        # learnable prompt-as-prefix tokens
TIMELLM_SEQ_LEN        = 12       # input context window (months)
TIMELLM_PRED_LEN       = 3        # forecast horizon   (months)

# ── Output table schema ──────────────────────────────────────
# Column order for the output table.
# Populate with your dataset's column names.
OUTPUT_BASE_COLUMNS = []  # list of column names from your source data
# Extra forecast-metadata columns (only filled for forecast rows)
OUTPUT_FORECAST_COLUMNS = [
    COL_PROB_COMING, COL_TYPE_PROBABILITY, COL_VISIT_NUMBER,
    COL_CALIBRATION_WEIGHT, "forecast_binary_approach",
    "forecast_binary_model", "forecast_multiclass_model",
    "forecast_duration_model",
    "forecast_threshold", "forecast_timestamp",
]

# ── Forecast row mapping ─────────────────────────────────────
# Label written into event_status for forecast rows
FORECAST_STATUS_LABEL = "Forecast"

# Source B columns that map 1:1 to source A columns.
FORECAST_DIRECT_COLUMNS = []  # populate with your column names

# Source B columns that map to a differently named source A column.
FORECAST_RENAME_COLUMNS = {}  # e.g. {"output_col": "input_col"}

# Source A columns that are always NULL for forecast rows.
FORECAST_NULL_COLUMNS = []  # populate with your column names

# ── Group → typical event-type components ────────────────────
# Used during multiclass K-expansion: classes whose components
# DON'T overlap with the group's typical set get penalised
# (probability × BO_ATYPICAL_TYPE_PENALTY).
# Groups not listed here → no constraint.
# Populate with your domain's group-to-type mapping.
BO_TYPICAL_TYPES = {}  # e.g. {"DeptA": {"TypeX", "TypeY"}, ...}
BO_ATYPICAL_TYPE_PENALTY = 0.8   # prob × this for types not in BO's typical set

# ── Priority ↔ type mismatch penalties ───────────────────────
# Penalise unlikely priority–type combinations during K-expansion.
# Populate with your domain's mismatch rules.
PRIO_TYPE_MISMATCH_PENALTY = 0.3   # moderate penalty (prob × 0.3)
PRIO_TYPE_MISMATCHES = {}  # e.g. {"3": {"TypeX"}, "1": {"TypeY"}}

# ── AutoGluon AutoML Benchmark ───────────────────────────────
AUTOML_PRESETS = "best_quality"            # "best_quality" | "high_quality" | "medium_quality"
AUTOML_TIME_LIMIT_BINARY = 600            # seconds (10 min)
AUTOML_TIME_LIMIT_MULTICLASS = 600        # seconds (10 min)
AUTOML_TIME_LIMIT_TEMPORAL = 600          # seconds (10 min)
AUTOML_SAVE_DIR = "/tmp/autogluon"        # ephemeral — models not persisted
